"""Tests for runpod_pod.py shared pod lifecycle module."""

import json
import os
import pytest
from runpod_pod import RunPodSession


class TestRunPodSessionState:
    """Test pod state file read/write."""

    def test_write_and_read_state(self, tmp_path):
        session = RunPodSession(str(tmp_path))
        session.pod_id = "pod-abc123"
        session.base_url = "https://pod-abc123-8188.proxy.runpod.net"
        session.ssh_host = "1.2.3.4"
        session.ssh_port = 12345
        session.gpu_hourly_rate = 0.34
        session.pod_start_time = 1000000.0
        session.write_state()

        # Verify file exists and is valid JSON
        state_path = os.path.join(str(tmp_path), "runpod_pod.json")
        assert os.path.exists(state_path)
        with open(state_path) as f:
            state = json.load(f)
        assert state["pod_id"] == "pod-abc123"
        assert state["base_url"] == "https://pod-abc123-8188.proxy.runpod.net"
        assert state["ssh_host"] == "1.2.3.4"
        assert state["ssh_port"] == 12345

        # Read back into a new session
        session2 = RunPodSession(str(tmp_path))
        assert session2.read_state() is True
        assert session2.pod_id == "pod-abc123"
        assert session2.base_url == "https://pod-abc123-8188.proxy.runpod.net"
        assert session2.ssh_host == "1.2.3.4"
        assert session2.ssh_port == 12345

    def test_read_state_missing(self, tmp_path):
        session = RunPodSession(str(tmp_path))
        assert session.read_state() is False
        assert session.pod_id is None

    def test_remove_state(self, tmp_path):
        session = RunPodSession(str(tmp_path))
        session.pod_id = "pod-xyz"
        session.base_url = "https://pod-xyz-8188.proxy.runpod.net"
        session.write_state()
        assert os.path.exists(session.state_path)

        session.remove_state()
        assert not os.path.exists(session.state_path)

    def test_remove_state_missing(self, tmp_path):
        """remove_state is a no-op when no state file exists."""
        session = RunPodSession(str(tmp_path))
        session.remove_state()  # should not raise

    def test_state_path(self, tmp_path):
        session = RunPodSession(str(tmp_path))
        assert session.state_path == os.path.join(str(tmp_path), "runpod_pod.json")


class TestRunPodSessionCleanup:
    """Test cleanup logic (keep-pod vs terminate)."""

    def test_keep_pod_clean_exit_writes_state(self, tmp_path):
        session = RunPodSession(str(tmp_path), keep_pod=True)
        session.pod_id = "pod-keep"
        session.base_url = "https://pod-keep-8188.proxy.runpod.net"
        session.pod_start_time = 1000000.0
        session._clean_exit = True

        # _cleanup should write state, not terminate
        session._cleanup()
        assert os.path.exists(session.state_path)
        assert session.pod_id == "pod-keep"  # not cleared

    def test_no_keep_pod_calls_terminate(self, tmp_path, monkeypatch):
        session = RunPodSession(str(tmp_path), keep_pod=False)
        session.pod_id = "pod-term"
        session.base_url = "https://pod-term-8188.proxy.runpod.net"
        session.pod_start_time = 1000000.0

        terminated = []
        def mock_terminate(self_inner):
            terminated.append(self_inner.pod_id)
            self_inner.pod_id = None
        monkeypatch.setattr(RunPodSession, "terminate", mock_terminate)

        session._cleanup()
        assert terminated == ["pod-term"]

    def test_keep_pod_dirty_exit_terminates(self, tmp_path, monkeypatch):
        session = RunPodSession(str(tmp_path), keep_pod=True)
        session.pod_id = "pod-crash"
        session.base_url = "https://pod-crash-8188.proxy.runpod.net"
        session.pod_start_time = 1000000.0
        # _clean_exit is False by default (crash/error case)

        terminated = []
        def mock_terminate(self_inner):
            terminated.append(self_inner.pod_id)
            self_inner.pod_id = None
        monkeypatch.setattr(RunPodSession, "terminate", mock_terminate)

        session._cleanup()
        assert terminated == ["pod-crash"]


class TestKeepPodTTL:
    """A kept-alive pod must always carry a deadline.

    A dress-rehearsal render finished in 2.5hrs and then billed for another
    10.5hrs ($3.40, more than the render itself) because --keep-pod meant
    "keep forever" and nothing local was still running to notice.
    """

    def test_keep_pod_false_has_no_ttl(self, tmp_path):
        s = RunPodSession(str(tmp_path), keep_pod=False)
        assert s.keep_pod is False
        assert s.keep_pod_minutes == 0

    def test_keep_pod_true_gets_the_default_ttl(self, tmp_path):
        from runpod_pod import KEEP_POD_DEFAULT_MINUTES
        s = RunPodSession(str(tmp_path), keep_pod=True)
        assert s.keep_pod is True
        assert s.keep_pod_minutes == KEEP_POD_DEFAULT_MINUTES
        assert s.keep_pod_minutes > 0, "keeping a pod with no deadline is the bug"

    def test_keep_pod_accepts_an_explicit_duration(self, tmp_path):
        s = RunPodSession(str(tmp_path), keep_pod=90)
        assert s.keep_pod is True
        assert s.keep_pod_minutes == 90

    def test_arm_self_destruct_uses_pod_local_id_not_an_api_key(self, tmp_path, monkeypatch):
        # The pod must remove *itself* via the injected RUNPOD_POD_ID -- copying
        # the account API key onto a community-cloud host is not acceptable.
        s = RunPodSession(str(tmp_path), keep_pod=30)
        s.ssh_host, s.ssh_port = "1.2.3.4", 22
        sent = []

        class R:
            returncode = 0
        monkeypatch.setattr(s, "ssh_cmd", lambda cmd, timeout=60: (sent.append(cmd), R())[1])

        assert s.arm_self_destruct(30) is True
        armed = sent[-1]
        assert "RUNPOD_POD_ID" in armed
        assert "runpodctl remove pod" in armed
        assert "sleep 1800" in armed
        assert not any("RUNPOD_API_KEY" in c for c in sent)

    def test_arm_self_destruct_disarms_first_so_reapers_do_not_stack(self, tmp_path, monkeypatch):
        s = RunPodSession(str(tmp_path), keep_pod=30)
        s.ssh_host, s.ssh_port = "1.2.3.4", 22
        sent = []

        class R:
            returncode = 0
        monkeypatch.setattr(s, "ssh_cmd", lambda cmd, timeout=60: (sent.append(cmd), R())[1])

        s.arm_self_destruct(30)
        assert "kill" in sent[0], "existing reaper must be cleared before arming a new one"

    def test_arm_self_destruct_without_ssh_reports_failure(self, tmp_path):
        s = RunPodSession(str(tmp_path), keep_pod=30)
        s.ssh_host = None
        assert s.arm_self_destruct(30) is False

    def test_zero_minutes_never_arms(self, tmp_path):
        s = RunPodSession(str(tmp_path), keep_pod=30)
        s.ssh_host, s.ssh_port = "1.2.3.4", 22
        assert s.arm_self_destruct(0) is False


class TestPodSetupRetry:
    """Pod provisioning must re-provision on failure, not stall the run.

    Across ten attempts in one rehearsal session, three pods died on CUDA init
    before ComfyUI started and one attempt found no capacity on any GPU type.
    Each stalled the run until a human reissued the command.
    """

    def test_succeeds_first_time_without_retrying(self, tmp_path):
        s = RunPodSession(str(tmp_path))
        calls = []
        s.with_setup_retry(lambda: calls.append(1))
        assert calls == [1]

    def test_retries_on_pod_setup_error_then_succeeds(self, tmp_path, monkeypatch):
        import runpod_pod
        monkeypatch.setattr(runpod_pod.time, "sleep", lambda *_: None)
        s = RunPodSession(str(tmp_path))
        calls = []

        def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise runpod_pod.PodSetupError("CUDA unknown error")
        s.with_setup_retry(flaky)
        assert len(calls) == 3, "should retry until a healthy pod comes up"

    def test_clears_stale_pod_state_between_attempts(self, tmp_path, monkeypatch):
        import runpod_pod
        monkeypatch.setattr(runpod_pod.time, "sleep", lambda *_: None)
        s = RunPodSession(str(tmp_path))
        seen_ids = []

        def flaky():
            seen_ids.append(s.pod_id)
            if len(seen_ids) < 2:
                s.pod_id = "dead-pod"
                raise runpod_pod.PodSetupError("ComfyUI did not come up")
        s.with_setup_retry(flaky)
        # the retry must not inherit the dead pod's id, or it would try to
        # reconnect to a terminated pod instead of provisioning a new one
        assert seen_ids[1] is None

    def test_gives_up_after_the_attempt_limit(self, tmp_path, monkeypatch):
        import runpod_pod
        import pytest
        monkeypatch.setattr(runpod_pod.time, "sleep", lambda *_: None)
        s = RunPodSession(str(tmp_path))
        calls = []

        def always_fails():
            calls.append(1)
            raise runpod_pod.PodSetupError("no capacity")
        with pytest.raises(runpod_pod.PodSetupError):
            s.with_setup_retry(always_fails, attempts=3)
        assert len(calls) == 3

    def test_non_setup_errors_are_not_retried(self, tmp_path):
        import pytest
        s = RunPodSession(str(tmp_path))
        calls = []

        def bug():
            calls.append(1)
            raise ValueError("a real bug, not a flaky host")
        with pytest.raises(ValueError):
            s.with_setup_retry(bug)
        assert len(calls) == 1, "retrying a genuine bug just wastes pods"


class TestCloudFallback:
    """Community stock genuinely runs out; escalate rather than fail the run.

    A rehearsal slice found no capacity across all five 48GB community types
    for ~15 minutes and gave up. Secure cloud is RunPod's own datacentres --
    pricier, far more reliably available.
    """

    def test_defaults_to_community_first_then_secure(self):
        import runpod_pod
        assert runpod_pod.CLOUD_TYPES[0] == "COMMUNITY"
        assert "SECURE" in runpod_pod.CLOUD_TYPES

    def test_cheapest_cloud_is_exhausted_before_escalating(self):
        import runpod_pod
        gpus = [("A", 0.33), ("B", 0.35)]
        attempts = [(c, g, r) for c in runpod_pod.CLOUD_TYPES for g, r in gpus]
        clouds_in_order = [c for c, _, _ in attempts]
        first = runpod_pod.CLOUD_TYPES[0]
        # every attempt on the cheap cloud must come before any on the pricier
        assert clouds_in_order[:len(gpus)] == [first] * len(gpus)

    def test_old_constant_still_resolves(self):
        import runpod_pod
        assert runpod_pod.CLOUD_TYPE == runpod_pod.CLOUD_TYPES[0]


class TestUniquePodNaming:
    """Each session names its pod uniquely so a sweep can tell whose is whose.

    Without a unique suffix every lossy pod is called the same thing, and
    "terminate the leftover named lossy-comfyui" would take down a concurrent
    run's active render.
    """

    def test_each_session_gets_its_own_name(self, tmp_path):
        import runpod_pod
        a = RunPodSession(str(tmp_path))
        b = RunPodSession(str(tmp_path))
        assert a.pod_name != b.pod_name
        assert a.pod_name.startswith(runpod_pod.POD_NAME_PREFIX)
        assert b.pod_name.startswith(runpod_pod.POD_NAME_PREFIX)

    def test_name_survives_a_state_round_trip(self, tmp_path):
        session = RunPodSession(str(tmp_path))
        session.pod_id = "pod-1"
        session.base_url = "https://pod-1-8188.proxy.runpod.net"
        original = session.pod_name
        session.write_state()

        resumed = RunPodSession(str(tmp_path))
        assert resumed.read_state() is True
        assert resumed.pod_name == original

    def test_state_without_a_name_falls_back_to_the_bare_prefix(self, tmp_path):
        """Pods created before unique naming were all called just the prefix."""
        import runpod_pod
        legacy = {"pod_id": "pod-old", "base_url": "https://x", "ssh_host": None,
                  "ssh_port": None, "gpu_hourly_rate": 0.34, "start_time": 1.0}
        state_path = tmp_path / "runpod_pod.json"
        state_path.write_text(json.dumps(legacy))

        session = RunPodSession(str(tmp_path))
        assert session.read_state() is True
        assert session.pod_name == runpod_pod.POD_NAME_PREFIX


class _FakeRunpod:
    """Stand-in for the runpod module, recording what got terminated."""

    def __init__(self, pods):
        self._pods = pods
        self.api_key = None
        self.terminated = []

    def get_pods(self):
        return self._pods

    def terminate_pod(self, pod_id):
        self.terminated.append(pod_id)


@pytest.fixture
def fake_runpod(monkeypatch):
    """Install a fake runpod module and an API key, and hand back a factory."""
    import sys
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    monkeypatch.delenv("LOSSY_RUNPOD_SWEEP", raising=False)

    def install(pods):
        fake = _FakeRunpod(pods)
        monkeypatch.setitem(sys.modules, "runpod", fake)
        return fake
    return install


class TestStrayPodSweep:
    """A pod bills until something terminates it, so verify what's left running.

    terminate() removes the id it recorded and only warns when that fails. A
    run was observed reporting "pod not found to terminate" while a pod under
    a different id was still up and billing -- nothing noticed, because once
    the process exits there is nobody left to look.
    """

    def _session(self, tmp_path, owned="pod-ours"):
        session = RunPodSession(str(tmp_path))
        session._owned_pod_ids.add(owned)
        return session

    def test_terminates_our_pod_when_the_recorded_id_went_stale(self, tmp_path, fake_runpod):
        """The whole point: identify by name, because the id is what drifted."""
        session = self._session(tmp_path)
        fake = fake_runpod([{"id": "pod-drifted", "name": session.pod_name,
                             "desiredStatus": "RUNNING"}])

        session.sweep_stray_pods()
        assert fake.terminated == ["pod-drifted"]

    def test_terminates_a_pod_we_own_by_id(self, tmp_path, fake_runpod):
        import runpod_pod
        session = self._session(tmp_path)
        fake = fake_runpod([{"id": "pod-ours", "name": runpod_pod.POD_NAME_PREFIX,
                             "desiredStatus": "RUNNING"}])

        session.sweep_stray_pods()
        assert fake.terminated == ["pod-ours"]

    def test_leaves_a_concurrent_runs_pod_alone(self, tmp_path, fake_runpod):
        import runpod_pod
        session = self._session(tmp_path)
        other = f"{runpod_pod.POD_NAME_PREFIX}-deadbeef"
        fake = fake_runpod([{"id": "pod-theirs", "name": other, "desiredStatus": "RUNNING"}])

        found = session.sweep_stray_pods()
        assert fake.terminated == []
        assert found == ["pod-theirs"]  # reported, not killed

    def test_opt_in_terminates_everyone_elses_too(self, tmp_path, fake_runpod, monkeypatch):
        import runpod_pod
        session = self._session(tmp_path)
        other = f"{runpod_pod.POD_NAME_PREFIX}-deadbeef"
        fake = fake_runpod([{"id": "pod-theirs", "name": other, "desiredStatus": "RUNNING"}])
        monkeypatch.setenv("LOSSY_RUNPOD_SWEEP", "1")

        session.sweep_stray_pods()
        assert fake.terminated == ["pod-theirs"]

    def test_ignores_pods_that_are_not_ours_by_name_at_all(self, tmp_path, fake_runpod):
        session = self._session(tmp_path)
        fake = fake_runpod([{"id": "pod-unrelated", "name": "someone-elses-thing",
                             "desiredStatus": "RUNNING"}])

        assert session.sweep_stray_pods() == []
        assert fake.terminated == []

    def test_ignores_already_exited_pods(self, tmp_path, fake_runpod):
        session = self._session(tmp_path)
        fake = fake_runpod([{"id": "pod-gone", "name": session.pod_name,
                             "desiredStatus": "EXITED"}])

        assert session.sweep_stray_pods() == []
        assert fake.terminated == []

    def test_a_session_that_never_provisioned_does_not_call_the_api(self, tmp_path, fake_runpod):
        """Nothing was created, so nothing can have escaped -- and no network."""
        session = RunPodSession(str(tmp_path))
        fake = fake_runpod([{"id": "pod-x", "name": session.pod_name,
                             "desiredStatus": "RUNNING"}])

        assert session.sweep_stray_pods() == []
        assert fake.terminated == []

    def test_an_api_failure_is_reported_not_raised(self, tmp_path, monkeypatch):
        import sys

        class _Broken:
            api_key = None
            def get_pods(self):
                raise RuntimeError("network down")

        monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
        monkeypatch.setitem(sys.modules, "runpod", _Broken())
        session = self._session(tmp_path)

        assert session.sweep_stray_pods() == []  # cleanup must not crash the run

    def test_cleanup_sweeps_after_terminating(self, tmp_path, fake_runpod, monkeypatch):
        session = RunPodSession(str(tmp_path), keep_pod=False)
        session.pod_id = "pod-recorded"
        session._owned_pod_ids.add("pod-recorded")
        session.pod_start_time = 1000000.0
        fake = fake_runpod([{"id": "pod-drifted", "name": session.pod_name,
                             "desiredStatus": "RUNNING"}])

        def mock_terminate(self_inner):
            self_inner.pod_id = None
        monkeypatch.setattr(RunPodSession, "terminate", mock_terminate)

        session._cleanup()
        assert fake.terminated == ["pod-drifted"]

    def test_a_deliberately_kept_pod_is_not_swept_away(self, tmp_path, fake_runpod, monkeypatch):
        session = RunPodSession(str(tmp_path), keep_pod=True)
        session.pod_id = "pod-keep"
        session._owned_pod_ids.add("pod-keep")
        session.pod_start_time = 1000000.0
        session._clean_exit = True
        monkeypatch.setattr(RunPodSession, "arm_self_destruct", lambda self, m: True)
        fake = fake_runpod([{"id": "pod-keep", "name": session.pod_name,
                             "desiredStatus": "RUNNING"}])

        session._cleanup()
        assert fake.terminated == []

    def test_cleanup_only_sweeps_once(self, tmp_path, fake_runpod, monkeypatch):
        """atexit and the signal handler both call _cleanup."""
        session = RunPodSession(str(tmp_path), keep_pod=False)
        session.pod_id = "pod-recorded"
        session._owned_pod_ids.add("pod-recorded")
        session.pod_start_time = 1000000.0
        fake = fake_runpod([{"id": "pod-drifted", "name": session.pod_name,
                             "desiredStatus": "RUNNING"}])
        monkeypatch.setattr(RunPodSession, "terminate",
                            lambda self_inner: setattr(self_inner, "pod_id", None))

        session._cleanup()
        session._cleanup()
        assert fake.terminated == ["pod-drifted"]


class TestPooledHttpClient:
    """Every ComfyUI call shares one pooled client.

    Each call used to be a bare `httpx.get`/`httpx.post`, opening and dropping
    a fresh TCP+TLS connection. Generation polls /history every ~2s, so the
    full run's first 50 shots opened 1,249 connections -- ~48,000 extrapolated
    across 2069 -- and that churn exhausted the local ephemeral port range,
    losing shot 56 to `[Errno 49] Can't assign requested address`.
    """

    def test_the_same_client_is_handed_out_every_time(self, tmp_path):
        session = RunPodSession(str(tmp_path))
        assert session.http is session.http

    def test_the_client_pools_and_keeps_connections_alive(self, tmp_path):
        session = RunPodSession(str(tmp_path))
        pool = session.http._transport._pool
        # Keepalive is the whole point: without it every request still costs a
        # fresh connection even though the client is shared.
        assert pool._max_keepalive_connections > 0
        assert pool._max_connections > 0

    def test_closing_releases_the_client(self, tmp_path):
        session = RunPodSession(str(tmp_path))
        first = session.http
        session.close_http()
        assert first.is_closed
        assert session.http is not first  # a later call rebuilds rather than reusing a closed one

    def test_closing_twice_is_harmless(self, tmp_path):
        session = RunPodSession(str(tmp_path))
        _ = session.http
        session.close_http()
        session.close_http()

    def test_closing_without_ever_making_one_is_harmless(self, tmp_path):
        RunPodSession(str(tmp_path)).close_http()

    def test_cleanup_releases_the_pool(self, tmp_path):
        session = RunPodSession(str(tmp_path))
        client = session.http
        session._cleanup()  # no pod_id, so this is the pure teardown path
        assert client.is_closed


class TestOutOfFundsIsNotACapacityProblem:
    """Running out of money must not look like a GPU shortage.

    RunPod reports "balance too low" through the same exception path as "no
    instances available", so a render that simply ran out of credit reported
    "no capacity on any of 5 GPU types across 2 cloud(s)" and retried four
    times against something no retry can fix. It happened 16 hours into a
    690-shot run when the balance hit zero and the pod was reclaimed.
    """

    def test_a_balance_error_is_recognised(self):
        from runpod_pod import _is_out_of_funds

        assert _is_out_of_funds(
            Exception("Your account balance is too low to rent a pod. Please add funds"))

    def test_a_capacity_error_is_not(self):
        from runpod_pod import _is_out_of_funds

        assert not _is_out_of_funds(
            Exception("There are no longer any instances available with the requested specifications"))

    def test_it_is_not_swallowed_by_the_setup_retry(self):
        """PodSetupError is retried four times; this must not be."""
        from runpod_pod import OutOfFundsError, PodSetupError

        assert not issubclass(OutOfFundsError, PodSetupError)

    def test_the_message_says_what_to_do(self):
        from runpod_pod import OutOfFundsError

        assert "OutOfFunds" in OutOfFundsError.__name__


class TestPodLiveness:
    """A pod that has gone away must be detectable, so the run can replace it."""

    def test_no_pod_id_means_not_alive(self, tmp_path):
        assert RunPodSession(str(tmp_path)).pod_alive() is False

    def test_forget_pod_clears_every_trace(self, tmp_path):
        s = RunPodSession(str(tmp_path))
        s.pod_id = "dead"; s._owned_pod_ids.add("dead")
        s.base_url = "http://x"; s.ssh_host = "1.2.3.4"; s.ssh_port = 22
        s.write_state()

        s.forget_pod()

        assert s.pod_id is None and s.base_url is None and s.ssh_host is None
        assert "dead" not in s._owned_pod_ids
        assert not os.path.exists(s.state_path)  # next run provisions, not reconnects

    def test_a_flaky_status_call_reports_alive(self, tmp_path, monkeypatch):
        """Doubt must never trigger a needless re-provision."""
        import runpod_pod

        s = RunPodSession(str(tmp_path))
        s.pod_id = "p1"
        monkeypatch.setenv("RUNPOD_API_KEY", "k")

        class Boom:
            api_key = None
            @staticmethod
            def get_pod(_):
                raise RuntimeError("network blip")

        monkeypatch.setitem(__import__("sys").modules, "runpod", Boom)
        assert s.pod_alive() is True

    def test_an_exited_pod_is_not_alive(self, tmp_path, monkeypatch):
        s = RunPodSession(str(tmp_path))
        s.pod_id = "p1"
        monkeypatch.setenv("RUNPOD_API_KEY", "k")

        class Gone:
            api_key = None
            @staticmethod
            def get_pod(_):
                return {"desiredStatus": "EXITED"}

        monkeypatch.setitem(__import__("sys").modules, "runpod", Gone)
        assert s.pod_alive() is False

    def test_a_pod_with_no_runtime_ports_is_not_alive(self, tmp_path, monkeypatch):
        s = RunPodSession(str(tmp_path))
        s.pod_id = "p1"
        monkeypatch.setenv("RUNPOD_API_KEY", "k")

        class Dead:
            api_key = None
            @staticmethod
            def get_pod(_):
                return {"desiredStatus": "RUNNING", "runtime": None}

        monkeypatch.setitem(__import__("sys").modules, "runpod", Dead)
        assert s.pod_alive() is False
