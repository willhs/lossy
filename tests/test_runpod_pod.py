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
