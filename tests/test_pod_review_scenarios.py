"""Scenarios from the sceptical review of the pod journal and exit sweep
(research 0026, "Review, 21 Nov 2026"), run against a fake RunPod account.

The first group pins behaviour the review found sound (so a later change that
breaks it fails). The second group was the OPEN findings (each was `xfail(strict)`);
F2, F3, F4, F8 and F9 are fixed and their markers removed.
Every assertion is about pods still live in the fake account.
"""

import os
import types

import pytest

import pod_journal
import runpod_pod
from runpod_pod import PodSetupError, RunPodSession, UnsafeToAllocateError
from test_runpod_pod import _Clock, _FleetRunpod, _Killed


@pytest.fixture
def fleet(monkeypatch):
    import sys

    f = _FleetRunpod()
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    monkeypatch.delenv("LOSSY_RUNPOD_SWEEP", raising=False)
    monkeypatch.setitem(sys.modules, "runpod", f)
    monkeypatch.setattr(RunPodSession, "_register_cleanup", lambda _self: None)
    return f


@pytest.fixture
def clock(monkeypatch):
    """Virtual time for runpod_pod (sleeps advance it, so a 40-minute stall is instant).

    Starts at the real epoch time: pod_journal reads the real clock to expire
    late-create windows, and the two must agree.
    """
    c = _Clock()
    c.now = __import__("time").time()
    monkeypatch.setattr(runpod_pod, "time", types.SimpleNamespace(
        time=c.time, sleep=c.sleep, strftime=__import__("time").strftime))
    return c


def _dies(session):
    """SIGKILL: no cleanup runs, the kernel drops the lock."""
    pod_journal._unlock(session._journal._fd)
    session._journal._fd = None


def _files(tmp_path):
    d = tmp_path / "runpod_journal"
    return sorted(os.listdir(d)) if d.exists() else []


# --- sound behaviour ---------------------------------------------------------

def test_300_replacements_keep_the_journal_constant_and_one_pod_live(tmp_path, fleet, clock):
    s = RunPodSession(str(tmp_path))
    s.create_pod()
    for _ in range(300):
        s.forget_pod()
        s.create_pod()
        assert len(fleet.live) == 1
    assert len(s._journal.snapshot.claims) == 1
    assert os.path.getsize(s._journal.json_path) < 1000   # 224 bytes when measured
    assert fleet.terminate_calls == 300
    s._cleanup()
    assert fleet.live == {} and _files(tmp_path) == []


class _CrashPoint:
    """Raise _Killed at the n-th journal write or API call (counted together)."""

    def __init__(self, n):
        self.n, self.i = n, 0

    def tick(self):
        self.i += 1
        if self.i == self.n:
            raise _Killed()


@pytest.mark.parametrize("n", range(1, 13))
def test_a_crash_at_any_point_of_a_replacement_is_settled_by_the_next_run(tmp_path, fleet, clock, n):
    """Kill the process at the n-th journal write / API call of forget_pod + create_pod
    (n past the last one means it finished). The next run must rent exactly one pod
    and leave none."""
    s = RunPodSession(str(tmp_path))
    s.create_pod()
    cp = _CrashPoint(n)
    real_write = pod_journal._write_atomic
    saved = {name: getattr(fleet, name) for name in ("create_pod", "terminate_pod", "get_pod", "get_pods")}
    pod_journal._write_atomic = lambda p, d: (cp.tick(), real_write(p, d))[1]
    for name, real in saved.items():
        setattr(fleet, name, (lambda real: lambda *a, **k: (cp.tick(), real(*a, **k))[1])(real))
    try:
        s.forget_pod()
        s.create_pod()
        crashed = False
    except _Killed:
        crashed = True
    finally:
        pod_journal._write_atomic = real_write
        for name, real in saved.items():
            setattr(fleet, name, real)
    if crashed:
        _dies(s)
    else:
        s._cleanup()

    nxt = RunPodSession(str(tmp_path))
    nxt.create_pod()
    assert len(fleet.live) == 1
    nxt._cleanup()
    assert fleet.live == {}


def test_a_terminate_outage_never_rents_across_a_restart_storm_and_recovers(tmp_path, fleet, clock):
    """capped_run --retries 6 means 7 processes. Each one refuses; none rents; the
    claim is carried (one journal, not one per process); the first process after
    the outage rents."""
    s = RunPodSession(str(tmp_path))
    s.create_pod()
    fleet.fail_terminate.add("pod-1")
    s.forget_pod()
    s._cleanup()                                   # the dying child
    for _ in range(7):
        child = RunPodSession(str(tmp_path))
        with pytest.raises(UnsafeToAllocateError):
            child.with_setup_retry(child.ensure_pod)
        child._cleanup()
        clock.sleep(30)                            # capped_run's pause between attempts
    assert fleet.created == 1 and list(fleet.live) == ["pod-1"]
    assert len(_files(tmp_path)) == 2              # .json + .lock, carried forward
    fleet.fail_terminate.clear()
    ok = RunPodSession(str(tmp_path))
    ok.with_setup_retry(ok.ensure_pod)
    assert list(fleet.live) == [ok.pod_id]


def test_a_full_disk_never_stops_a_terminate_and_only_blocks_renting(tmp_path, fleet, monkeypatch):
    s = RunPodSession(str(tmp_path))
    s.create_pod()
    real = pod_journal._write_atomic
    full = {"on": True}

    def write(path, payload):
        if full["on"]:
            raise OSError(28, "No space left on device")
        real(path, payload)
    monkeypatch.setattr(pod_journal, "_write_atomic", write)
    s.forget_pod()
    assert fleet.live == {}
    with pytest.raises(UnsafeToAllocateError):
        s.create_pod()
    full["on"] = False
    s.create_pod()
    s._cleanup()
    assert fleet.live == {}


class _RaisingGetPodFleet(_FleetRunpod):
    def get_pod(self, pod_id):
        if pod_id not in self.live:
            raise RuntimeError("Pod not found")      # an API answer, not None
        return super().get_pod(pod_id)


def test_get_pod_raising_for_a_gone_pod_does_not_block_or_unconfirm(tmp_path, monkeypatch, clock):
    import sys

    f = _RaisingGetPodFleet()
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "runpod", f)
    monkeypatch.setattr(RunPodSession, "_register_cleanup", lambda _self: None)
    s = RunPodSession(str(tmp_path))
    s.create_pod()
    real = f.terminate_pod

    def lands_but_times_out(pod_id):
        real(pod_id)
        raise TimeoutError("terminate timed out after the server acted")
    f.terminate_pod = lands_but_times_out
    s.forget_pod()
    f.terminate_pod = real
    s.create_pod()
    s._cleanup()
    assert f.live == {} and s.cleanup_confirmed is True


# --- fixed by this review ------------------------------------------------------

def test_cleanup_in_a_total_api_outage_finishes_inside_the_supervisors_grace(tmp_path, fleet, clock):
    """Every request costs the SDK's 30 s timeout. Terminate's three attempts and
    then the sweep passes took ~200 s; the supervisor's grace was 120 s, so it
    SIGKILLed the child mid-cleanup. (The old test compared two constants.)"""
    from tools import capped_run

    s = RunPodSession(str(tmp_path))
    s.create_pod()

    def timing_out(*_a, **_k):
        clock.now += 30
        raise TimeoutError("api timeout")
    fleet.terminate_pod = fleet.get_pods = fleet.get_pod = timing_out
    t0 = clock.now
    s._cleanup()

    assert s.cleanup_confirmed is False
    assert clock.now - t0 <= capped_run.CHILD_GRACE_S


# --- Findings from the research 0026 review, 21 Nov 2026 (fixed) ---------------

class _LaggyFleet(_FleetRunpod):
    """A pod bills from creation, but the API cannot see it for `lag_s` seconds."""

    def __init__(self, lag_s, clock):
        super().__init__()
        self.lag_s, self.clock = lag_s, clock
        self.born: dict[str, float] = {}

    def _visible(self, pod_id):
        return self.clock.now - self.born.get(pod_id, -1e9) >= self.lag_s

    def create_pod(self, name, **kw):
        r = super().create_pod(name, **kw)
        self.born[r["id"]] = self.clock.now
        return r

    def get_pod(self, pod_id):
        return super().get_pod(pod_id) if self._visible(pod_id) else None

    def get_pods(self):
        return [p for p in super().get_pods() if self._visible(p["id"])]

    def terminate_pod(self, pod_id):
        if pod_id in self.live and not self._visible(pod_id):
            raise RuntimeError("pod not found to terminate")
        return super().terminate_pod(pod_id)


def test_a_pod_invisible_past_the_ready_timeout_is_not_forgotten(tmp_path, monkeypatch, clock):
    import sys

    lag = runpod_pod.POD_READY_TIMEOUT + 120
    f = _LaggyFleet(lag, clock)
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "runpod", f)
    monkeypatch.setattr(RunPodSession, "_register_cleanup", lambda _self: None)
    s = RunPodSession(str(tmp_path))
    with pytest.raises(PodSetupError):
        s.create_pod()
    s._cleanup()
    clock.sleep(lag)                                # it finally shows up
    assert f.live == {}


def test_a_peer_that_dies_after_we_reconciled_is_reaped_at_the_next_create(tmp_path, fleet, clock):
    a = RunPodSession(str(tmp_path))
    a.create_pod()
    b = RunPodSession(str(tmp_path))
    b.create_pod()                                  # A is alive: protected
    a_pod = a.pod_id
    _dies(a)
    b.forget_pod()
    b.create_pod()                                  # B's next replacement
    assert a_pod not in fleet.live


def test_a_confirmed_clean_run_leaves_no_journal_after_an_earlier_ambiguous_failure(tmp_path, fleet, clock):
    s = RunPodSession(str(tmp_path))
    real = fleet.create_pod
    calls = {"n": 0}

    def flaky(name, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("gateway timeout")   # ambiguous; nothing was made
        return real(name, **kw)
    fleet.create_pod = flaky
    s.create_pod()
    s._cleanup()
    assert s.cleanup_confirmed is True
    assert _files(tmp_path) == []


# --- F2 / F4 / F8 / F9 edges (fixed in the journal hardening bundle) ------------

def _laggy_session(tmp_path, monkeypatch, clock, lag):
    import sys

    f = _LaggyFleet(lag, clock)
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "runpod", f)
    monkeypatch.setattr(RunPodSession, "_register_cleanup", lambda _self: None)
    return f, RunPodSession(str(tmp_path))


def test_an_unseen_pod_is_not_replaced_while_its_claim_is_unconfirmed(tmp_path, monkeypatch, clock):
    lag = runpod_pod.POD_READY_TIMEOUT + 120
    f, s = _laggy_session(tmp_path, monkeypatch, clock, lag)
    with pytest.raises(PodSetupError):
        s.create_pod()
    with pytest.raises(UnsafeToAllocateError):      # the retry must not rent beside it
        s.create_pod()
    assert f.created == 1
    clock.sleep(lag)
    f.lag_s = 0                                     # the API has caught up
    s.create_pod()                                  # visible now: gate terminates it, then rents
    assert list(f.live) == [s.pod_id] and s.pod_id != "pod-1"
    s._cleanup()
    assert f.live == {}


def test_a_session_killed_while_a_pod_is_unseen_leaves_a_claim_the_next_run_settles(
        tmp_path, monkeypatch, clock):
    lag = runpod_pod.POD_READY_TIMEOUT + 120
    f, s = _laggy_session(tmp_path, monkeypatch, clock, lag)
    with pytest.raises(PodSetupError):
        s.create_pod()
    _dies(s)                                        # SIGKILL before the exit sweep ran
    assert len(_files(tmp_path)) == 2               # the claim is on disk
    clock.sleep(lag)
    nxt = RunPodSession(str(tmp_path))
    nxt.reconcile_journals()
    assert f.live == {}


def test_not_found_on_a_pod_the_api_has_shown_us_is_still_proof_it_is_gone(tmp_path, fleet, clock):
    s = RunPodSession(str(tmp_path))
    s.create_pod()                                  # the ready wait saw it
    pod_id = s.pod_id
    fleet.live.pop(pod_id)                          # gone behind our back
    t0 = clock.now
    s._cleanup()
    assert s.cleanup_confirmed is True
    assert clock.now - t0 < 60                      # no waiting for a pod that was seen
    assert _files(tmp_path) == []


def test_the_exit_sweep_does_not_wait_for_an_unseen_pod_when_the_api_is_down(tmp_path, monkeypatch, clock):
    from tools import capped_run

    lag = runpod_pod.POD_READY_TIMEOUT + 120
    f, s = _laggy_session(tmp_path, monkeypatch, clock, lag)
    with pytest.raises(PodSetupError):
        s.create_pod()

    def down(*_a, **_k):
        clock.now += 30
        raise TimeoutError("api timeout")
    f.get_pods = down
    t0 = clock.now
    s._cleanup()
    assert s.cleanup_confirmed is False
    assert clock.now - t0 <= capped_run.CHILD_GRACE_S


def test_the_unseen_wait_at_exit_is_bounded(tmp_path, monkeypatch, clock):
    from tools import capped_run

    f, s = _laggy_session(tmp_path, monkeypatch, clock, 10 * runpod_pod.UNSEEN_POD_GRACE_S)
    with pytest.raises(PodSetupError):
        s.create_pod()
    t0 = clock.now
    s._cleanup()
    assert s.cleanup_confirmed is False
    assert clock.now - t0 <= capped_run.CHILD_GRACE_S
    assert f.live                                   # still owed: the journal keeps the claim
    assert any(n.endswith(".json") for n in _files(tmp_path))


def test_a_ghost_create_followed_by_a_success_on_another_gpu_keeps_the_window_open(
        tmp_path, fleet, clock):
    """Two pods under one name (the first request landed after all): do not close the window."""
    s = RunPodSession(str(tmp_path))
    real = fleet.create_pod
    calls = {"n": 0}

    def ghost_then_ok(name, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            real(name, **kw)                        # made server-side...
            raise TimeoutError("gateway timeout")   # ...but the caller never hears
        return real(name, **kw)
    fleet.create_pod = ghost_then_ok
    s.create_pod()
    assert len(fleet.live) == 2
    assert any(w.name == s.pod_name for w in s._journal.snapshot.windows)
    s._cleanup()
    assert fleet.live == {}                         # the sweep takes both, by name


def test_the_window_stays_open_when_the_name_listing_fails_after_an_ambiguous_create(
        tmp_path, fleet, clock):
    s = RunPodSession(str(tmp_path))
    real = fleet.create_pod
    calls = {"n": 0}

    def flaky(name, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("gateway timeout")
        fleet.list_failures = 1                     # the by-name check cannot list
        return real(name, **kw)
    fleet.create_pod = flaky
    s.create_pod()
    assert any(w.name == s.pod_name for w in s._journal.snapshot.windows)
    s._cleanup()
    assert fleet.live == {}


def test_write_atomic_flushes_the_file_and_then_its_directory(tmp_path, monkeypatch):
    import stat

    kinds = []
    monkeypatch.setattr(pod_journal, "_fsync",
                        lambda fd: kinds.append("dir" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file"))
    pod_journal._write_atomic(str(tmp_path / "x.json"), {"a": 1})
    assert kinds == ["file", "dir"]


def test_fsync_asks_for_a_full_flush_where_the_platform_has_one(tmp_path, monkeypatch):
    import fcntl
    calls = []
    monkeypatch.setattr(fcntl, "F_FULLFSYNC", 51, raising=False)
    monkeypatch.setattr(fcntl, "fcntl", lambda fd, cmd, *a: calls.append(cmd))
    monkeypatch.setattr(os, "fsync", lambda fd: calls.append("fsync"))
    with open(tmp_path / "f", "w") as f:
        pod_journal._fsync(f.fileno())
    assert calls == [51]                                    # not the plain fsync

    calls.clear()

    def unsupported(fd, cmd, *a):
        raise OSError("not supported")
    monkeypatch.setattr(fcntl, "fcntl", unsupported)
    with open(tmp_path / "f", "w") as f:
        pod_journal._fsync(f.fileno())
    assert calls == ["fsync"]                               # falls back


def test_scan_peers_racing_a_closing_peer_leaves_no_lock_file(tmp_path, monkeypatch):
    d = tmp_path / "runpod_journal"
    d.mkdir()
    (d / "gone0001.json").write_text("{}")
    real_lock = pod_journal._lock

    def lock_then_peer_closes(path):
        fd = real_lock(path)
        os.remove(d / "gone0001.json")              # the owner closed between listdir and open
        return fd
    monkeypatch.setattr(pod_journal, "_lock", lock_then_peer_closes)
    assert pod_journal.scan_peers(str(d), "me") == []
    assert os.listdir(d) == []
