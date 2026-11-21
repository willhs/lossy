"""Scenarios from the sceptical review of the pod journal and exit sweep
(research 0026, "Review, 21 Nov 2026"), run against a fake RunPod account.

The first group pins behaviour the review found sound (so a later change that
breaks it fails). The second group is OPEN findings: each is `xfail(strict)`, so
it turns into a failure the day the defect is fixed and the marker must go.
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


# --- OPEN findings (research 0026 review, 21 Nov 2026) -------------------------

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


@pytest.mark.xfail(strict=True, reason="finding F2: 'not found' on a pod we just created is "
                   "taken as proof it is gone; the claim is dropped and the exit sweep says "
                   "confirmed while the pod later appears and bills")
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


@pytest.mark.xfail(strict=True, reason="finding F3: reconcile runs once per session, so a peer "
                   "that dies after it is never reaped by a still-running session")
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


@pytest.mark.xfail(strict=True, reason="finding F4: an earlier ambiguous create failure keeps the "
                   "late-create window open after a later success, so a confirmed-clean run "
                   "leaves a journal for an hour")
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
