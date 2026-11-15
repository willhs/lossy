"""tools/capped_run.py: the supervisor must end every run with no pod left up.

Review of 08a72e8 (research 0026 note, 2026-11-15). capped_run is the backstop
for every other cleanup mechanism, so it has to survive the things the others
cannot: a transient RunPod API error while sweeping, and its own interruption
(Ctrl-C / SIGTERM), which used to orphan the child in its own session with
nothing left to enforce the cap. All RunPod calls are faked; nothing touches a
network or starts a pod.
"""

import os
import signal
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))

import capped_run  # noqa: E402


class _Account:
    """Fake RunPod account: pods that exist, and calls that may fail."""

    def __init__(self, spawn=(), fail_calls=()):
        self.live: list[str] = []
        self.spawn = list(spawn)            # pods the supervised command "starts" after the baseline read
        self.fail_calls = set(fail_calls)   # 1-based numbers of balance_and_pods calls that raise
        self.calls = 0
        self.terminate_calls: list[str] = []

    def balance_and_pods(self):
        self.calls += 1
        if self.calls == 2:
            self.live, self.spawn = self.live + self.spawn, []
        if self.calls in self.fail_calls:
            raise RuntimeError("graphql 502")
        return 5.0, list(self.live)

    def terminate(self, ids):
        for pid in ids:
            self.terminate_calls.append(pid)
            if pid in self.live:
                self.live.remove(pid)


@pytest.fixture(autouse=True)
def restore_signal_handlers():
    """main() installs handlers (and ignores signals while sweeping); undo that
    for the rest of the test process, or later children inherit SIG_IGN."""
    sigs = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    saved = {s: signal.getsignal(s) for s in sigs}
    yield
    for s, h in saved.items():
        signal.signal(s, h)


@pytest.fixture
def run(monkeypatch):
    def install(account, child_cmd, extra_args=()):
        monkeypatch.setattr(capped_run, "balance_and_pods", account.balance_and_pods)
        monkeypatch.setattr(capped_run, "terminate", account.terminate)
        real_sleep = capped_run.time.sleep
        monkeypatch.setattr(capped_run.time, "sleep",
                            lambda s: real_sleep(0.05) if s <= 30 else real_sleep(s))
        monkeypatch.setattr(sys, "argv", [
            "capped_run", "--cap", "5", "--margin", "0.1", "--no-project",
            "--poll", "3600", "--retries", "0", *extra_args, "--", *child_cmd])
    return install


class TestFinalSweep:
    def test_a_transient_listing_error_does_not_leave_the_pod_up(self, run):
        """The sweep that ends the run used to be one unguarded API call: a
        single 502 raised out of main() with the pod still billing."""
        # Call 1 is the baseline (no pods). The child "starts" a pod; the first
        # sweep listing (call 2) then fails, the next works.
        account = _Account(spawn=["pod-leak"], fail_calls={2})
        run(account, [sys.executable, "-c", "pass"])

        assert capped_run.main() == 0
        assert account.live == []

    def test_a_pod_that_cannot_be_confirmed_gone_is_not_reported_clean(self, run, capsys):
        account = _Account(spawn=["pod-stuck"])
        account.terminate = lambda ids: account.terminate_calls.extend(ids)  # never works
        run(account, [sys.executable, "-c", "pass"])

        assert capped_run.main() == 5               # "succeeded, but a pod may still be up"
        assert "pod-stuck" in capsys.readouterr().out


class TestInterruptedSupervisor:
    @pytest.fixture
    def handlers(self, monkeypatch):
        recorded = {}
        monkeypatch.setattr(capped_run.signal, "signal",
                            lambda sig, h: recorded.__setitem__(sig, h))
        return recorded

    def _hanging_child(self, monkeypatch, trigger):
        """Popen whose wait() stands in for 'the operator pressed Ctrl-C / sent SIGTERM'."""
        real_popen = subprocess.Popen
        kids = []

        class Child(real_popen):
            def wait(self, *a, **k):
                if not getattr(self, "_triggered", False):
                    self._triggered = True
                    trigger()
                return super().wait(*a, **k)

        def make(*a, **k):
            kids.append(Child(*a, **k))
            return kids[-1]
        monkeypatch.setattr(capped_run.subprocess, "Popen", make)
        return kids

    def test_ctrl_c_stops_the_child_and_sweeps_the_pods(self, run, monkeypatch, handlers):
        account = _Account(spawn=["pod-child"])

        def ctrl_c():
            raise KeyboardInterrupt
        kids = self._hanging_child(monkeypatch, ctrl_c)
        run(account, [sys.executable, "-c", "import time; time.sleep(60)"])

        rc = capped_run.main()
        try:
            assert rc == 130
            assert kids[0].poll() is not None        # the child did not outlive its supervisor
            assert account.live == []
        finally:
            if kids[0].poll() is None:
                kids[0].kill()

    def test_sigterm_is_handled_like_ctrl_c(self, run, monkeypatch, handlers):
        """A bare SIGTERM used to kill capped_run outright: no sweep, and the
        child (its own session) kept running with nobody enforcing the cap."""
        account = _Account(spawn=["pod-child"])

        def sigterm():
            handlers[signal.SIGTERM](signal.SIGTERM, None)   # KeyError if none installed
        kids = self._hanging_child(monkeypatch, sigterm)
        run(account, [sys.executable, "-c", "import time; time.sleep(60)"])

        rc = capped_run.main()
        try:
            assert rc == 130
            assert kids[0].poll() is not None
            assert account.live == []
        finally:
            if kids[0].poll() is None:
                kids[0].kill()

    def test_unexpected_child_wait_error_still_sweeps(self, run, monkeypatch, handlers):
        account = _Account()

        def error():
            account.live.append("pod-child")
            raise RuntimeError("wait failed")
        kids = self._hanging_child(monkeypatch, error)
        run(account, [sys.executable, "-c", "import time; time.sleep(60)"])
        with pytest.raises(RuntimeError, match="wait failed"):
            capped_run.main()
        assert kids[0].poll() is not None
        assert account.live == []

    def test_cap_counts_account_spend_and_sweeps_both_replacement_pods(
            self, run, monkeypatch, handlers):
        account = _Account()
        watch = []
        monkeypatch.setattr(capped_run.threading, "Thread",
                            lambda target, **kw: type("Thread", (), {"start": lambda self: watch.append(target)})())

        def spend():
            account.live.extend(["old-wedged", "replacement"])
            # Account billing includes both pods and their overlap; no per-pod
            # timer or current pod handle participates in the stop decision.
            monkeypatch.setattr(capped_run, "balance_and_pods", lambda: (3.9, list(account.live)))
            watch[0]()
        kids = self._hanging_child(monkeypatch, spend)
        run(account, [sys.executable, "-c", "import time; time.sleep(60)"],
            ["--cap", "1", "--poll", "0"])
        assert capped_run.main() == 3
        assert kids[0].poll() is not None
        assert account.live == []
