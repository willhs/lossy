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
import threading
import time

import pytest

import pod_journal

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
        return 5.0, [capped_run.Pod(pid, f"lossy-comfyui-{pid}") for pid in self.live]

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
    def install(account, child_cmd, extra_args=(), scope=("--sweep-account",)):
        monkeypatch.setattr(capped_run, "balance_and_pods", account.balance_and_pods)
        monkeypatch.setattr(capped_run, "terminate", account.terminate)
        real_sleep = capped_run.time.sleep
        monkeypatch.setattr(capped_run.time, "sleep",
                            lambda s: real_sleep(0.05) if s <= 30 else real_sleep(s))
        monkeypatch.setattr(sys, "argv", [
            "capped_run", "--cap", "5", "--margin", "0.1", "--no-project", *scope,
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
            monkeypatch.setattr(capped_run, "balance_and_pods", lambda: (
                3.9, [capped_run.Pod(p, "") for p in account.live]))
            watch[0]()
        kids = self._hanging_child(monkeypatch, spend)
        run(account, [sys.executable, "-c", "import time; time.sleep(60)"],
            ["--cap", "1", "--poll", "0"])
        assert capped_run.main() == 3
        assert kids[0].poll() is not None
        assert account.live == []


# -- fail closed on a stale balance (research 0026 follow-up 3) ----------------

class TestSpendGuard:
    """Pure: the clock is a parameter. Spend we cannot see is assumed, never zero."""

    def guard(self, **kw):
        args = dict(baseline=10.0, limit=2.0, burn_per_hr=3.6, max_stale_s=600, now=0.0, balance=10.0)
        args.update(kw)
        return capped_run.SpendGuard.start(
            args["baseline"], args["balance"], args["limit"], args["burn_per_hr"],
            args["max_stale_s"], args["now"])

    def test_a_fresh_reading_below_the_line_does_not_stop(self):
        g = self.guard()
        g.observe(9.0, now=90)
        assert g.verdict(90) is None

    def test_a_fresh_reading_at_the_line_stops_on_the_cap(self):
        g = self.guard()
        g.observe(8.0, now=90)
        reason, rc = g.verdict(90)
        assert rc == capped_run.RC_CAP and "reached" in reason

    def test_unseen_time_is_billed_at_the_assumed_burn_rate(self):
        """$1.00 spent at the last read, 3.6 $/h = $0.001/s: 1000 s later the
        estimate crosses the $2.00 line though no balance read says so."""
        g = self.guard(max_stale_s=3600)
        g.observe(9.0, now=0)
        assert g.verdict(900) is None
        reason, rc = g.verdict(1001)
        assert rc == capped_run.RC_CAP and "estimated" in reason

    def test_a_reading_older_than_max_stale_stops_whatever_the_estimate(self):
        g = self.guard(burn_per_hr=0.0, max_stale_s=600)   # estimate never grows
        assert g.verdict(600) is None
        reason, rc = g.verdict(601)
        assert rc == capped_run.RC_BLIND and "unknown" in reason

    def test_a_new_reading_clears_the_staleness(self):
        g = self.guard(burn_per_hr=0.0, max_stale_s=600)
        g.observe(10.0, now=500)
        assert g.verdict(1000) is None
        assert g.verdict(1101)[1] == capped_run.RC_BLIND


class TestReserve:
    def test_margin_is_the_floor(self):
        assert capped_run.reserve_usd(0.5, burn_per_hr=3.0, poll_s=90, grace_s=240, sweep_s=25) == 0.5

    def test_reserve_covers_poll_grace_and_sweep_at_the_burn_rate(self):
        # 3.6 $/h = $0.001/s over (3600 + 240 + 60) s
        assert capped_run.reserve_usd(0.1, 3.6, 3600, grace_s=240, sweep_s=60) == pytest.approx(3.9)


class TestJournalScope:
    def journal(self, tmp_path, run_id, pods=(), window=None):
        j = pod_journal.PodJournal(str(tmp_path / pod_journal.JOURNAL_DIRNAME), run_id,
                                   f"lossy-comfyui-{run_id}")
        for pid in pods:
            j.set_pod(pid, f"lossy-comfyui-{run_id}")
        if window:
            j.open_window(window, time.time() + 3600)
        return j

    def test_owns_pods_by_claimed_id_and_by_name_but_nobody_elses(self, tmp_path):
        j = self.journal(tmp_path, "aaaa1111", pods=["claimed"], window="lossy-comfyui-latecreate")
        pods = [capped_run.Pod("claimed", "renamed"), capped_run.Pod("lost-id", "lossy-comfyui-latecreate"),
                capped_run.Pod("foreign", "lossy-comfyui-ffff0000"), capped_run.Pod("bare", "")]
        scope = capped_run.JournalScope(j.directory)
        assert scope.owned(pods) == ["claimed", "lost-id"]
        j.close()

    def test_a_dead_runs_journal_still_counts_as_this_runs_ownership(self, tmp_path):
        """Journals in the output dir are this run's family, live or dead: a pod a
        SIGKILLed child left behind is exactly what the sweep is for."""
        j = self.journal(tmp_path, "dead0000", pods=["orphan"])
        j.close()   # releases the lock, keeps the claim
        scope = capped_run.JournalScope(j.directory)
        assert scope.owned([capped_run.Pod("orphan", "")]) == ["orphan"]

    def test_no_journal_dir_owns_nothing(self, tmp_path):
        scope = capped_run.JournalScope(str(tmp_path / "nope"))
        assert scope.owned([capped_run.Pod("x", "lossy-comfyui-x")]) == []

    def test_an_unreadable_journal_terminates_nothing_and_is_reported_unconfirmed(
            self, tmp_path, monkeypatch, capsys):
        d = tmp_path / pod_journal.JOURNAL_DIRNAME
        d.mkdir()
        (d / "broken.json").write_text("{not json")
        account = _Account()
        account.live = ["pod-x"]
        monkeypatch.setattr(capped_run, "balance_and_pods", account.balance_and_pods)
        monkeypatch.setattr(capped_run, "terminate", account.terminate)
        left = capped_run.sweep(capped_run.JournalScope(str(d)), attempts=2, delay=0)
        assert left and account.terminate_calls == [] and account.live == ["pod-x"]
        assert "broken.json" in capsys.readouterr().out


class TestScopedSweepEndToEnd:
    def test_the_sweep_ends_this_runs_pods_and_spares_a_concurrent_runs(self, run, tmp_path):
        """Two runs share the account. The other run's pod appears during ours and
        is in no journal of ours: it must survive our sweep."""
        mine = pod_journal.PodJournal(str(tmp_path / pod_journal.JOURNAL_DIRNAME), "aaaa1111",
                                      "lossy-comfyui-aaaa1111")
        mine.set_pod("pod-mine", "lossy-comfyui-aaaa1111")
        account = _Account(spawn=["pod-mine", "pod-theirs"])
        run(account, [sys.executable, "-c", "pass", str(tmp_path)],
            scope=("--output-dir", str(tmp_path)))

        assert capped_run.main() == 0
        mine.close()
        assert account.live == ["pod-theirs"]

    def test_a_cap_below_the_reserve_is_refused_before_anything_starts(self, run):
        account = _Account()
        run(account, [sys.executable, "-c", "pass"], ["--cap", "1", "--poll", "3600"],
            scope=("--sweep-account",))
        # the fixture put --cap 5 first; the later --cap 1 wins in argparse
        with pytest.raises(SystemExit) as e:
            capped_run.main()
        assert e.value.code == 2

    def test_the_old_account_wide_sweep_needs_an_explicit_flag(self, run, monkeypatch):
        account = _Account()
        run(account, [sys.executable, "-c", "pass"], scope=())
        with pytest.raises(SystemExit) as e:
            capped_run.main()
        assert e.value.code == 2
        assert account.calls == 0                  # refused before touching the account


class TestBlindSupervision:
    """The balance API (or the watcher) fails: the child must still be stopped."""

    def blind_account(self, monkeypatch, account):
        """First call (the baseline) works; every later balance read fails."""
        def balance_and_pods():
            account.calls += 1
            if account.calls == 1:
                return 5.0, []
            raise RuntimeError("graphql 502")
        monkeypatch.setattr(capped_run, "balance_and_pods", balance_and_pods)

    def test_a_balance_that_never_reads_again_stops_the_child(self, run, monkeypatch):
        account = _Account()
        run(account, [sys.executable, "-c", "import time; time.sleep(60)"],
            ["--poll", "0", "--max-stale", "0.3", "--burn-per-hr", "0"])
        self.blind_account(monkeypatch, account)
        kids = []
        real_popen = subprocess.Popen
        monkeypatch.setattr(capped_run.subprocess, "Popen",
                            lambda *a, **k: kids.append(real_popen(*a, **k)) or kids[-1])
        try:
            assert capped_run.main() == capped_run.RC_BLIND
            assert kids[0].poll() is not None
        finally:
            for k in kids:
                if k.poll() is None:
                    k.kill()

    def test_a_failing_progress_report_does_not_blind_the_supervisor(self, run, monkeypatch):
        """done_shots raising used to kill the watcher thread: no more polls, no cap."""
        account = _Account()
        run(account, [sys.executable, "-c", "import time; time.sleep(60)"],
            ["--poll", "0", "--max-stale", "0.3", "--burn-per-hr", "0"])
        self.blind_account(monkeypatch, account)
        monkeypatch.setattr(capped_run, "done_shots",
                            lambda p: (_ for _ in ()).throw(RuntimeError("bad progress file")))
        assert capped_run.main() == capped_run.RC_BLIND

    def test_a_watcher_that_dies_stops_the_child(self, run, monkeypatch):
        account = _Account()
        run(account, [sys.executable, "-c", "import time; time.sleep(60)"], ["--poll", "0"])

        def boom(self, now):
            raise RuntimeError("watcher bug")
        monkeypatch.setattr(capped_run.SpendGuard, "verdict", boom)
        kids = []
        real_popen = subprocess.Popen
        monkeypatch.setattr(capped_run.subprocess, "Popen",
                            lambda *a, **k: kids.append(real_popen(*a, **k)) or kids[-1])
        try:
            assert capped_run.main() == capped_run.RC_BLIND
            assert kids[0].poll() is not None
        finally:
            for k in kids:
                if k.poll() is None:
                    k.kill()


class TestLaunchStopCoordination:
    def test_nothing_launches_after_a_stop(self):
        run = capped_run.Run()
        run.stop("cap reached")
        started = []
        assert run.launch(lambda: started.append(1)) is None
        assert started == [] and run.reason == "cap reached"

    def test_a_stop_during_a_launch_waits_for_it_and_stops_that_child(self):
        """The old race: the watcher stopped the previous child while the retry
        loop was between its stop check and Popen, so the new child ran with no
        stop ever sent."""
        run = capped_run.Run()
        in_launch, go = threading.Event(), threading.Event()
        kids = []

        def start():
            in_launch.set()
            go.wait(5)
            kids.append(subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                         start_new_session=True))
            return kids[0]

        t_launch = threading.Thread(target=run.launch, args=(start,))
        t_launch.start()
        assert in_launch.wait(5)
        t_stop = threading.Thread(target=run.stop, args=("cap reached",))
        t_stop.start()
        time.sleep(0.2)
        assert t_stop.is_alive()            # blocked behind the launch, not stopping a stale child
        go.set()
        t_launch.join(10)
        t_stop.join(10)
        try:
            assert kids and kids[0].poll() is not None
        finally:
            if kids and kids[0].poll() is None:
                kids[0].kill()

    def test_the_first_stop_reason_wins(self):
        run = capped_run.Run()
        run.stop("spend", capped_run.RC_CAP)
        run.stop("watcher failed", capped_run.RC_BLIND)
        assert (run.reason, run.rc) == ("spend", capped_run.RC_CAP)
