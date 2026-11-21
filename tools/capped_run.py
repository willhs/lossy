"""Run a lossy command under a hard RunPod spend cap, with retries and a pod sweep.

decode.py has no spend limit and exits (rather than re-provisioning) when a pod
dies, so an unattended run needs a supervisor. This one:

  * reads the RunPod balance before starting and treats `baseline - balance` as
    spend (ground truth, includes setup and idle time);
  * polls every --poll seconds and, once the cap is hit (or, after 10% of the
    shots are done, the projected total would cross it), stops the command at
    its next resume point and terminates the pods;
  * FAILS CLOSED when it cannot see the balance: spend since the last good
    reading is assumed to run at --burn-per-hr, and the run stops once that
    estimate reaches the stop line or the reading is older than --max-stale
    seconds (exit 6). A dead balance API, a watcher bug or a dead watcher thread
    all end in a stopped child, never an unsupervised one;
  * holds back a reserve under the cap for the money still spent while it is
    stopping (poll interval + the child's cleanup grace + the sweep) -- see
    `reserve_usd`; --margin is the floor;
  * re-runs the command after a non-zero exit (decode resumes from its progress
    file), up to --retries times;
  * always ends by terminating the pods THIS RUN OWNS and printing the final
    balance and pod list. Ownership is the pod ownership journal the child writes
    in --output-dir (`<output-dir>/runpod_journal/`, ADR 0010): pods it claims by
    id, and pods carrying a name it spoke for. Pods in no journal -- another
    run's -- are never touched. (`--sweep-account` restores the old rule,
    "every pod that was not there at start", and is only safe if nothing else
    uses the account.) The sweep retries transient API errors, runs even when
    capped_run itself is interrupted (Ctrl-C, SIGTERM, SIGHUP: the child is
    stopped first, since it runs in its own session and would otherwise carry on
    with nobody enforcing the cap), and is not reported clean unless no owned pod
    shows in the account.

    uv run python tools/capped_run.py --cap 8 --output-dir output/ep \\
        --progress output/ep/decode_progress_runpod-ltx25.json \\
        --total-shots 315 -- uv run python decode.py output/ep --strategy runpod-ltx25

Not covered: fal and other non-RunPod spend; balance LAG (RunPod crediting spend
late) beyond what the reserve and --burn-per-hr bound.

Exit codes: 0 command succeeded, 3 stopped by the cap/projection, 4 retries exhausted,
5 the command succeeded but a pod could not be confirmed terminated (check the
RunPod console), 6 stopped because spend could not be measured (balance unreadable
for --max-stale seconds, or the watcher failed), 130 interrupted.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, NamedTuple

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config  # noqa: E402
import pod_journal  # noqa: E402

config.load_env()
HEADERS = {"Authorization": f"Bearer {os.environ.get('RUNPOD_API_KEY', '')}"}
GQL = "https://api.runpod.io/graphql"

RC_CAP = 3
RC_BLIND = 6


class Pod(NamedTuple):
    id: str
    name: str


def _gql(query: str) -> dict:
    r = requests.post(GQL, headers=HEADERS, json={"query": query}, timeout=30)
    r.raise_for_status()
    return r.json()["data"]


def balance_and_pods() -> tuple[float, list[Pod]]:
    me = _gql("query{myself{clientBalance pods{id name}}}")["myself"]
    return float(me["clientBalance"]), [Pod(p["id"], p.get("name") or "") for p in me["pods"]]


def terminate(pod_ids: list[str]) -> None:
    for pid in pod_ids:
        try:
            _gql(f'mutation{{podTerminate(input:{{podId:"{pid}"}})}}')
            print(f"[capped_run] terminated pod {pid}", flush=True)
        except Exception as e:  # keep going: the next poll or the final sweep retries
            print(f"[capped_run] terminate {pid} failed: {e}", flush=True)


# -- spend knowledge ----------------------------------------------------------

DEFAULT_BURN_PER_HR = 3.0   # two overlapping SECURE L40S-class pods (0.79 x 1.8 x 2); see reserve_usd
DEFAULT_MAX_STALE_S = 600


@dataclass
class SpendGuard:
    """What we know about spend, and when we knew it. Pure: callers pass the clock.

    Spend is `baseline - balance`, but a balance reading is only a fact about the
    moment it was taken. Without a fresh one the spend is *estimated* as the last
    reading plus `burn_per_hr` for the time since, and past `max_stale_s` there
    is no estimate worth trusting at all: the run stops. Both rules fail closed:
    not knowing the balance is a reason to stop, never a reason to carry on.

    `limit` is the stop line (cap minus reserve); `clock` values are only
    compared with each other, so monotonic seconds are right.
    """

    baseline: float
    limit: float
    burn_per_hr: float
    max_stale_s: float
    last_ok: float          # clock time of the last good balance read
    last_spent: float       # spend that reading showed

    @classmethod
    def start(cls, baseline: float, balance: float, limit: float, burn_per_hr: float,
              max_stale_s: float, now: float) -> "SpendGuard":
        return cls(baseline, limit, burn_per_hr, max_stale_s, now, baseline - balance)

    def observe(self, balance: float, now: float) -> None:
        self.last_ok, self.last_spent = now, self.baseline - balance

    def stale_s(self, now: float) -> float:
        return max(0.0, now - self.last_ok)

    def spent(self, now: float) -> float:
        """Best conservative estimate of spend right now."""
        return self.last_spent + self.burn_per_hr * self.stale_s(now) / 3600

    def verdict(self, now: float) -> tuple[str, int] | None:
        """(reason, exit code) if the run must stop now, else None."""
        stale = self.stale_s(now)
        if stale > self.max_stale_s:
            return (f"balance not read for {stale:.0f}s (limit {self.max_stale_s:.0f}s); "
                    f"spend is unknown", RC_BLIND)
        spent = self.spent(now)
        if spent >= self.limit:
            how = "reached" if stale == 0 else f"estimated ({stale:.0f}s since the last balance read) to have reached"
            return f"spend ${spent:.2f} {how} the ${self.limit:.2f} stop line", RC_CAP
        return None


def reserve_usd(margin: float, burn_per_hr: float, poll_s: float,
                grace_s: float | None = None, sweep_s: float | None = None) -> float:
    """Money still spent AFTER the stop decision, at the assumed worst-case burn.

    From the moment spend crosses the line, billing continues for up to one poll
    interval (we only notice at a poll), the child's cleanup grace and the
    sweep. `margin` stays the floor for balance lag, which nothing here bounds.
    """
    grace_s = CHILD_GRACE_S if grace_s is None else grace_s
    sweep_s = SWEEP_ATTEMPTS * SWEEP_DELAY_S if sweep_s is None else sweep_s
    return max(margin, burn_per_hr * (poll_s + grace_s + sweep_s) / 3600)


# -- pod ownership ------------------------------------------------------------

class AccountScope:
    """Old rule: every pod that was not there at start is ours. Needs an exclusive account."""

    def __init__(self, pods0: list[str]):
        self.pods0 = pods0

    def describe(self) -> str:
        return f"every account pod not present at start (--sweep-account); at start: {self.pods0}"

    def owned(self, pods: list[Pod]) -> list[str]:
        return [p.id for p in pods if p.id not in self.pods0]


class JournalScope:
    """A pod is ours iff a journal in the run's output dir claims its id or its name."""

    def __init__(self, journal_dir: str):
        self.journal_dir = journal_dir

    def describe(self) -> str:
        return f"pods claimed in the ownership journals under {self.journal_dir}"

    def owned(self, pods: list[Pod]) -> list[str]:
        own = pod_journal.read_ownership(self.journal_dir)   # raises: unknown ownership is not "none"
        return [p.id for p in pods if p.id in own.ids or (p.name and p.name in own.names)]


SWEEP_ATTEMPTS = 5
SWEEP_DELAY_S = 5
# After SIGTERM the child's handler terminates its pod (three attempts plus
# backoff) and THEN runs a confirmed exit sweep (CLEANUP_BUDGET_S = 90 s, checked
# only between passes). Against a dead API every request costs the SDK's 30 s
# timeout, which measured ~200 s end to end (research 0026 review, 21 Nov): the
# old 120 s grace SIGKILLed the child mid-cleanup. 240 s covers that with margin;
# a longer wait only costs time on a path that is already failing. The account
# sweep below is still the backstop.
CHILD_GRACE_S = 240


def sweep(scope, attempts: int = SWEEP_ATTEMPTS, delay: float = SWEEP_DELAY_S) -> list[str]:
    """Terminate every pod `scope` says this run owns. Returns the ones not confirmed gone.

    One unguarded API call used to end the run: a transient 5xx from RunPod
    while listing pods raised out of main() with the pod still billing. List,
    terminate, and list again until nothing owned is left or attempts run out.
    If ownership itself cannot be determined (an unreadable journal) nothing is
    terminated on a guess and the sweep reports itself unconfirmed.
    """
    extra: list[str] = []
    for i in range(attempts):
        try:
            _, pods = balance_and_pods()
            extra = scope.owned(pods)
            if not extra:
                return []
            terminate(extra)
        except Exception as e:
            print(f"[capped_run] pod sweep attempt {i + 1}/{attempts} failed: {e}", flush=True)
            extra = extra or ["<pods unknown: listing or ownership unavailable>"]
        time.sleep(delay)
    print(f"[capped_run] WARNING: could not confirm these pods are gone: {extra} -- "
          f"check https://www.runpod.io/console/pods", flush=True)
    return extra


def stop_child(p, grace: float = CHILD_GRACE_S) -> None:
    """SIGTERM the child's process group (its handlers terminate its pod), then SIGKILL."""
    if p is None or p.poll() is not None:
        return
    try:
        os.killpg(p.pid, signal.SIGTERM)
        deadline = time.time() + grace
        while p.poll() is None and time.time() < deadline:
            time.sleep(1)
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


class Run:
    """The child process and the single decision to stop it.

    Launching and stopping share one lock. Without it the watcher could decide to
    stop while the retry loop was between "no stop requested yet" and `Popen`:
    the watcher would stop the old (dead) child, the loop would then start a new
    one, and nothing would ever stop it. Now a launch either happens before the
    stop (and is the child the stop kills) or sees the stop and does not happen.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.child = None
        self.reason: str | None = None
        self.rc = RC_CAP

    def launch(self, start: Callable[[], object]):
        """Start a child unless a stop was requested. Returns it, or None if stopped."""
        with self._lock:
            if self.reason is not None:
                return None
            self.child = start()
            return self.child

    def stop(self, reason: str, rc: int = RC_CAP) -> None:
        """Record why, then stop the child. The first reason wins; later calls only re-stop."""
        with self._lock:
            if self.reason is None:
                self.reason, self.rc = reason, rc
            try:
                stop_child(self.child)
            except Exception as e:  # e.g. EPERM from killpg: still try the hard kill
                print(f"[capped_run] stop failed ({e}); killing", flush=True)
                try:
                    os.killpg(self.child.pid, signal.SIGKILL)
                except Exception as e2:
                    print(f"[capped_run] kill failed too: {e2}", flush=True)


def _raise_interrupt(signum, frame):
    raise KeyboardInterrupt


class _Attached:
    """Popen-alike for a process we did not start. Exit status is unknowable, so wait() returns 1."""

    def __init__(self, pid: int):
        self.pid = pid

    def poll(self):
        try:
            os.kill(self.pid, 0)
            return None
        except OSError:
            return 1

    def wait(self) -> int:
        while self.poll() is None:
            time.sleep(5)
        return 1


def done_shots(progress_path: str | None) -> int:
    try:
        with open(progress_path) as f:  # type: ignore[arg-type]
            return len(json.load(f).get("completed", []))
    except Exception:
        return 0


def done_seconds(progress_path: str | None, shots_path: str | None) -> float:
    """Seconds of source video whose shots are complete (cost follows duration, not shot count)."""
    try:
        with open(shots_path) as f:  # type: ignore[arg-type]
            dur = {s["index"]: s["duration_s"] for s in json.load(f)["shots"]}
        with open(progress_path) as f:  # type: ignore[arg-type]
            return sum(dur.get(i, 0.0) for i in json.load(f).get("completed", []))
    except Exception:
        return 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--cap", type=float, required=True, help="hard spend cap, USD")
    ap.add_argument("--margin", type=float, default=0.5,
                    help="minimum reserve under the cap, for balance lag (the reserve grows "
                         "with --burn-per-hr x poll + stop time when that is larger)")
    ap.add_argument("--burn-per-hr", type=float, default=DEFAULT_BURN_PER_HR,
                    help="assumed worst-case billing rate, USD/h, for the reserve and for "
                         "estimating spend while the balance is unreadable (default covers "
                         "two overlapping SECURE pods)")
    ap.add_argument("--max-stale", type=float, default=DEFAULT_MAX_STALE_S,
                    help="stop (exit 6) when the balance has not been read for this many seconds")
    ap.add_argument("--poll", type=int, default=90)
    ap.add_argument("--retries", type=int, default=6)
    ap.add_argument("--output-dir",
                    help="the child's output dir: pods are swept only if its ownership journals "
                         "(<dir>/runpod_journal/) claim them. Required unless --sweep-account")
    ap.add_argument("--sweep-account", action="store_true",
                    help="instead of journal ownership, terminate every account pod that was not "
                         "there at start. ONLY safe if nothing else uses the RunPod account")
    ap.add_argument("--progress", help="decode progress json, for the spend projection")
    ap.add_argument("--total-shots", type=int, help="shot count, for the spend projection")
    ap.add_argument("--shots", help="shots.json, to project spend per second of video instead of per shot")
    ap.add_argument("--no-project", action="store_true",
                    help="only stop on the spend line, never on a projection (the per-second "
                         "projection ignores fixed per-shot cost and over-estimates early on)")
    ap.add_argument("--baseline", type=float,
                    help="balance the cap is measured from (default: the balance now); "
                         "pass the original one when taking over a run that has already spent")
    ap.add_argument("--attach-pid", type=int,
                    help="supervise an already-running first attempt (a process-group leader) "
                         "instead of starting one; `cmd` is then only used for retries")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if bool(a.output_dir) == a.sweep_account:
        ap.error("give exactly one of --output-dir (sweep only pods this run's journals own) "
                 "or --sweep-account (sweep every new pod; assumes the account is not shared)")
    if a.output_dir and not any(os.path.abspath(x) == os.path.abspath(a.output_dir) for x in cmd):
        print(f"[capped_run] WARNING: --output-dir {a.output_dir} is not an argument of the command; "
              f"if the child writes its journal elsewhere its pods will NOT be swept", flush=True)

    baseline, pods0 = balance_and_pods()
    balance0 = baseline
    if a.baseline is not None:
        baseline = a.baseline
    if a.sweep_account:
        # An attached run owns any pod that exists now, so the final sweep ends it.
        scope = AccountScope([] if a.attach_pid else [p.id for p in pods0])
    else:
        scope = JournalScope(os.path.join(a.output_dir, pod_journal.JOURNAL_DIRNAME))
    reserve = reserve_usd(a.margin, a.burn_per_hr, a.poll)
    limit = a.cap - reserve
    if limit <= 0:
        ap.error(f"cap ${a.cap:.2f} does not exceed the ${reserve:.2f} reserve "
                 f"(--poll {a.poll}s and --burn-per-hr ${a.burn_per_hr:.2f} dominate it): "
                 f"nothing could be spent safely; lower --poll or raise the cap")
    guard = SpendGuard.start(baseline, balance0, limit, a.burn_per_hr, a.max_stale, time.monotonic())
    print(f"[capped_run] baseline ${baseline:.4f}, stop at ${limit:.2f} spent "
          f"(cap ${a.cap:.2f} - reserve ${reserve:.2f}); balance unreadable for "
          f"{a.max_stale:.0f}s stops the run, spend meanwhile assumed ${a.burn_per_hr:.2f}/h", flush=True)
    print(f"[capped_run] sweep scope: {scope.describe()}", flush=True)

    total_s = 0.0
    if a.shots:
        with open(a.shots) as f:
            total_s = sum(x["duration_s"] for x in json.load(f)["shots"])
    ref: dict = {}
    run = Run()

    def watch():
        try:
            while run.reason is None:
                time.sleep(a.poll)
                now = time.monotonic()
                try:
                    bal, _ = balance_and_pods()
                    guard.observe(bal, now)
                except Exception as e:
                    print(f"[capped_run] balance poll failed ({guard.stale_s(now):.0f}s since the "
                          f"last good read): {e}", flush=True)
                verdict = guard.verdict(now)
                projected = 0.0
                try:
                    spent = guard.last_spent
                    done = done_shots(a.progress)
                    line = f"[capped_run] {time.strftime('%H:%M:%S')} spent ${guard.spent(now):.2f}"
                    if a.total_shots:
                        line += f", {done}/{a.total_shots} shots"
                    if total_s:
                        secs = done_seconds(a.progress, a.shots)
                        ref.setdefault("r", (spent, secs))  # first reading: excludes boot cost already paid
                        r_spent, r_secs = ref["r"]
                        line += f", {secs:.0f}/{total_s:.0f}s video"
                        if secs - r_secs >= 0.1 * (total_s - r_secs) and secs > r_secs:
                            projected = spent + (spent - r_spent) / (secs - r_secs) * (total_s - secs)
                            line += f", projected ${projected:.2f}"
                    elif a.total_shots and done >= 0.1 * a.total_shots:
                        projected = spent * a.total_shots / done
                    print(line, flush=True)
                except Exception as e:  # reporting must never take the supervisor down
                    print(f"[capped_run] progress report failed: {e!r}", flush=True)
                if verdict is None and projected > a.cap and not a.no_project:
                    verdict = (f"projected ${projected:.2f} would cross the ${a.cap:.2f} cap", RC_CAP)
                if verdict:
                    print(f"[capped_run] STOP: {verdict[0]}", flush=True)
                    run.stop(*verdict)   # the main thread's finally block sweeps once the child is down
        except BaseException as e:  # a dead watcher must not leave the child running unenforced
            print(f"[capped_run] watcher failed ({e!r}): stopping the command", flush=True)
            run.stop(f"watcher failed: {e!r}", RC_BLIND)

    # The child runs in its own session, so a signal to us never reaches it.
    # Turn SIGTERM/SIGHUP into the same path as Ctrl-C: stop the child, sweep.
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _raise_interrupt)

    threading.Thread(target=watch, daemon=True).start()

    rc = 4
    try:
        for attempt in range(1, a.retries + 2):
            if run.reason:
                rc = run.rc
                break
            if attempt == 1 and a.attach_pid:
                print(f"[capped_run] attaching to running pid {a.attach_pid}", flush=True)
                child = run.launch(lambda: _Attached(a.attach_pid))
                code = child.wait() if child else 1
                if code != 0 and a.total_shots and done_shots(a.progress) >= a.total_shots:
                    code = 0
            else:
                print(f"[capped_run] attempt {attempt}: {' '.join(cmd)}", flush=True)
                child = run.launch(lambda: subprocess.Popen(cmd, start_new_session=True))
                code = child.wait() if child else 1
            if run.reason:
                rc = run.rc
                break
            if code == 0:
                rc = 0
                break
            print(f"[capped_run] command exited {code}; resuming", flush=True)
            sweep(scope)  # a failed run may leave its pod up
            time.sleep(30)
    except KeyboardInterrupt:
        rc = 130
        run.reason = run.reason or "interrupted"
        print("[capped_run] interrupted: stopping the command and sweeping pods", flush=True)
    finally:
        # Cleanup must not itself be interruptible: a second Ctrl-C mid-sweep
        # would abort the one step that stops the billing.
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)
        run.stop("finished")   # no-op on the reason if one is set; takes the launch lock
        time.sleep(2)
        leftover = sweep(scope)
        if leftover and rc == 0:
            rc = 5
    if rc == 4 and a.output_dir:
        try:
            own = pod_journal.read_ownership(os.path.join(a.output_dir, pod_journal.JOURNAL_DIRNAME))
        except pod_journal.JournalUnreadable:
            own = None
        if own is None or own.ids:
            print(f"[capped_run] !!! RUN STALLED (exit 4) with a pod journal claim outstanding: "
                  f"verify in the RunPod console, then `python -m pod_journal list {a.output_dir}` "
                  f"and `python -m pod_journal forget {a.output_dir} <pod_id>`, then resume (ADR 0010)",
                  flush=True)
    try:
        bal, pods = balance_and_pods()
        print(f"[capped_run] FINAL rc={rc} spent ${baseline - bal:.4f}, balance ${bal:.4f}, "
              f"pods {[p.id for p in pods]}", flush=True)
    except Exception as e:
        print(f"[capped_run] FINAL rc={rc}; final balance unavailable ({e}); "
              f"check https://www.runpod.io/console/pods", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
