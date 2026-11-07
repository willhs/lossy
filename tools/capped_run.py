"""Run a lossy command under a hard RunPod spend cap, with retries and a pod sweep.

decode.py has no spend limit and exits (rather than re-provisioning) when a pod
dies, so an unattended run needs a supervisor. This one:

  * reads the RunPod balance before starting and treats `baseline - balance` as
    spend (ground truth, includes setup and idle time);
  * polls every --poll seconds and, once the cap is hit (or, after 10% of the
    shots are done, the projected total would cross it), stops the command at
    its next resume point and terminates the pods;
  * re-runs the command after a non-zero exit (decode resumes from its progress
    file), up to --retries times;
  * always ends by terminating every pod on the account that was not there at
    start, and prints the final balance and pod list.

    uv run python tools/capped_run.py --cap 8 --progress output/ep/decode_progress_runpod-ltx25.json \
        --total-shots 315 -- uv run python decode.py output/ep --strategy runpod-ltx25

Exit codes: 0 command succeeded, 3 stopped by the cap/projection, 4 retries exhausted.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config  # noqa: E402

config.load_env()
HEADERS = {"Authorization": f"Bearer {os.environ.get('RUNPOD_API_KEY', '')}"}
GQL = "https://api.runpod.io/graphql"


def _gql(query: str) -> dict:
    r = requests.post(GQL, headers=HEADERS, json={"query": query}, timeout=30)
    r.raise_for_status()
    return r.json()["data"]


def balance_and_pods() -> tuple[float, list[str]]:
    me = _gql("query{myself{clientBalance pods{id}}}")["myself"]
    return float(me["clientBalance"]), [p["id"] for p in me["pods"]]


def terminate(pod_ids: list[str]) -> None:
    for pid in pod_ids:
        try:
            _gql(f'mutation{{podTerminate(input:{{podId:"{pid}"}})}}')
            print(f"[capped_run] terminated pod {pid}", flush=True)
        except Exception as e:  # keep going: the next poll or the final sweep retries
            print(f"[capped_run] terminate {pid} failed: {e}", flush=True)


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
                    help="stop this far under the cap: balance lags real spend")
    ap.add_argument("--poll", type=int, default=90)
    ap.add_argument("--retries", type=int, default=6)
    ap.add_argument("--progress", help="decode progress json, for the spend projection")
    ap.add_argument("--total-shots", type=int, help="shot count, for the spend projection")
    ap.add_argument("--shots", help="shots.json, to project spend per second of video instead of per shot")
    ap.add_argument("--baseline", type=float,
                    help="balance the cap is measured from (default: the balance now); "
                         "pass the original one when taking over a run that has already spent")
    ap.add_argument("--attach-pid", type=int,
                    help="supervise an already-running first attempt (a process-group leader) "
                         "instead of starting one; `cmd` is then only used for retries")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd

    baseline, pods0 = balance_and_pods()
    if a.baseline is not None:
        baseline = a.baseline
    if a.attach_pid:
        pods0 = []  # the attached run owns any pod that exists now, so the final sweep ends it
    limit = a.cap - a.margin
    print(f"[capped_run] baseline ${baseline:.4f}, pods at start {pods0}, stop at ${limit:.2f} spent", flush=True)

    total_s = 0.0
    if a.shots:
        with open(a.shots) as f:
            total_s = sum(x["duration_s"] for x in json.load(f)["shots"])
    ref: dict = {}
    stop: dict[str, str | None] = {"reason": None}
    child: dict = {"p": None}

    def watch():
        while stop["reason"] is None:
            time.sleep(a.poll)
            try:
                bal, _ = balance_and_pods()
            except Exception as e:
                print(f"[capped_run] balance poll failed: {e}", flush=True)
                continue
            spent = baseline - bal
            done = done_shots(a.progress)
            line = f"[capped_run] {time.strftime('%H:%M:%S')} spent ${spent:.2f}"
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
                else:
                    projected = 0
            else:
                projected = spent * a.total_shots / done if (a.total_shots and done >= 0.1 * a.total_shots) else 0
            print(line, flush=True)
            if spent >= limit:
                stop["reason"] = f"spend ${spent:.2f} reached the ${limit:.2f} stop line"
            elif projected > a.cap:
                stop["reason"] = f"projected ${projected:.2f} would cross the ${a.cap:.2f} cap"
            if stop["reason"]:
                print(f"[capped_run] STOP: {stop['reason']}", flush=True)
                p = child["p"]
                if p and p.poll() is None:
                    os.killpg(p.pid, signal.SIGTERM)
                    time.sleep(20)
                    if p.poll() is None:
                        os.killpg(p.pid, signal.SIGKILL)
                _, pods = balance_and_pods()
                terminate([x for x in pods if x not in pods0])

    threading.Thread(target=watch, daemon=True).start()

    rc = 4
    for attempt in range(1, a.retries + 2):
        if stop["reason"]:
            rc = 3
            break
        if attempt == 1 and a.attach_pid:
            print(f"[capped_run] attaching to running pid {a.attach_pid}", flush=True)
            child["p"] = _Attached(a.attach_pid)
            code = child["p"].wait()
            if code != 0 and a.total_shots and done_shots(a.progress) >= a.total_shots:
                code = 0
        else:
            print(f"[capped_run] attempt {attempt}: {' '.join(cmd)}", flush=True)
            child["p"] = subprocess.Popen(cmd, start_new_session=True)
            code = child["p"].wait()
        if stop["reason"]:
            rc = 3
            break
        if code == 0:
            rc = 0
            break
        print(f"[capped_run] command exited {code}; resuming", flush=True)
        _, pods = balance_and_pods()
        terminate([x for x in pods if x not in pods0])  # a failed run may leave its pod up
        time.sleep(30)
    stop["reason"] = stop["reason"] or "finished"

    time.sleep(2)
    bal, pods = balance_and_pods()
    terminate([x for x in pods if x not in pods0])
    time.sleep(5)
    bal, pods = balance_and_pods()
    print(f"[capped_run] FINAL rc={rc} spent ${baseline - bal:.4f}, balance ${bal:.4f}, pods {pods}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
