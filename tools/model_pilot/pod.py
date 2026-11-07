"""Pod + spend harness for the video-model pilot (research 0025).

One long-lived dev pod, driven step by step, built on the production
RunPodSession so it inherits the same self-destruct and stray sweep:

    pod.py up      provision (big disk), arm self-destruct, write state
    pod.py ssh CMD run CMD on the pod (re-arms the self-destruct afterwards)
    pod.py put LOCAL... REMOTE_DIR / get REMOTE LOCAL
    pod.py spend   RunPod balance delta vs the pilot's starting balance
    pod.py down    terminate, sweep, and confirm via the API that 0 pods remain

Spend is tracked from the account balance itself (ground truth, not an
estimate) against PILOT_START_BALANCE; every command refuses to run once the
drop passes the soft cap, leaving headroom under the task's hard $10 cap.

The state file lives in --dir, so decode.py pointed at the same dir reconnects
to this pod instead of provisioning its own -- the baseline runs through the
real production code path.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import config  # noqa: E402
import runpod_pod  # noqa: E402

PILOT_START_BALANCE = 17.6500742468  # read 2026-11-07 before any pilot spend
HARD_CAP = 10.00
SOFT_CAP = 8.50   # stop starting new work here; headroom for teardown + Gemini
TTL_MIN = 45      # self-destruct armed after every command

LEDGER = os.path.join(os.path.dirname(__file__), "..", "..", "..", "evidence", "spend_ledger.jsonl")


def _runpod():
    import runpod
    config.load_env()
    runpod.api_key = os.environ["RUNPOD_API_KEY"]
    return runpod


def balance() -> float:
    from runpod.api.graphql import run_graphql_query
    _runpod()
    r = run_graphql_query("query { myself { clientBalance currentSpendPerHr } }")
    return float(r["data"]["myself"]["clientBalance"])


def spent() -> float:
    return PILOT_START_BALANCE - balance()


def log(event: str, **kw):
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    rec = {"t": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "event": event, **kw}
    with open(LEDGER, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


def guard():
    s = spent()
    if s >= SOFT_CAP:
        log("cap_refused", spent=round(s, 4))
        sys.exit(f"Spend ${s:.2f} >= soft cap ${SOFT_CAP}; refusing new work. Run `pod.py down`.")
    return s


def session(d: str, gpus=None) -> runpod_pod.RunPodSession:
    config.load_env()
    return runpod_pod.RunPodSession(d, keep_pod=TTL_MIN, gpu_types=gpus)


def cmd_up(a):
    guard()
    runpod_pod.CONTAINER_DISK_GB = a.disk
    gpus = [(g, r) for g, r in (x.split("=") for x in a.gpu)]
    gpus = [(g, float(r)) for g, r in gpus]
    s = session(a.dir, gpus)
    s.with_setup_retry(s.ensure_pod)
    s.mark_clean_exit()
    log("pod_up", pod_id=s.pod_id, rate=s.gpu_hourly_rate, disk_gb=a.disk, ssh=f"{s.ssh_host}:{s.ssh_port}")


def _reconnect(d: str) -> runpod_pod.RunPodSession:
    s = session(d)
    if not s.reconnect():
        sys.exit("No live pod in state file.")
    s.mark_clean_exit()
    return s


def cmd_ssh(a):
    guard()
    s = _reconnect(a.dir)
    r = s.ssh_cmd(a.command, timeout=a.timeout)
    sys.stdout.write(r.stdout or "")
    sys.stderr.write(r.stderr or "")
    sys.exit(r.returncode)


def cmd_put(a):
    s = _reconnect(a.dir)
    r = s.scp_to(a.paths[:-1], a.paths[-1], timeout=a.timeout)
    print(r.returncode, (r.stderr or "")[-500:])


def cmd_get(a):
    s = _reconnect(a.dir)
    r = subprocess.run(["scp", "-r", *s._ssh_opts(), "-P", str(s.ssh_port),
                        f"root@{s.ssh_host}:{a.remote}", a.local],
                       capture_output=True, text=True, timeout=a.timeout)
    print(r.returncode, (r.stderr or "")[-500:])


def cmd_spend(a):
    b = balance()
    s = PILOT_START_BALANCE - b
    log("spend", balance=round(b, 4), spent=round(s, 4), hard_cap=HARD_CAP)


def cmd_down(a):
    rp = _runpod()
    s = session(a.dir)
    if s.read_state() and s.pod_id:
        s._owned_pod_ids.add(s.pod_id)
        s.terminate()
        s.remove_state()
    s.sweep_stray_pods()
    pods = rp.get_pods()
    log("pod_down", remaining_pods=[(p["id"], p["name"]) for p in pods], spent=round(spent(), 4))
    if pods:
        sys.exit(f"WARNING: {len(pods)} pod(s) still exist")


def main():
    ap = argparse.ArgumentParser(description="Pod + spend harness for the video-model pilot")
    ap.add_argument("--dir", default="output/pilot0025/encode")
    sub = ap.add_subparsers(dest="action", required=True)
    u = sub.add_parser("up")
    u.add_argument("--disk", type=int, default=250)
    u.add_argument("--gpu", nargs="+", default=["NVIDIA RTX A6000=0.33", "NVIDIA A40=0.35"])
    x = sub.add_parser("ssh")
    x.add_argument("command")
    x.add_argument("--timeout", type=int, default=600)
    p = sub.add_parser("put")
    p.add_argument("paths", nargs="+")
    p.add_argument("--timeout", type=int, default=300)
    g = sub.add_parser("get")
    g.add_argument("remote")
    g.add_argument("local")
    g.add_argument("--timeout", type=int, default=600)
    sub.add_parser("spend")
    sub.add_parser("down")
    a = ap.parse_args()
    {"up": cmd_up, "ssh": cmd_ssh, "put": cmd_put, "get": cmd_get,
     "spend": cmd_spend, "down": cmd_down}[a.action](a)


if __name__ == "__main__":
    main()
