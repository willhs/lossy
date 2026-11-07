"""LTX-2.3 resolution sweep for the cheaper-levers check (research 0025).

Builds the exact production graph (RunPodLtx2Strategy._build_workflow) with
only WIDTH/HEIGHT overridden, submits it to the pilot pod's ComfyUI, and
records wall-clock per shot. Baseline 1280x704 timings come from the real
decode run; this measures what a smaller render saves.

    uv run python tools/model_pilot/run_ltx23_variant.py --size 960x544 --only 247 405 551 900 1890
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import config  # noqa: E402
import runpod_pod  # noqa: E402
from strategies_video import RunPodLtx2Strategy  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="LTX-2.3 resolution sweep")
    ap.add_argument("--size", required=True, help="WxH, both divisible by 32")
    ap.add_argument("--bench", default="output/pilot0025/benchmark.json")
    ap.add_argument("--dir", default="output/pilot0025/encode")
    ap.add_argument("--only", type=int, nargs="*")
    a = ap.parse_args()
    w, h = (int(x) for x in a.size.split("x"))
    config.load_env()

    strat = RunPodLtx2Strategy.__new__(RunPodLtx2Strategy)
    strat.WIDTH, strat.HEIGHT = w, h
    bench = json.load(open(a.bench))
    out_dir = os.path.join(os.path.dirname(a.bench), "clips", f"ltx23-{w}x{h}")
    os.makedirs(out_dir, exist_ok=True)
    tpath = os.path.join(out_dir, "timings.json")
    timings = json.load(open(tpath)) if os.path.exists(tpath) else {}

    s = runpod_pod.RunPodSession(a.dir, keep_pod=45)
    if not s.reconnect():
        sys.exit("no pod")
    s.mark_clean_exit()
    for shot in bench["shots"]:
        idx = shot["index"]
        if a.only and idx not in a.only:
            continue
        n = strat._target_frames(shot["target_s"])
        t0 = time.time()
        hist = s.submit_workflow(strat._build_workflow(shot["prompt"], idx, length=n), timeout=1200)
        dt = time.time() - t0
        item = None
        for out in (hist or {}).get("outputs", {}).values():
            for key in ("images", "gifs", "videos"):
                if out.get(key):
                    item = out[key][0]
        data = s.download_output(item) if item else None
        if data:
            open(os.path.join(out_dir, f"{idx:04d}.webm"), "wb").write(data)
        timings[str(idx)] = {"ok": bool(data), "seconds": round(dt, 1), "frames": n, "out_s": n / 25}
        print(f"  {idx}: {dt:.1f}s for {n} frames ok={bool(data)}", flush=True)
        json.dump(timings, open(tpath, "w"), indent=1)


if __name__ == "__main__":
    main()
