"""Measured pilot timings -> per-shot cost and projected full-run cost (research 0025).

Cost tracks generated SECONDS, not shot count (0024 showed a per-shot
extrapolation underestimates by ~35%), so the unit here is GPU-seconds per
second of output, measured over the warm shots of each run (the first shot
carries model load and is reported separately).

    uv run python tools/model_pilot/project.py output/pilot0025 ltx23 ltx25-base k6
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re

# Runtime targets (seconds of film, shot count, mean shot) from the encodes.
TARGETS = {
    "star_wars_iv": (124.7 * 60, 2070),
    "seinfeld_s01e03": (1325.6, 343),
}
# RunPod community on-demand $/hr, read off runpod.io/pricing 2026-11-07.
GPU_RATES = {"A6000 community": 0.33, "A6000 secure": 0.59, "L40S community": 0.79,
             "RTX 4090 community": 0.34, "RTX 5090 community": 0.69}
SETUP_MIN = 20  # per pod: model download + load (0023/0024: 15-20 min)


def ltx23_timings(root: str) -> dict:
    """Recover per-shot wall-clock from the timestamped decode log."""
    log = os.path.join(root, "logs", "ltx23_baseline.log")
    bench = {s["index"]: s for s in json.load(open(os.path.join(root, "benchmark.json")))["shots"]}
    starts = []
    end = None
    for line in open(log):
        m = re.match(r"(\d+)\s+Shot (\d+) ", line)
        if m:
            starts.append((int(m.group(1)), int(m.group(2))))
        if re.match(r"\d+\s+Done\.", line):
            end = int(line.split()[0])
    out = {}
    for i, (t, idx) in enumerate(starts):
        t_next = starts[i + 1][0] if i + 1 < len(starts) else end
        if t_next is None:
            continue
        n = max(25, min(121, round(bench[idx]["duration_s"] * 25)))
        out[str(idx)] = {"ok": True, "seconds": t_next - t, "frames": n, "out_s": n / 25}
    return out


def load(root: str, model: str) -> dict:
    if model == "ltx23":
        return ltx23_timings(root)
    return json.load(open(os.path.join(root, "clips", model, "timings.json")))


def rate(timings: dict) -> dict:
    rows = [(k, v) for k, v in timings.items() if not k.startswith("_") and v.get("ok")]
    order = list(timings)  # insertion order = render order
    rows.sort(key=lambda kv: order.index(kv[0]))
    warm = rows[1:] if len(rows) > 2 else rows
    gpu_s = sum(v["seconds"] for _, v in warm)
    out_s = sum(v["out_s"] for _, v in warm)
    return {
        "ok_shots": len(rows),
        "failed": sum(1 for k, v in timings.items() if not k.startswith("_") and not v.get("ok")),
        "first_shot_s": rows[0][1]["seconds"] if rows else None,
        "warm_s_per_shot": round(gpu_s / max(len(warm), 1), 1),
        "gpu_s_per_out_s": round(gpu_s / out_s, 2) if out_s else None,
        "out_s_mean": round(out_s / max(len(warm), 1), 2),
    }


def project(r: dict, film_s: float, shots: int, usd_hr: float, native_5s: bool = False) -> dict:
    if native_5s:  # fixed-length model: every part is a full 5s render
        out_s = shots * 5.0 * 1.15  # ~15% of shots split into 2+ parts (0023/0024)
    else:
        out_s = film_s * 1.03  # frame rounding + MIN_FRAMES floor on sub-1s shots
    gpu_h = out_s * r["gpu_s_per_out_s"] / 3600
    pods = max(1, math.ceil(gpu_h / 12))  # assume a fresh pod every ~12h
    setup_h = pods * SETUP_MIN / 60
    return {"gpu_hours": round(gpu_h + setup_h, 1), "usd": round((gpu_h + setup_h) * usd_hr, 2),
            "wall_hours_1gpu": round(gpu_h + setup_h, 1)}


def main():
    ap = argparse.ArgumentParser(description="Pilot timings -> projected cost")
    ap.add_argument("root")
    ap.add_argument("models", nargs="+")
    ap.add_argument("--native-5s", nargs="*", default=[], help="models billed as fixed 5s parts")
    a = ap.parse_args()
    report = {}
    for m in a.models:
        try:
            r = rate(load(a.root, m))
        except FileNotFoundError:
            continue
        r["projections"] = {
            t: {g: project(r, fs, n, usd, m in a.native_5s) for g, usd in GPU_RATES.items()}
            for t, (fs, n) in TARGETS.items()
        }
        report[m] = r
    json.dump(report, open(os.path.join(a.root, "projections.json"), "w"), indent=1)
    for m, r in report.items():
        sw = r["projections"]["star_wars_iv"]["A6000 community"]
        ep = r["projections"]["seinfeld_s01e03"]["A6000 community"]
        print(f"{m:14s} ok={r['ok_shots']:2d} fail={r['failed']} first={r['first_shot_s']}s "
              f"warm={r['warm_s_per_shot']}s/shot  {r['gpu_s_per_out_s']} GPU-s/out-s  "
              f"| SW-IV ${sw['usd']} ({sw['gpu_hours']} GPU-h)  | episode ${ep['usd']} ({ep['gpu_hours']} GPU-h)")


if __name__ == "__main__":
    main()
