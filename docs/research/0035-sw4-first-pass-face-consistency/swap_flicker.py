"""Research 0035: does the swap toggle on and off inside a shot? Free, CPU only.

The swap only fires on frames whose best face clears MIN_SIM, so a shot whose faces hover near it
can alternate swapped and unswapped faces frame to frame. Per frame, a frame counts as swapped
when the unswapped and swapped videos differ locally (largest 31x31-mean absolute difference >
THRESH; the re-encode alone measures 3-4, a swapped face 17-50 on this window); runs of
swapped/unswapped frames shorter than 0.5 s inside a shot are what a viewer sees as flicker.

    swap_flicker.py BASE_VIDEO SWAPPED_VIDEO RUN_DIR START SHOT [SHOT ...] [--window N] [--thresh T]

With no flags this is the measure 0035 reported. The fix (see research.md) added two flags:
``--window N`` (the video holds N shots from START) takes each shot's frames from the video's own
cuts (tools/face_swap.py) instead of shots.json times, which are up to 3 frames out here and so
count a neighbour's frames; ``--thresh T`` moves the swapped/unswapped line (default 10).
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.getcwd(), "tools"))
import face_swap  # noqa: E402

THRESH = 10.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_v")
    ap.add_argument("swap_v")
    ap.add_argument("run")
    ap.add_argument("start", type=int)
    ap.add_argument("wanted", type=int, nargs="+")
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--thresh", type=float, default=THRESH)
    args = ap.parse_args()
    base_v, swap_v, run, start, wanted = args.base_v, args.swap_v, args.run, args.start, args.wanted
    shots = {s["index"]: s for s in json.load(open(os.path.join(run, "shots.json")))["shots"]}
    origin = shots[start]["start_s"]
    spans = face_swap.shot_spans(base_v, run, start, args.window) if args.window else None
    a, b = cv2.VideoCapture(base_v), cv2.VideoCapture(swap_v)
    fps = a.get(cv2.CAP_PROP_FPS)
    flags: dict = {}
    n = 0
    while True:
        ok1, f1 = a.read()
        ok2, f2 = b.read()
        if not (ok1 and ok2):
            break
        t = n / fps
        for s in wanted:
            inside = (spans[s][0] <= n < spans[s][1]) if spans else (
                shots[s]["start_s"] - origin <= t < shots[s]["end_s"] - origin)
            if inside:
                d = np.abs(f1.astype(np.int16) - f2.astype(np.int16)).astype(np.float32).mean(2)
                flags.setdefault(s, []).append(float(cv2.blur(d, (31, 31)).max()) > args.thresh)
        n += 1
    print(f"{'shot':<6}{'frames':>7}{'swapped':>8}{'toggles':>8}{'short runs (<0.5 s)':>21}")
    for s in wanted:
        f = flags.get(s, [])
        runs, cur = [], 1
        for x, y in zip(f, f[1:]):
            if x == y:
                cur += 1
            else:
                runs.append(cur)
                cur = 1
        runs.append(cur)
        toggles = len(runs) - 1
        short = sum(1 for r in runs[1:-1] if r < 0.5 * fps)
        print(f"{s:<6}{len(f):>7}{sum(f):>8}{toggles:>8}{short:>21}")


if __name__ == "__main__":
    main()
