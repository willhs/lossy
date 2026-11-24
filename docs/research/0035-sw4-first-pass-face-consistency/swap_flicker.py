"""Research 0035: does the swap toggle on and off inside a shot? Free, CPU only.

The swap only fires on frames whose best face clears MIN_SIM, so a shot whose faces hover near it
can alternate swapped and unswapped faces frame to frame. Per frame, a frame counts as swapped
when the unswapped and swapped videos differ locally (largest 31x31-mean absolute difference >
THRESH; the re-encode alone measures 3-4, a swapped face 17-50 on this window); runs of
swapped/unswapped frames shorter than 0.5 s inside a shot are what a viewer sees as flicker.

    swap_flicker.py BASE_VIDEO SWAPPED_VIDEO RUN_DIR START SHOT [SHOT ...]
"""
import json
import os
import sys

import cv2
import numpy as np

THRESH = 10.0


def main():
    base_v, swap_v, run, start = sys.argv[1:5]
    wanted = [int(s) for s in sys.argv[5:]]
    shots = {s["index"]: s for s in json.load(open(os.path.join(run, "shots.json")))["shots"]}
    origin = shots[int(start)]["start_s"]
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
            if shots[s]["start_s"] - origin <= t < shots[s]["end_s"] - origin:
                d = np.abs(f1.astype(np.int16) - f2.astype(np.int16)).astype(np.float32).mean(2)
                flags.setdefault(s, []).append(float(cv2.blur(d, (31, 31)).max()) > THRESH)
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
