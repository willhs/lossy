"""Research 0035's by-eye damage check, made inspectable. Free, CPU only.

For each keyframe shot, from each arm's clip: a filmstrip of 6 evenly spaced frames, plus two
numbers that flag the damage the pre-registration names:
  motion  mean absolute frame-to-frame difference (0-255) over the clip: a frozen clip is low;
  snap    the largest frame-to-frame difference in the clip's last quarter, divided by the clip's
          median one: a pop back to the keyframe near the end shows as a spike.

    anchor_eye_check.py OUT.jpg RUN_DIR[:label] RUN_DIR[:label] ... --shots 1285 1286 ...
"""
import argparse
import os

import cv2
import numpy as np

KF = "runpod-ltx25-keyframe"


def frames(path):
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    return out


def stats(fs):
    g = [cv2.cvtColor(cv2.resize(f, (320, 176)), cv2.COLOR_BGR2GRAY).astype(np.float32) for f in fs]
    d = np.array([np.abs(a - b).mean() for a, b in zip(g, g[1:])])
    tail = d[int(len(d) * 0.75):]
    return float(d.mean()), float(tail.max() / max(np.median(d), 1e-3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--shots", nargs="+", type=int, required=True)
    a = ap.parse_args()
    runs = [(r.split(":")[0], r.split(":")[1] if ":" in r else os.path.basename(r)) for r in a.runs]
    rows = []
    print(f"{'shot':<6}" + "".join(f"{lab:>22}" for _, lab in runs))
    for s in a.shots:
        strips, cells = [], []
        for run, lab in runs:
            fs = frames(os.path.join(run, "clips", KF, f"{s:04d}.mp4"))
            motion, snap = stats(fs)
            cells.append(f"motion {motion:5.2f} snap {snap:4.1f}")
            picks = [fs[round(i * (len(fs) - 1) / 5)] for i in range(6)]
            strip = np.hstack([cv2.resize(f, (213, 117)) for f in picks])
            cv2.rectangle(strip, (0, 0), (300, 18), (0, 0, 0), -1)
            cv2.putText(strip, f"{s} {lab} m{motion:.1f} s{snap:.1f}", (3, 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
            strips.append(strip)
        print(f"{s:<6}" + "".join(f"{c:>22}" for c in cells))
        rows.append(np.vstack(strips + [np.zeros((6, strips[0].shape[1], 3), np.uint8)]))
    cv2.imwrite(a.out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 82])


if __name__ == "__main__":
    main()
