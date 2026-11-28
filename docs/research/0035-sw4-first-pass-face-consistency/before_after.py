"""Side-by-side clip: per-frame swap (before) | per-shot swap (after), chosen shots in order. Free.

    before_after.py BASE_VIDEO BEFORE_VIDEO AFTER_VIDEO RUN_DIR OUT.mp4 SHOT [SHOT ...]

The video holds shots 1284-1321; each shot's frames come from the base video's own cuts. The
shot number is drawn on every frame, with each side's label.
"""
import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.getcwd(), "tools"))
import face_swap  # noqa: E402

W, H = 640, 352


def tile(frame, text):
    t = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
    cv2.rectangle(t, (0, 0), (W, 26), (0, 0, 0), -1)
    cv2.putText(t, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    return t


def main():
    base, before, after, run, out = sys.argv[1:6]
    spans = face_swap.shot_spans(base, run, 1284, 38)
    wanted = [int(s) for s in sys.argv[6:]]
    a, b = cv2.VideoCapture(before), cv2.VideoCapture(after)
    fps = a.get(cv2.CAP_PROP_FPS)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{2 * W}x{H}",
                            "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", out],
                           stdin=subprocess.PIPE)
    n = written = 0
    while True:
        ok1, f1 = a.read()
        ok2, f2 = b.read()
        if not (ok1 and ok2):
            break
        shot = next((s for s in wanted if spans[s][0] <= n < spans[s][1]), None)
        if shot is not None:
            enc.stdin.write(np.hstack([tile(f1, f"{shot}  BEFORE: decided per frame"),
                                       tile(f2, f"{shot}  AFTER: decided per shot")]).tobytes())
            written += 1
        n += 1
    enc.stdin.close()
    enc.wait()
    print(f"{written} frames ({written / fps:.1f} s) -> {out}")


if __name__ == "__main__":
    main()
