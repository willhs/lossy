"""Research 0035 rule 3: sheet | unswapped | swapped head crops for the swap look test. Free.

    look_crops.py OUT.jpg SCORES_DIR BASE_VIDEO SWAPPED_VIDEO SHEETS_DIR SHOT/POS [SHOT/POS ...]

SCORES_DIR holds the unswapped video's candidates.json and labels.json (score_window.py); each
SHOT/POS (e.g. 1306/first) uses its labelled face box, padded wide enough to show hair and head
shape, cut from the same instant of both videos. The sheet crop is its largest YuNet face.
"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.getcwd(), "docs", "research", "0034-sw4-mixed-shot-keyframes"))
sys.path.insert(0, os.getcwd())
import score_window as sw  # noqa: E402

TILE = 240
START = 1284   # first shot of the window (research 0030's 1284-1321)


def head(img, box, pad=0.9):
    x, y, w, h = box
    p = int(pad * h)
    c = img[max(y - p, 0):y + h + int(0.4 * p), max(x - p, 0):x + w + p]
    return cv2.resize(c, (TILE, TILE)) if c.size else np.zeros((TILE, TILE, 3), np.uint8)


def label(tile, text):
    cv2.rectangle(tile, (0, 0), (TILE, 20), (0, 0, 0), -1)
    cv2.putText(tile, text, (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    return tile


def main():
    out, scores, base_v, swap_v, sheets = sys.argv[1:6]
    picks = sys.argv[6:]
    cand = json.load(open(os.path.join(scores, "candidates.json")))
    labels = json.load(open(os.path.join(scores, "labels.json")))
    run = os.path.dirname(sheets.rstrip("/"))
    shots = {s["index"]: s for s in json.load(open(os.path.join(run, "shots.json")))["shots"]}
    origin = shots[START]["start_s"]   # both videos are the stitched window from shot START
    faces = sw.Faces(os.environ["LOSSY_FACE_MODELS"])
    rows = []
    for key in picks:
        fr = cand["frames"][key]
        box = fr["faces"][labels[key]]["box"]
        s = shots[fr["shot"]]
        t0, t1 = s["start_s"] - origin, s["end_s"] - origin
        t = {"first": t0 + sw.EDGE, "middle": (t0 + t1) / 2, "last": t1 - sw.EDGE}[fr["pos"]]
        sheet = cv2.imread(os.path.join(sheets, f"{fr['character']}_sheet.png"))
        sf = max(faces.find(sheet), key=lambda f: f["box"][3])
        rows.append(np.hstack([label(head(sheet, sf["box"]), f"sheet {fr['character'][:10]}"),
                               label(head(sw.frame(base_v, t), box), f"{key} unswapped"),
                               label(head(sw.frame(swap_v, t), box), f"{key} swapped")]))
    cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"{len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
