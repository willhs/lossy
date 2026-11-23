"""Identity grid for research 0034: each character's labelled face, shot by shot, one row per
version (e.g. source, old window, new window). Reads score_window.py's candidates.json,
labels.json and faces/ from each DIR. Free.

    uv run python docs/research/0034-sw4-mixed-shot-keyframes/identity_grid.py OUT.jpg NAME=DIR ...

A cell is the labelled middle-frame face, else first, else last; grey if the shot has none.
"""
import json
import os
import sys

import cv2
import numpy as np

out, versions = sys.argv[1], [v.split("=", 1) for v in sys.argv[2:]]
data = {}
for name, d in versions:
    cand = json.load(open(os.path.join(d, "candidates.json")))["frames"]
    lp = os.path.join(d, "labels.json")
    labels = json.load(open(lp)) if os.path.exists(lp) else {}
    data[name] = (d, cand, labels)
shots_by_char: dict = {}
for _, cand, _ in data.values():
    for fr in cand.values():
        shots_by_char.setdefault(fr["character"], set()).add(fr["shot"])

C = 128


def cell(d, cand, labels, shot):
    for pos in ("middle", "first", "last"):
        key = f"{shot}/{pos}"
        fr = cand.get(key)
        if not fr or not fr["faces"]:
            continue
        k = labels.get(key, max(range(len(fr["faces"])), key=lambda i: fr["faces"][i]["box"][3]))
        if k is None:
            continue
        img = cv2.imread(os.path.join(d, "faces", f"{shot}_{pos}_{k}.jpg"))
        if img is not None:
            return cv2.resize(img, (C, C))
    return np.full((C, C, 3), 60, np.uint8)


def label(text, w, h=24):
    bar = np.zeros((h, w, 3), np.uint8)
    cv2.putText(bar, text, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return bar


blocks = []
width = 110 + C * max(len(s) for s in shots_by_char.values())
for char, shots in sorted(shots_by_char.items()):
    shots = sorted(shots)
    rows = [label(char + "   shots: " + "  ".join(str(s) for s in shots), width)]
    for name, (d, cand, labels) in data.items():
        row = np.hstack([cell(d, cand, labels, s) for s in shots])
        tag = label(name, 110, C)
        row = np.hstack([tag, row])
        rows.append(np.hstack([row, np.zeros((C, width - row.shape[1], 3), np.uint8)]))
    blocks.append(np.vstack(rows))
cv2.imwrite(out, np.vstack(blocks))
print(out)
