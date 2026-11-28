"""Carry by-eye face labels from one video's score_window.py candidates to another's. Free.

A swap does not move faces, so the face labelled in the unswapped video is the candidate in the
same place in the swapped one (box overlap >= 0.5); none there means the frame is dropped.

    carry_labels.py LABELLED_SCORES_DIR NEW_SCORES_DIR [--check HAND_LABELS.json]

Writes NEW_SCORES_DIR/labels.json. ``--check`` first reports where it differs from hand labels.
"""
import json
import os
import sys


def overlap(a, b):
    """IoU of two (x, y, w, h) boxes."""
    w = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    h = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    both = w * h
    return both / (a[2] * a[3] + b[2] * b[3] - both)


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    src, dst = sys.argv[1:3]
    old = load(os.path.join(src, "candidates.json"))["frames"]
    new = load(os.path.join(dst, "candidates.json"))["frames"]
    labels = load(os.path.join(src, "labels.json"))
    carried = {}
    for key, k in labels.items():
        carried[key] = None
        if k is None or key not in new:
            continue
        box = old[key]["faces"][k]["box"]
        hits = [(overlap(box, f["box"]), i) for i, f in enumerate(new[key]["faces"])]
        if hits and max(hits)[0] >= 0.5:
            carried[key] = max(hits)[1]
    if "--check" in sys.argv:
        hand = load(sys.argv[sys.argv.index("--check") + 1])
        differs = {k: (carried.get(k), v) for k, v in hand.items() if carried.get(k) != v}
        print("differs from hand labels:", differs or "none", f"(of {len(hand)})")
    with open(os.path.join(dst, "labels.json"), "w") as f:
        json.dump(carried, f)
    found = sum(v is not None for v in carried.values())
    print(f"{found} faces labelled of {len(carried)} frames -> {dst}/labels.json")


if __name__ == "__main__":
    main()
