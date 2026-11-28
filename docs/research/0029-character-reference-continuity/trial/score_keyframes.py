"""Keyframe-stage face cosine to the sheet, per character, for one or more decode output dirs (task 0017).

Same measure as score.py's "keyframe" column of `k_decay_vs_sheet`: a sheet's signature is the
mean of every face found in it; a keyframe's score for a character is the best cosine of its two
largest faces to that signature; a character's figure is the mean over its keyframes. Reads the
`characters/<name>_sheet.png` and `keyframes/kf_NNNN.{png,json}` caches a keyframe run leaves.

Also writes a contact sheet: one row per shot, one column per run (labelled), sheets on top.
No network, no spend.

    uv run python .../score_keyframes.py --models models/face --out OUT_DIR pod=RUN_A fal=RUN_B
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.getcwd())
import faces as fid  # noqa: E402


def find(fi: fid.FaceIdentity, img) -> list[np.ndarray]:
    """Unit embeddings, largest face first."""
    return [f.emb for f in fid.by_area(fi.find(img))]


def score_run(run: str, faces: fid.FaceIdentity) -> dict:
    sheets = {}
    for p in glob.glob(os.path.join(run, "characters", "*_sheet.png")):
        fs = find(faces, cv2.imread(p))
        if fs:
            m = np.mean(fs, axis=0)
            sheets[os.path.basename(p)[:-len("_sheet.png")]] = m / np.linalg.norm(m)
    per_char: dict[str, list[float]] = {}
    per_shot = {}
    for meta_path in sorted(glob.glob(os.path.join(run, "keyframes", "kf_*.json"))):
        meta = json.load(open(meta_path))
        fs = find(faces, cv2.imread(meta_path[:-5] + ".png"))[:2]
        row = {}
        for c in meta["characters"]:
            if c in sheets and fs:
                v = max(float(np.dot(f, sheets[c])) for f in fs)
                per_char.setdefault(c, []).append(v)
                row[c] = round(v, 4)
        per_shot[meta["shot"]] = {"characters": meta["characters"], "cosine": row, "faces": len(fs)}
    means = {c: round(float(np.mean(v)), 4) for c, v in per_char.items()}
    allv = [x for v in per_char.values() for x in v]
    return {"sheets": sorted(sheets), "per_character": means,
            "n_per_character": {c: len(v) for c, v in per_char.items()},
            "mean_over_keyframes": round(float(np.mean(allv)), 4) if allv else None,
            "mean_of_characters": round(float(np.mean(list(means.values()))), 4) if means else None,
            "per_shot": per_shot}


def contact_sheet(runs: dict[str, str], out: str, tile=(320, 176)) -> None:
    shots = sorted({int(os.path.basename(p)[3:7]) for r in runs.values()
                    for p in glob.glob(os.path.join(r, "keyframes", "kf_*.png"))})
    names = sorted({os.path.basename(p)[:-len("_sheet.png")] for r in runs.values()
                    for p in glob.glob(os.path.join(r, "characters", "*_sheet.png"))})
    label_w, head = 70, 24
    sheet_h = tile[0] // 2
    W = label_w + tile[0] * len(runs)
    H = head + len(names) * sheet_h + len(shots) * tile[1]
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    for j, label in enumerate(runs):
        d.text((label_w + j * tile[0] + 4, 4), label, fill="black")
    y = head
    for n in names:
        d.text((2, y + 4), n[:10], fill="black")
        for j, r in enumerate(runs.values()):
            p = os.path.join(r, "characters", f"{n}_sheet.png")
            if os.path.exists(p):
                canvas.paste(Image.open(p).convert("RGB").resize((sheet_h, sheet_h)), (label_w + j * tile[0], y))
        y += sheet_h
    for s in shots:
        d.text((2, y + 4), str(s), fill="black")
        for j, r in enumerate(runs.values()):
            p = os.path.join(r, "keyframes", f"kf_{s:04d}.png")
            if os.path.exists(p):
                canvas.paste(Image.open(p).convert("RGB").resize(tile), (label_w + j * tile[0], y))
        y += tile[1]
    canvas.save(out, quality=85)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="models/face")
    ap.add_argument("--out", required=True)
    ap.add_argument("runs", nargs="+", help="label=decode_output_dir")
    a = ap.parse_args()
    runs = dict(r.split("=", 1) for r in a.runs)
    os.makedirs(a.out, exist_ok=True)
    faces = fid.FaceIdentity(a.models)
    result = {label: score_run(path, faces) for label, path in runs.items()}
    json.dump(result, open(os.path.join(a.out, "keyframe_scores.json"), "w"), indent=1)
    contact_sheet(runs, os.path.join(a.out, "keyframes_contact.jpg"))
    for label, r in result.items():
        print(label, json.dumps({k: r[k] for k in ("per_character", "n_per_character",
                                                   "mean_over_keyframes", "mean_of_characters")}))


if __name__ == "__main__":
    main()
