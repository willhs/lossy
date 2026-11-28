"""Face consistency of a rendered window (task 462c92a3), with the pipeline's own embedder.

Usage (from the repo root, after window_test.sh):
    uv run python docs/research/0030-sw4-full-run-preflight/window_faces.py RUN_DIR START LIMIT MODELS_DIR

For every shot whose characters.json entry names exactly one sheeted person, takes the
middle frame of its first clip, embeds the first face YuNet finds (SFace), and prints per
character: shots with a face, mean pairwise cosine across shots, and mean cosine to the
character's reference sheet (<RUN_DIR>/characters/<name>_sheet.png). Free, CPU only.

Superseded by research 0034's score_window.py: "the first face YuNet finds" is often a helmet or
Chewbacca. The rule is kept as it was so this reproduces the table in research.md; do not use it
for new scoring (faces.pick is the rule).
"""
import glob
import itertools
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
import faces as fid  # noqa: E402
from decode import build_character_shot_map  # noqa: E402

run, start, limit, models = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
emb = fid.FaceIdentity(models)
chars = json.load(open(os.path.join(run, "characters.json")))
shot_map = build_character_shot_map(chars)
clips = glob.glob(os.path.join(run, "clips", "*", "*.mp4"))
by_shot = {}
for c in sorted(clips):
    idx = int(os.path.basename(c).split(".")[0].split("-")[0])
    by_shot.setdefault(idx, c)


def middle_frame(path):
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                "-of", "csv=p=0", path], capture_output=True, text=True).stdout)
    return subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{dur / 2:.2f}", "-i", path,
                           "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
                          capture_output=True).stdout


per_char = {}
for idx in range(start, start + limit):
    names = [n for n in shot_map.get(idx, [])
             if os.path.exists(os.path.join(run, "characters", f"{n}_sheet.png"))]
    if len(names) != 1 or idx not in by_shot:
        continue
    faces = emb.faces(middle_frame(by_shot[idx]))
    if faces:
        per_char.setdefault(names[0], []).append(faces[0])

print(f"{'character':<24} shots  across-shots  to-sheet")
for name, vecs in sorted(per_char.items()):
    sheet = emb.faces(open(os.path.join(run, "characters", f"{name}_sheet.png"), "rb").read())
    pairs = [float(a @ b) for a, b in itertools.combinations(vecs, 2)]
    to_sheet = [float(v @ sheet[0]) for v in vecs] if sheet else []
    print(f"{name:<24} {len(vecs):>5}  {np.mean(pairs) if pairs else float('nan'):>12.3f}"
          f"  {np.mean(to_sheet) if to_sheet else float('nan'):>8.3f}")
