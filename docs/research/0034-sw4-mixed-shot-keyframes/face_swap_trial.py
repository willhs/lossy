"""Research 0034 step 4: a HyperSwap face-swap post-pass on a stitched window. Free, Mac CPU.

Superseded by tools/face_swap.py: this trial decides per frame, so faces flicker (research 0035).
Kept as the record of how 0034's and 0035's swapped videos were made.

Uses screenquote's engine (docs/research/0028 there; SPEC-021): SCRFD detect, ArcFace identity,
HyperSwap 1b at 256 px, BiSeNet face mask. Run with screenquote's venv, which has onnxruntime
(lossy does not, and this trial adds no dependency to it). The module file is loaded directly,
and nothing is written into the screenquote checkout:

    PYTHONDONTWRITEBYTECODE=1 /Users/will/projects/screenquote/.venv/bin/python \
        docs/research/0034-sw4-mixed-shot-keyframes/face_swap_trial.py \
        VIDEO RUN_DIR START LIMIT MODELS_DIR OUT.mp4

Scope (a trial, not a pipeline stage): only shots whose shot map names exactly one person (the
shots score_window.py scores). The identity is that person's reference sheet (its largest SCRFD
face). In each frame the target is the face most like the sheet by ArcFace, and only if that
cosine is >= MIN_SIM, so helmets, droids and Chewbacca are left alone. Audio is copied as is.
Weights are facefusion's research / non-commercial ones (screenquote ADR 0012).
"""
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ENGINE = "/Users/will/projects/screenquote/screenquote_web/face_swap_engine.py"
MIN_SIM = 0.10   # ArcFace cosine to the sheet a face needs to be swapped: strangers ~0-0.2, helmets lower

spec = importlib.util.spec_from_file_location("face_swap_engine", ENGINE)
fse = importlib.util.module_from_spec(spec)
sys.modules["face_swap_engine"] = fse
spec.loader.exec_module(fse)

video, run, start, limit, models, out = sys.argv[1:7]
start, limit = int(start), int(limit)
engine = fse.OnnxEngine(Path(models), threads=6)

chars = json.load(open(os.path.join(run, "characters.json")))
people = {c["name"] for c in chars["characters"] if not c.get("keep_name")}
shot_map: dict = {}
for c in chars["characters"]:
    for s in c.get("shots", []):
        shot_map.setdefault(s, []).append(c["name"])
shots = {s["index"]: s for s in json.load(open(os.path.join(run, "shots.json")))["shots"]}
origin = shots[start]["start_s"]

spans, identities = [], {}
for idx in range(start, start + limit):
    names = [n for n in shot_map.get(idx, []) if n in people]
    if len(names) != 1:
        continue
    name = names[0]
    if name not in identities:
        sheet = cv2.imread(os.path.join(run, "characters", f"{name}_sheet.png"))
        faces = engine.detect(sheet)
        if not faces:
            continue
        best = max(faces, key=lambda f: (f.box[2] - f.box[0]) * (f.box[3] - f.box[1]))
        identities[name] = engine.embed(sheet, best)
    spans.append((shots[idx]["start_s"] - origin, shots[idx]["end_s"] - origin, idx, name))

cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS)
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
silent = out + ".video.mp4"
enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                        "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", silent],
                       stdin=subprocess.PIPE)
stats: dict = {}
n, t0 = 0, time.time()
while True:
    ok, frame = cap.read()
    if not ok:
        break
    t = n / fps
    span = next((s for s in spans if s[0] <= t < s[1]), None)
    if span:
        idx, name = span[2], span[3]
        st = stats.setdefault(idx, {"character": name, "frames": 0, "swapped": 0, "sims": []})
        st["frames"] += 1
        faces = [f for f in engine.detect(frame) if f.box[3] - f.box[1] >= 40]
        if faces:
            sims = [float(engine.embed(frame, f) @ identities[name]) for f in faces]
            k = int(np.argmax(sims))
            st["sims"].append(round(sims[k], 3))
            if sims[k] >= MIN_SIM:
                frame = engine.swap_aligned(frame, faces[k].landmarks, identities[name])[0]
                st["swapped"] += 1
    enc.stdin.write(frame.tobytes())
    n += 1
enc.stdin.close()
enc.wait()
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", silent, "-i", video, "-map", "0:v", "-map", "1:a?",
                "-c:v", "copy", "-c:a", "copy", out], check=True)
os.remove(silent)
for st in stats.values():
    st["median_sim"] = float(np.median(st["sims"])) if st["sims"] else None
    del st["sims"]
json.dump({"frames": n, "seconds": round(time.time() - t0, 1), "shots": stats}, open(out + ".json", "w"), indent=1)
print(f"{n} frames in {time.time() - t0:.0f} s; swapped "
      f"{sum(s['swapped'] for s in stats.values())} of {sum(s['frames'] for s in stats.values())} single-person frames")
