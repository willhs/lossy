"""Face consistency of a 2-minute window, pre-registered for research 0034. Free, CPU only.

Research 0030's window_faces.py embedded "the first face YuNet finds" in each single-person
shot. In this window that is often a stormtrooper helmet (YuNet scores them up to 0.88),
Chewbacca (0.82) or the wrong person, and the real film scored as low as the reconstruction.
So this scorer splits finding faces from deciding whose they are:

1. ``candidates``: for every shot whose shot map names exactly one person (non-design
   character; 0030's shot set), take the first, middle and last frame (0.06 s in from each
   cut, scaled to 1280x704), keep every YuNet face (score >= 0.6, box >= 40 px tall), embed
   it with SFace (the pipeline's embedder) and write a crop sheet to label by eye.

       score_window.py candidates --video V --run RUN --start 1284 --limit 38 \
           --models FACE_MODELS --label NAME [--source-offset S] [--keyframes DIR] -o OUT_DIR

   ``--source-offset 0`` means V is the source film (shot k at start_s(k)); otherwise shot k
   is at start_s(k) - start_s(START). ``--keyframes`` also embeds each kf_NNNN.png.

2. ``score``: per character, faces, mean pairwise cosine across shots and mean cosine to the
   sheet, at each frame position, from ``labels.json`` in OUT_DIR: {"<shot>/<pos>": k} picks
   candidate k as the character, null drops the frame (no face of that character in it). An
   unlabelled frame uses the largest candidate ("auto", the no-human-in-the-loop number).

       score_window.py score OUT_DIR --sheets SHEETS_DIR --models FACE_MODELS

Label rule (fixed before the after-run was scored): the pick is the character's own human
face. Not a helmet, droid or Wookiee; not the other named person (Luke is the blond young
man, Han the dark-haired one, Obi-Wan the old bearded man, Leia the young woman). Two
candidates that fit equally: the larger. A face under 60 px tall or turned fully away: null.
"""
import argparse
import itertools
import json
import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.getcwd())
from decode import build_character_shot_map  # noqa: E402

EDGE = 0.06
POSITIONS = ("first", "middle", "last")
SIZE = (1280, 704)


class Faces:
    def __init__(self, models):
        self.det = cv2.FaceDetectorYN.create(os.path.join(models, "yunet.onnx"), "", (320, 320), 0.6)
        self.rec = cv2.FaceRecognizerSF.create(os.path.join(models, "sface.onnx"), "")

    def find(self, img, min_h=40):
        self.det.setInputSize((img.shape[1], img.shape[0]))
        _, found = self.det.detect(img)
        out = []
        for f in (found if found is not None else []):
            if f[3] < min_h:
                continue
            e = self.rec.feature(self.rec.alignCrop(img, f)).flatten()
            out.append({"box": [int(v) for v in f[:4]], "score": round(float(f[-1]), 3),
                        "emb": (e / np.linalg.norm(e)).tolist()})
        return out


def frame(video, t):
    png = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(t, 0):.3f}", "-i", video, "-vf",
                          f"scale={SIZE[0]}:{SIZE[1]}:flags=lanczos", "-frames:v", "1", "-f",
                          "image2pipe", "-vcodec", "png", "-"], capture_output=True, check=True).stdout
    return cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)


def crop(img, box, label):
    x, y, w, h = box
    pad = int(0.3 * h)
    c = img[max(y - pad, 0):y + h + pad, max(x - pad, 0):x + w + pad]
    c = cv2.resize(c, (160, 160)) if c.size else np.zeros((160, 160, 3), np.uint8)
    cv2.rectangle(c, (0, 0), (160, 22), (0, 0, 0), -1)
    cv2.putText(c, label, (3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    return c


def candidates(a):
    faces = Faces(a.models)
    chars = json.load(open(os.path.join(a.run, "characters.json")))
    people = {c["name"] for c in chars["characters"] if not c.get("keep_name")}
    shot_map = build_character_shot_map(chars)
    shots = {s["index"]: s for s in json.load(open(os.path.join(a.run, "shots.json")))["shots"]}
    origin = a.source_offset if a.source_offset is not None else shots[a.start]["start_s"]
    os.makedirs(a.out, exist_ok=True)
    frames, tiles = {}, []
    for idx in range(a.start, a.start + a.limit):
        names = [n for n in shot_map.get(idx, []) if n in people]
        if len(names) != 1:
            continue
        s = shots[idx]
        t0, t1 = s["start_s"] - origin, s["end_s"] - origin
        imgs = {p: frame(a.video, t) for p, t in
                zip(POSITIONS, (t0 + EDGE, (t0 + t1) / 2, t1 - EDGE))}
        if a.keyframes and os.path.exists(os.path.join(a.keyframes, f"kf_{idx:04d}.png")):
            imgs["keyframe"] = cv2.resize(cv2.imread(os.path.join(a.keyframes, f"kf_{idx:04d}.png")), SIZE)
        for pos, img in imgs.items():
            found = faces.find(img)
            frames[f"{idx}/{pos}"] = {"shot": idx, "pos": pos, "character": names[0], "faces": found}
            row = [crop(img, f["box"], f"{idx}/{pos[0]} k{k} {f['box'][3]}px") for k, f in enumerate(found)]
            if not row:
                row = [crop(np.zeros((160, 160, 3), np.uint8), (0, 0, 160, 160), f"{idx}/{pos[0]} none")]
            tiles.extend(row)
    json.dump({"label": a.label, "video": a.video, "frames": frames}, open(os.path.join(a.out, "candidates.json"), "w"))
    cols = 10
    tiles += [np.zeros((160, 160, 3), np.uint8)] * (-len(tiles) % cols)
    sheet = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    cv2.imwrite(os.path.join(a.out, "crops.jpg"), sheet)
    print(f"{len(frames)} frames, {sum(len(f['faces']) for f in frames.values())} faces -> {a.out}")


def mean(xs):
    return float(np.mean(xs)) if xs else float("nan")


def score(a):
    data = json.load(open(os.path.join(a.out, "candidates.json")))
    labels_path = os.path.join(a.out, "labels.json")
    labels = json.load(open(labels_path)) if os.path.exists(labels_path) else {}
    faces = Faces(a.models)
    sheets = {}
    for fr in data["frames"].values():
        name = fr["character"]
        path = os.path.join(a.sheets, f"{name}_sheet.png") if a.sheets else ""
        if name not in sheets and os.path.exists(path):
            found = faces.find(cv2.imread(path))
            sheets[name] = np.array(found[0]["emb"]) if found else None
    result = {}
    for mode in ("auto", "labelled"):
        picks = {}
        for key, fr in data["frames"].items():
            if mode == "labelled" and key in labels:
                k = labels[key]
                face = fr["faces"][k] if k is not None else None
            else:
                face = max(fr["faces"], key=lambda f: f["box"][3]) if fr["faces"] else None
            if face is not None:
                picks.setdefault((fr["character"], fr["pos"]), []).append((fr["shot"], np.array(face["emb"])))
        out = {}
        for (name, pos), got in sorted(picks.items()):
            vecs = [v for _, v in got]
            sheet = sheets.get(name)
            out.setdefault(name, {})[pos] = {
                "faces": len(vecs),
                "across": mean([float(x @ y) for x, y in itertools.combinations(vecs, 2)]),
                "to_sheet": mean([float(v @ sheet) for v in vecs]) if sheet is not None else float("nan"),
                "per_shot_to_sheet": {s: round(float(v @ sheet), 3) for s, v in got} if sheet is not None else {},
            }
        # Pooled (added before the after-run was scored, once the middle frame alone proved
        # thin): each shot's video frames averaged into one face, then compared across shots;
        # and within a shot, first frame vs last frame (how far the clip drifts).
        per_shot: dict = {}
        for (name, pos), got in picks.items():
            if pos == "keyframe":
                continue
            for shot, v in got:
                per_shot.setdefault(name, {}).setdefault(shot, {})[pos] = v
        for name, shots_ in per_shot.items():
            pooled = [sum(d.values()) / np.linalg.norm(sum(d.values())) for d in shots_.values()]
            within = [float(d["first"] @ d["last"]) for d in shots_.values() if "first" in d and "last" in d]
            out.setdefault(name, {})["pooled"] = {
                "faces": len(pooled),
                "across": mean([float(x @ y) for x, y in itertools.combinations(pooled, 2)]),
                "to_sheet": mean([float(v @ sheets[name]) for v in pooled])
                if sheets.get(name) is not None else float("nan"),
                "first_vs_last": mean(within)}
        result[mode] = out
    result["labelled_frames"] = len(labels)
    json.dump(result, open(os.path.join(a.out, "score.json"), "w"), indent=1)
    for mode in ("auto", "labelled"):
        print(f"[{data['label']} {mode}] character: faces/across/to-sheet at " + ", ".join(
            p for p in POSITIONS + ("keyframe", "pooled")))
        for name, by_pos in result[mode].items():
            cells = "  ".join(f"{p[0]}:{v['faces']}/{v['across']:.2f}/{v['to_sheet']:.2f}"
                              for p in POSITIONS + ("keyframe", "pooled") if (v := by_pos.get(p)))
            if "pooled" in by_pos:
                cells += f"  first-vs-last {by_pos['pooled']['first_vs_last']:.2f}"
            print(f"[{data['label']} {mode}] {name:<22} {cells}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("candidates")
    c.add_argument("--video", required=True)
    c.add_argument("--run", required=True)
    c.add_argument("--start", type=int, required=True)
    c.add_argument("--limit", type=int, required=True)
    c.add_argument("--models", required=True)
    c.add_argument("--label", required=True)
    c.add_argument("--source-offset", type=float, default=None)
    c.add_argument("--keyframes", default=None)
    c.add_argument("-o", "--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("out")
    s.add_argument("--sheets", default=None)
    s.add_argument("--models", required=True)
    a = ap.parse_args()
    candidates(a) if a.cmd == "candidates" else score(a)


if __name__ == "__main__":
    main()
