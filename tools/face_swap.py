"""HyperSwap face-swap post-pass on a stitched window, decided once per shot. Free, Mac CPU.

Research 0034's trial decided per frame (swap if this frame's best face is MIN_SIM like the
sheet), so a face hovering near the line switched between two people several times a second
(research 0035). Here a shot is swapped on every frame or on none.

Uses screenquote's engine (SCRFD detect, ArcFace identity, HyperSwap 1b, BiSeNet mask), so run
it with screenquote's venv; lossy takes no onnxruntime dependency and nothing is written into
that checkout:

    PYTHONDONTWRITEBYTECODE=1 LOSSY_FACE_SWAP_ENGINE=<screenquote>/screenquote_web/face_swap_engine.py \
        <screenquote>/.venv/bin/python tools/face_swap.py \
        VIDEO RUN_DIR START LIMIT MODELS_DIR OUT.mp4

Only shots whose shot map names exactly one person (non-design character) are considered; the
identity is that person's reference sheet. Writes OUT.mp4 (audio copied) and OUT.mp4.json (per
shot: frame range, whether it was swapped and why). Weights are facefusion's research /
non-commercial ones (screenquote ADR 0012).
"""
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import faces as fid  # noqa: E402

MIN_SIM = 0.10      # ArcFace cosine to the sheet a face must reach in some frame: strangers and helmets sit below
MIN_FACE_H = 40     # px; smaller faces are not followed
MIN_OVERLAP = 0.3   # box IoU between consecutive frames that makes it the same face
CUT_SEARCH = 5      # frames either side of a shot's start time; stitched cuts land within 3


@dataclass(frozen=True)
class Seen:
    """One face in one frame."""

    box: tuple
    landmarks: object
    sim: float   # cosine to the character's sheet


def find_cuts(diffs, expected):
    """The real first frame of each shot: the biggest frame-to-frame change near its start time.

    ``diffs[n]`` is the change from frame n-1 to n. The stitch lands cuts a few frames off
    ``shots.json``, and a neighbour's frame inside a shot would be swapped (or break) with it.
    """
    cuts, last = [], 0
    for n in expected:
        window = range(max(last + 1, n - CUT_SEARCH), min(len(diffs), n + CUT_SEARCH + 1))
        last = max(window, key=lambda k: diffs[k], default=n)
        cuts.append(last)
    return cuts


def _overlap(a, b):
    w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    both = w * h
    either = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - both
    return both / either if either else 0.0


def full_tracks(frames):
    """Faces that can be followed through every frame of a shot, one ``Seen`` per frame each."""
    tracks = [[face] for face in frames[0]] if frames else []
    for faces in frames[1:]:
        followed = []
        for track in tracks:
            near = max(faces, key=lambda f: _overlap(track[-1].box, f.box), default=None)
            if near is not None and _overlap(track[-1].box, near.box) >= MIN_OVERLAP:
                followed.append(track + [near])
        tracks = followed
    return tracks


def decide_shot(frames):
    """Which face to swap, for a whole shot: ``(one Seen per frame, reason)`` or ``(None, reason)``.

    There is no per-frame answer to give. The character is the followed face most like the sheet
    over the shot, and it is swapped if it reaches MIN_SIM at all.
    """
    tracks = full_tracks(frames)
    if not tracks:
        return None, "no face on every frame"
    best = max(tracks, key=lambda t: float(np.median([f.sim for f in t])))
    if max(f.sim for f in best) < MIN_SIM:
        return None, "never reaches the similarity line"
    return best, "swapped"


def swap_shot(frames, engine, identity):
    """The shot's frames, all swapped or all as they came, and what was decided."""
    seen = [[Seen(tuple(f.box), f.landmarks, float(engine.embed(frame, f) @ identity))
             for f in engine.detect(frame) if f.box[3] - f.box[1] >= MIN_FACE_H] for frame in frames]
    track, reason = decide_shot(seen)
    out = frames if track is None else [engine.swap_aligned(frame, face.landmarks, identity)[0]
                                        for frame, face in zip(frames, track)]
    sims = [f.sim for f in track] if track else [max(f.sim for f in faces) for faces in seen if faces]
    return out, {"frames": len(frames), "swapped": 0 if track is None else len(frames), "reason": reason,
                 "median_sim": round(float(np.median(sims)), 3) if sims else None,
                 "max_sim": round(max(sims), 3) if sims else None}


def _read(video):
    cap = cv2.VideoCapture(video)
    while True:
        ok, frame = cap.read()
        if not ok:
            return
        yield frame


def frame_diffs(video):
    """Mean change from each frame to the next, on a small grey copy."""
    diffs, last = [], None
    for frame in _read(video):
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 88),
                           interpolation=cv2.INTER_AREA).astype(np.float32)
        diffs.append(0.0 if last is None else float(np.abs(small - last).mean()))
        last = small
    return diffs


def shot_spans(video, run, start, limit):
    """``{shot index: (first frame, end frame)}`` for the window as the video really cuts."""
    shots = {s["index"]: s for s in json.load(open(os.path.join(run, "shots.json")))["shots"]}
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    diffs = frame_diffs(video)
    indices = range(start, start + limit)
    expected = [round((shots[i]["start_s"] - shots[start]["start_s"]) * fps) for i in indices[1:]]
    edges = [0, *find_cuts(diffs, expected), len(diffs)]
    return {i: (edges[k], edges[k + 1]) for k, i in enumerate(indices)}


def one_person_shots(run, start, limit):
    """``{shot index: character name}`` for the window's shots that name exactly one person."""
    chars = json.load(open(os.path.join(run, "characters.json")))["characters"]
    people = {c["name"] for c in chars if not c.get("keep_name")}
    shot_map: dict = {}
    for c in chars:
        for s in c.get("shots", []):
            shot_map.setdefault(s, []).append(c["name"])
    named = {i: [n for n in shot_map.get(i, []) if n in people] for i in range(start, start + limit)}
    return {i: names[0] for i, names in named.items() if len(names) == 1}


def _sheet_identity(engine, run, name):
    sheet = cv2.imread(os.path.join(run, "characters", f"{name}_sheet.png"))
    faces = engine.detect(sheet) if sheet is not None else []
    if not faces:
        return None
    return engine.embed(sheet, max(faces, key=lambda f: (f.box[2] - f.box[0]) * (f.box[3] - f.box[1])))


def main():
    video, run, start, limit, models, out = sys.argv[1:7]
    start, limit = int(start), int(limit)
    t0 = time.time()
    engine = fid.load_swap_engine(models)
    spans = shot_spans(video, run, start, limit)
    names = one_person_shots(run, start, limit)
    identities = {n: _sheet_identity(engine, run, n) for n in set(names.values())}
    wanted = {spans[i][0]: (i, n) for i, n in names.items() if identities[n] is not None}

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    silent = out + ".video.mp4"
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                            "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", silent],
                           stdin=subprocess.PIPE)
    report: dict = {}
    shot, held = None, []   # a single-person shot is held back until all of it has been seen
    for n, frame in enumerate(_read(video)):
        if n in wanted:
            shot = wanted[n]
        if shot is None:
            enc.stdin.write(frame.tobytes())
            continue
        held.append(frame)
        index, name = shot
        if n + 1 == spans[index][1]:
            done, decided = swap_shot(held, engine, identities[name])
            for f in done:
                enc.stdin.write(f.tobytes())
            report[index] = {"character": name, "first_frame": spans[index][0], **decided}
            shot, held = None, []
    enc.stdin.close()
    if enc.wait() or held:
        raise SystemExit(f"encode failed, or the video ended inside shot {shot}")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", silent, "-i", video, "-map", "0:v", "-map", "1:a?",
                    "-c:v", "copy", "-c:a", "copy", out], check=True)
    os.remove(silent)
    with open(out + ".json", "w") as f:
        json.dump({"seconds": round(time.time() - t0, 1), "spans": spans, "shots": report}, f, indent=1)
    for index, r in report.items():
        print(f"{index} {r['character']:<22} frames {r['frames']:>4}  swapped {r['swapped']:>4}  "
              f"sim med {r['median_sim']} max {r['max_sim']}  {r['reason']}")
    print(f"{sum(r['swapped'] > 0 for r in report.values())} of {len(report)} single-person shots swapped "
          f"in {time.time() - t0:.0f} s -> {out}")


if __name__ == "__main__":
    main()
