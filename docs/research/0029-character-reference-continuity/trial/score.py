# ruff: noqa: E501  -- judge rubrics are prose sent verbatim
"""Score research 0029's trial: arm T (text) vs arm K (keyframe-first), or T vs I (Ingredients).
Free except ~cents of Gemini.

Usage (from the repo root, after the render):
    uv run python docs/research/0029-character-reference-continuity/trial/score.py RUN_DIR MODELS_DIR [--no-judge] [--arms T,I]

--arms picks the two arms to compare (default T,K). Arm I's clips are read from
clips/runpod-ltx25-ingredients-ab/I/, written by trial/make_ingredients_arm.py's manifest.

MODELS_DIR holds OpenCV-zoo yunet.onnx (face detector) and sface.onnx (face embedder,
Apache-2.0); both run on CPU. Writes everything to RUN_DIR/score/ and prints a summary;
results.json is the machine-readable record.

Stages:
  1. frames   first / middle / last frame of every clip (ffmpeg).
  2. faces    detect + embed; a single-character shot takes its largest face; a two-person
              shot's two largest faces are assigned to its characters by similarity to the
              SAME arm's single-shot centroid (no sheet, so T is not disadvantaged).
  3. metrics  per character and arm: mean pairwise cosine across shots (middle frames);
              within-shot first-vs-last cosine; K decay (cosine to the sheet at keyframe,
              first, middle, last); cross-character cosine as the "different people" floor.
  4. grids    per character and arm, the character's face crops from every shot's middle
              frame, tiles shuffled by a seeded RNG; two passes with different shuffles and
              arm letters. The arm key is written outside the grid dir.
  5. judge    Gemini scores each grid blind (same person 1-5, odd one out, matches the
              description 1-5); cross-arm pairwise per character; per shot, both clips as
              inline video under shuffled letters for adherence / artefacts / motion (two
              passes, letters swapped); style of every middle frame (0028's classifier prompt).
  6. motion   mean absolute frame-to-frame difference per clip, and the share of near-frozen steps.
"""
from __future__ import annotations

import importlib.util
import itertools
import json
import os
import random
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.getcwd())
import config  # noqa: E402

ARMS = ("T", "K")   # the two arms compared; `--arms T,I` swaps in arm I (Ingredients)
# Arm I runs under its own strategy, so its clips sit in a different directory.
ARM_STRATEGY = {"I": "runpod-ltx25-ingredients"}
POS = ("first", "mid", "last")
# 0025 judged with gemini-2.5-flash; when AI Studio is rate-limited, run with
# LOSSY_GEMINI_BACKEND=vertex LOSSY_JUDGE_MODEL=gemini-3.1-flash-lite (what Vertex serves here).
JUDGE_MODEL = os.environ.get("LOSSY_JUDGE_MODEL", "gemini-2.5-flash")
PRICE_IN, PRICE_OUT = 0.30, 2.50   # $/1M tokens, for the spend log
TILE = 256

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "classify_style", os.path.join(HERE, "..", "..", "0028-style-drift", "classify_style.py"))
classify_style = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(classify_style)  # type: ignore[union-attr]


# ---------------------------------------------------------------- frames

def clip_path(run: str, arm: str, shot: int) -> str:
    strategy = ARM_STRATEGY.get(arm, "runpod-ltx25")
    return os.path.join(run, "clips", f"{strategy}-ab", arm, f"{shot:04d}.mp4")


def n_frames(path: str) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                          "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path],
                         capture_output=True, text=True).stdout.strip()
    return int(out or 0)


def read_frames(path: str) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    return frames


# ---------------------------------------------------------------- faces

class Faces:
    def __init__(self, models: str):
        self.det = cv2.FaceDetectorYN.create(os.path.join(models, "yunet.onnx"), "", (320, 320), 0.6)
        self.rec = cv2.FaceRecognizerSF.create(os.path.join(models, "sface.onnx"), "")

    def find(self, img: np.ndarray) -> list[dict]:
        """Faces largest first: {'box': (x, y, w, h), 'emb': unit vector}."""
        h, w = img.shape[:2]
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(img)
        out = []
        for f in (faces if faces is not None else []):
            emb = self.rec.feature(self.rec.alignCrop(img, f)).flatten()
            out.append({"box": tuple(int(v) for v in f[:4]), "emb": emb / np.linalg.norm(emb),
                        "score": float(f[-1])})
        return sorted(out, key=lambda d: -d["box"][2] * d["box"][3])


def cos(a, b) -> float:
    return float(np.dot(a, b))


def mean_pairwise(embs: list) -> float | None:
    pairs = list(itertools.combinations(embs, 2))
    return round(float(np.mean([cos(a, b) for a, b in pairs])), 4) if pairs else None


def crop_face(img: np.ndarray, box, scale: float = 2.2) -> Image.Image:
    x, y, w, h = box
    cx, cy, s = x + w / 2, y + h / 2, max(w, h) * scale
    H, W = img.shape[:2]
    x0, y0 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
    x1, y1 = int(min(W, cx + s / 2)), int(min(H, cy + s / 2))
    return Image.fromarray(cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)).resize((TILE, TILE))


def whole(img: np.ndarray) -> Image.Image:
    H, W = img.shape[:2]
    s = min(H, W)
    x0 = (W - s) // 2
    return Image.fromarray(cv2.cvtColor(img[:, x0:x0 + s], cv2.COLOR_BGR2RGB)).resize((TILE, TILE))


# ---------------------------------------------------------------- judge

def _gemini():
    from google.genai import types
    return config.gemini_client(timeout_ms=120_000), types


USAGE = {"in": 0, "out": 0}


def ask(client, types, parts: list, schema: dict) -> dict:
    err = None
    for attempt in range(4):
        try:
            resp = client.models.generate_content(
                model=JUDGE_MODEL, contents=parts,
                config=types.GenerateContentConfig(response_mime_type="application/json",
                                                   response_schema=schema, temperature=0))
            m = resp.usage_metadata
            if m is not None:
                USAGE["in"] += m.prompt_token_count or 0
                USAGE["out"] += (m.candidates_token_count or 0) + (m.thoughts_token_count or 0)
            return json.loads(resp.text or "{}")
        except Exception as e:
            err = e
            print(f"  judge retry {attempt + 1}: {str(e)[:100]}", flush=True)
            time.sleep(10 * (attempt + 1))
    return {"error": str(err)[:300]}


def jpeg(img: Image.Image) -> bytes:
    import io
    b = io.BytesIO()
    img.save(b, "JPEG", quality=88)
    return b.getvalue()


GRID_PROMPT = """These {n} numbered tiles are crops from different video clips that are meant to show ONE recurring character.
Judge ONLY the person's identity (face shape, features, hair, apparent age), not clothing, lighting, pose or image quality.
1. same_person: 1 = clearly different people in most tiles, 3 = plausibly the same person but with noticeable differences, 5 = unmistakably the same person in every tile.
2. odd_one_out: the tile number least like the rest, or 0 if none stands out.
3. matches_description: 1-5, how well the person fits this description: "{desc}"
Tiles with no visible face: ignore them and say so in note."""
GRID_SCHEMA = {"type": "object", "properties": {
    "same_person": {"type": "integer"}, "odd_one_out": {"type": "integer"},
    "matches_description": {"type": "integer"}, "note": {"type": "string"}},
    "required": ["same_person", "odd_one_out", "matches_description", "note"]}

PAIR_PROMPT = """Two sets of numbered face crops, Set A (first image) and Set B (second image). Each set is meant to show ONE recurring character across different video clips.
Which set shows the same person more consistently from tile to tile? Judge identity only (face, features, hair, age), not clothing, lighting or quality.
Answer "A", "B" or "tie", and give each set a 1-5 same-person score."""
PAIR_SCHEMA = {"type": "object", "properties": {
    "more_consistent": {"type": "string", "enum": ["A", "B", "tie"]},
    "score_A": {"type": "integer"}, "score_B": {"type": "integer"}, "note": {"type": "string"}},
    "required": ["more_consistent", "score_A", "score_B", "note"]}

CLIP_PROMPT = """You are scoring {k} short generated video clips (labelled {letters}, in that order) made from the same shot description:
"{desc}"
Score each clip 1-5 on:
- prompt_adherence: does the clip show the described setting, action, shot type, camera move and lighting? (Clothing counts: the description names what people wear.)
- artefacts: 5 = clean; deduct for morphing, melting or extra limbs, garbled faces, flicker, letterbox bars, on-screen text, frozen frames.
- motion: natural, physically plausible and appropriately energetic for the description. 1 = frozen slideshow or chaotic warping.
Return scores for every letter."""
CLIP_SCHEMA = {"type": "object", "properties": {"scores": {"type": "array", "items": {
    "type": "object", "properties": {"letter": {"type": "string"}, "prompt_adherence": {"type": "integer"},
                                     "artefacts": {"type": "integer"}, "motion": {"type": "integer"},
                                     "note": {"type": "string"}},
    "required": ["letter", "prompt_adherence", "artefacts", "motion", "note"]}}}, "required": ["scores"]}


# ---------------------------------------------------------------- main

def main(run: str, models: str, judge: bool) -> None:
    config.load_env()
    plan = json.load(open(os.path.join(run, "trial_plan.json")))
    prompts = {(j["arm"], j["shot"]): j["prompt"] for j in json.load(open(os.path.join(run, "trial_manifest.json")))["jobs"]}
    out = os.path.join(run, "score")
    for d in ("frames", "grids", "pairs"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    faces = Faces(models)
    chars = list(plan["identity"])
    shots = [s for s in plan["shots"] if all(os.path.exists(clip_path(run, a, s["shot"])) for a in ARMS)]
    print(f"{len(shots)}/{len(plan['shots'])} shots have both arms rendered")

    # 1-2. frames, faces, motion
    rec: dict = {}   # (arm, shot) -> {"frames": {pos: img}, "faces": {pos: [..]}, "motion": ..}
    for s in shots:
        for arm in ARMS:
            fr = read_frames(clip_path(run, arm, s["shot"]))
            pick = {"first": fr[0], "mid": fr[len(fr) // 2], "last": fr[-1]}
            g = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (320, 176)).astype(np.float32) for f in fr]
            diffs = [float(np.mean(np.abs(a - b))) for a, b in zip(g, g[1:])]
            for pos, img in pick.items():
                cv2.imwrite(os.path.join(out, "frames", f"{arm}_{s['shot']:04d}_{pos}.jpg"), img)
            rec[(arm, s["shot"])] = {"frames": pick, "faces": {p: faces.find(i) for p, i in pick.items()},
                                     "motion": {"mean_abs_diff": round(float(np.mean(diffs)), 3),
                                                "frozen_share": round(float(np.mean([d < 0.5 for d in diffs])), 3),
                                                "n_frames": len(fr)}}

    sheet_emb = {}
    for c in chars:
        fs = faces.find(cv2.imread(os.path.join(run, "sheets", f"{c}.png")))
        if fs:
            m = np.mean([f["emb"] for f in fs], axis=0)
            sheet_emb[c] = m / np.linalg.norm(m)

    def centroid(arm: str, c: str, pos: str):
        e = [rec[(arm, s["shot"])]["faces"][pos][0]["emb"] for s in shots
             if s["characters"] == [c] and rec[(arm, s["shot"])]["faces"][pos]]
        if not e:
            return None
        m = np.mean(e, axis=0)
        return m / np.linalg.norm(m)

    # face of character c in (arm, shot, pos), or None
    def face_of(arm: str, s: dict, c: str, pos: str):
        fs = rec[(arm, s["shot"])]["faces"][pos]
        if not fs:
            return None
        if len(s["characters"]) == 1:
            return fs[0]
        a, b = s["characters"]
        ca, cb = centroid(arm, a, "mid"), centroid(arm, b, "mid")
        if ca is None or cb is None:
            return None
        if len(fs) == 1:
            want = a if cos(fs[0]["emb"], ca) >= cos(fs[0]["emb"], cb) else b
            return fs[0] if c == want else None
        f0, f1 = fs[0], fs[1]
        straight = cos(f0["emb"], ca) + cos(f1["emb"], cb)
        swapped = cos(f0["emb"], cb) + cos(f1["emb"], ca)
        fa, fb = (f0, f1) if straight >= swapped else (f1, f0)
        return fa if c == a else fb

    # 3. metrics
    metrics: dict = {"identity_cosine": {}, "identity_cosine_singles": {}, "within_shot_first_last": {},
                     "cross_character": {}, "k_decay_vs_sheet": {}, "t_vs_sheet": {}, "face_found": {}}
    for arm in ARMS:
        for c in chars:
            mine = [s for s in shots if c in s["characters"]]
            embs = [f["emb"] for s in mine if (f := face_of(arm, s, c, "mid"))]
            singles = [f["emb"] for s in mine if len(s["characters"]) == 1 and (f := face_of(arm, s, c, "mid"))]
            metrics["identity_cosine"].setdefault(arm, {})[c] = mean_pairwise(embs)
            metrics["identity_cosine_singles"].setdefault(arm, {})[c] = mean_pairwise(singles)
            metrics["face_found"].setdefault(arm, {})[c] = f"{len(embs)}/{len(mine)}"
            fl = [cos(f1["emb"], f2["emb"]) for s in mine
                  if (f1 := face_of(arm, s, c, "first")) and (f2 := face_of(arm, s, c, "last"))]
            metrics["within_shot_first_last"].setdefault(arm, {})[c] = round(float(np.mean(fl)), 4) if fl else None
            if c in sheet_emb:
                curve = {}
                if arm == "K":
                    kf = [faces.find(cv2.imread(os.path.join(run, "k_frames", f"{s['shot']:04d}.png"))) for s in mine]
                    # keyframe faces: largest for singles, best sheet match for pairs
                    kv = [max(cos(f["emb"], sheet_emb[c]) for f in fs[:2]) for fs in kf if fs]
                    curve["keyframe"] = round(float(np.mean(kv)), 4) if kv else None
                for pos in POS:
                    v = [cos(f["emb"], sheet_emb[c]) for s in mine if (f := face_of(arm, s, c, pos))]
                    curve[pos] = round(float(np.mean(v)), 4) if v else None
                mk = "k_decay_vs_sheet" if arm == "K" else f"{arm.lower()}_vs_sheet"
                metrics.setdefault(mk, {})[c] = curve
        cents = {c: centroid(arm, c, "mid") for c in chars}
        cc = [cos(cents[a], cents[b]) for a, b in itertools.combinations(chars, 2)
              if cents[a] is not None and cents[b] is not None]
        metrics["cross_character"][arm] = round(float(np.mean(cc)), 4) if cc else None
    for mk in ("identity_cosine", "identity_cosine_singles", "within_shot_first_last"):
        for arm in ARMS:
            vals = [v for v in metrics[mk][arm].values() if v is not None]
            metrics[mk][arm]["mean"] = round(float(np.mean(vals)), 4) if vals else None
    metrics["source_identity_cosine_singles"] = source_reference(run, faces, shots, chars)
    metrics["motion"] = {arm: {
        "mean_abs_diff": round(float(np.mean([rec[(arm, s["shot"])]["motion"]["mean_abs_diff"] for s in shots])), 3),
        "frozen_share": round(float(np.mean([rec[(arm, s["shot"])]["motion"]["frozen_share"] for s in shots])), 3)}
        for arm in ARMS}
    metrics["motion_per_clip"] = {f"{a}/{s['shot']}": rec[(a, s["shot"])]["motion"] for s in shots for a in ARMS}

    # 4. grids (two passes)
    key: dict = {}
    grids: dict = {}
    for p in (1, 2):
        for c in chars:
            letters = ["X", "Y"]
            random.Random(f"{p}-{c}-letters").shuffle(letters)
            for arm, letter in zip(ARMS, letters):
                tiles = []
                for s in [s for s in shots if c in s["characters"]]:
                    img = rec[(arm, s["shot"])]["frames"]["mid"]
                    f = face_of(arm, s, c, "mid")
                    tiles.append(crop_face(img, f["box"]) if f else whole(img))
                random.Random(f"{p}-{c}-{arm}-tiles").shuffle(tiles)
                cols = min(4, len(tiles))
                rows = (len(tiles) + cols - 1) // cols
                g = Image.new("RGB", (cols * TILE, rows * TILE), "white")
                for i, t in enumerate(tiles):
                    ImageDraw.Draw(t).text((6, 4), str(i + 1), fill=(255, 255, 0))
                    g.paste(t, ((i % cols) * TILE, (i // cols) * TILE))
                name = f"p{p}_{c}_{letter}.jpg"
                g.save(os.path.join(out, "grids", name), quality=88)
                grids[(p, c, arm)] = (g, len(tiles), name)
                key[name] = arm
    json.dump(key, open(os.path.join(out, "grid_key.json"), "w"), indent=1)   # outside grids/

    results = {"shots": [s["shot"] for s in shots], "metrics": metrics}
    if judge:
        client, types = _gemini()
        # 5a. blind identity grids
        gscores = []
        for (p, c, arm), (g, n, name) in grids.items():
            r = ask(client, types, [types.Part.from_bytes(data=jpeg(g), mime_type="image/jpeg"),
                                    GRID_PROMPT.format(n=n, desc=plan["identity"][c])], GRID_SCHEMA)
            gscores.append({"pass": p, "character": c, "arm": arm, "grid": name, **r})
        results["grid_scores"] = gscores
        # 5b. cross-arm pairwise (pass 1: first arm as A; pass 2: second arm as A)
        pw = []
        for p, (a_arm, b_arm) in ((1, ARMS), (2, ARMS[::-1])):
            for c in chars:
                ga, gb = grids[(p, c, a_arm)][0], grids[(p, c, b_arm)][0]
                r = ask(client, types, [types.Part.from_bytes(data=jpeg(ga), mime_type="image/jpeg"),
                                        types.Part.from_bytes(data=jpeg(gb), mime_type="image/jpeg"),
                                        PAIR_PROMPT], PAIR_SCHEMA)
                win = {"A": a_arm, "B": b_arm}.get(r.get("more_consistent", ""), "tie")
                pw.append({"pass": p, "character": c, "A": a_arm, "B": b_arm, "winner": win, **r})
        results["pairwise"] = pw
        # 5c. per-shot clip regressions (two passes, letters swapped)
        cs = []
        for p in (1, 2):
            for s in shots:
                order = list(ARMS) if (p == 1) == (s["shot"] % 2 == 0) else list(ARMS)[::-1]
                letters = ["A", "B"]
                parts = []
                for arm in order:
                    parts.append(types.Part.from_bytes(data=open(clip_path(run, arm, s["shot"]), "rb").read(),
                                                       mime_type="video/mp4"))
                desc = prompts[("T", s["shot"])]
                parts.append(CLIP_PROMPT.format(k=2, letters=", ".join(letters), desc=desc))
                r = ask(client, types, parts, CLIP_SCHEMA)
                by = {x.get("letter"): x for x in r.get("scores", [])}
                for letter, arm in zip(letters, order):
                    if letter in by:
                        cs.append({"pass": p, "shot": s["shot"], "arm": arm, **by[letter]})
        results["clip_scores"] = cs
        # 5d. style
        st = []
        for s in shots:
            for arm in ARMS:
                img = open(os.path.join(out, "frames", f"{arm}_{s['shot']:04d}_mid.jpg"), "rb").read()
                r = ask(client, types, [types.Part.from_bytes(data=img, mime_type="image/jpeg"),
                                        classify_style.PROMPT], classify_style.SCHEMA)
                st.append({"shot": s["shot"], "arm": arm, **r})
        results["style"] = st
        results["judge"] = {"model": JUDGE_MODEL, "tokens": USAGE,
                            "usd": round(USAGE["in"] / 1e6 * PRICE_IN + USAGE["out"] / 1e6 * PRICE_OUT, 4)}
        results["summary"] = summarise(results)
    json.dump(results, open(os.path.join(out, "results.json"), "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in metrics.items() if k != "motion_per_clip"}, indent=1))
    if judge:
        print(json.dumps(results["summary"], indent=1))
        print(json.dumps(results["judge"]))


def source_reference(run: str, faces: Faces, shots: list, chars: list) -> dict:
    """The same metric on the real episode's frames (local only, never sent to a judge).

    Real footage of one actor is the ceiling this metric can show for "same person";
    single-character shots only, largest face, from encode's source keyframes.
    """
    out = {}
    for c in chars:
        embs = []
        for s in shots:
            path = os.path.join(run, "source_kf", f"{s['shot']:04d}.jpg")
            if s["characters"] == [c] and os.path.exists(path):
                fs = faces.find(cv2.imread(path))
                if fs:
                    embs.append(fs[0]["emb"])
        out[c] = mean_pairwise(embs)
    vals = [v for v in out.values() if v is not None]
    out["mean"] = round(float(np.mean(vals)), 4) if vals else None
    return out


def summarise(r: dict) -> dict:
    def mean(xs):
        xs = [x for x in xs if isinstance(x, (int, float))]
        return round(float(np.mean(xs)), 2) if xs else None
    out: dict = {}
    for arm in ARMS:
        g = [x for x in r["grid_scores"] if x["arm"] == arm]
        c = [x for x in r["clip_scores"] if x["arm"] == arm]
        st = [x for x in r["style"] if x["arm"] == arm]
        out[arm] = {
            "same_person": mean([x.get("same_person") for x in g]),
            "same_person_by_pass": {p: mean([x.get("same_person") for x in g if x["pass"] == p]) for p in (1, 2)},
            "same_person_by_char": {ch: mean([x.get("same_person") for x in g if x["character"] == ch])
                                    for ch in sorted({x["character"] for x in g})},
            "matches_description": mean([x.get("matches_description") for x in g]),
            "prompt_adherence": mean([x.get("prompt_adherence") for x in c]),
            "artefacts": mean([x.get("artefacts") for x in c]),
            "motion": mean([x.get("motion") for x in c]),
            "photoreal_rate": round(sum(x.get("style") == "photoreal" for x in st) / len(st), 3) if st else None,
        }
    out["pairwise_wins"] = {w: sum(x["winner"] == w for x in r["pairwise"]) for w in (*ARMS, "tie")}
    return out


if __name__ == "__main__":
    if "--arms" in sys.argv:   # e.g. --arms T,I ; exactly two, since pairwise and clip judging compare two
        i = sys.argv.index("--arms")
        ARMS = tuple(sys.argv[i + 1].split(","))
        del sys.argv[i:i + 2]
        assert len(ARMS) == 2, f"--arms takes exactly two arms, got {ARMS}"
    main(sys.argv[1], sys.argv[2], "--no-judge" not in sys.argv)
