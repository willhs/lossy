"""Words-per-second vs fidelity chart: variant writer, bit-rate meter and decode plan.

Protocol: docs/research/0032-rate-distortion-protocol/research.md (read it first; it
is pre-registered). This script only prepares: nothing here renders a clip or rents a
pod. The one paid-ish call, the Gemini rewrite, is behind `--go` and has not been run.

    rewrite SRC RUN --go   Gemini rewrites each scene shot to 8/32/64 words (cents)
    build   SRC RUN        variants/<budget>/shots.json, bitrate.json, manifests/<step>.json
    plan    RUN            the decode commands (capped_run), AV1 commands, projected cost

SRC is the SW4 encode dir (shots.json, characters.json); RUN is a fresh output dir for
this experiment (e.g. output/rd_curve). `build` and `plan` are offline and free.
"""

import argparse
import json
import lzma
import math
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import manifest  # noqa: E402

# ---------------------------------------------------------------------------
# Pre-registered constants (change them only by amending the protocol note)
# ---------------------------------------------------------------------------

SCENE_FIRST, SCENE_COUNT = 430, 40          # SW4 shots 430-469: desert, Luke/Obi-Wan/droids
CORE_STRIDE = 4                             # the 10-shot core: every 4th scene shot
STYLE_PREFIX = ("Live-action 1977 science-fiction film footage of real actors and practical "
                "sets, photographed on 35mm film.")
GENERIC_TEXT = "a shot from a 1977 science-fiction film"   # the 0-word description
# The fields format_prompt turns into video-prompt words. `sound`, `music` and
# `dialogue` feed the audio channels, not the video decoder, so they are out of
# the axis (and untouched in every variant).
VIDEO_FIELDS = ("shot_type", "camera_movement", "subjects", "action", "lighting",
                "color_palette", "mood", "setting")
STRATEGY = "runpod-ltx25"
AV1_KBPS = (10, 30)
FPS = 25
MAX_CLIP_FRAMES = 121                       # ltx25_frames hi: a longer shot renders its first 4.84 s

# Cost model: research 0026 fit (46 s fixed + 16 s per video-second, A6000 SECURE) and
# 0026's measured all-in ~$0.024 a shot. A projection quotes the larger of the two.
FIXED_S, PER_VIDEO_S, RATE_PER_H, BOOT_MIN = 46, 16, 0.59, 20
MEASURED_PER_CLIP = 0.024


@dataclass(frozen=True)
class Budget:
    label: str
    words: int | None            # None = the description as encoded

    @property
    def generic(self) -> bool:
        return self.words == 0


# The spec's 128-word point is 64 here: this scene's descriptions average ~124 words, so 128
# would be "full" for most shots (protocol note, "Budgets").
BUDGETS = (Budget("w0", 0), Budget("w8", 8), Budget("w32", 32),
           Budget("w64", 64), Budget("full", None))
BY_LABEL = {b.label: b for b in BUDGETS}
LLM_BUDGETS = tuple(b for b in BUDGETS if b.words)   # 8, 32, 64 need a rewrite


@dataclass(frozen=True)
class Arm:
    budget: Budget
    seed_k: int                  # 1, 2, 3; seed 1 is the shot index itself
    identity: bool = False       # True: characters.json identity blocks ride along

    @property
    def name(self) -> str:
        return f"{self.budget.label}_s{self.seed_k}" + ("_id" if self.identity else "")

    def seed(self, shot_index: int) -> int:
        return shot_index + 10000 * (self.seed_k - 1)


@dataclass(frozen=True)
class Step:
    name: str
    cells: tuple                 # ((Arm, (shot indices,)), ...)

    def n_clips(self) -> int:
        return sum(len(shots) for _, shots in self.cells)


def core_indices(scene: list[dict]) -> tuple:
    return tuple(s["index"] for s in scene[::CORE_STRIDE])


def make_steps(scene: list[dict]) -> list[Step]:
    """pilot: 10 core shots x (8, 32, full) x 1 seed. chart: all 40 shots x 5 budgets x
    seed 1, plus the core x 5 budgets x seeds 2 and 3 (the seed-noise estimate).
    identity: the core x (0, 32, full) with the identity blocks switched on."""
    every = tuple(s["index"] for s in scene)
    core = core_indices(scene)
    pilot = Step("pilot", tuple((Arm(BY_LABEL[b], 1), core) for b in ("w8", "w32", "full")))
    chart = Step("chart", tuple([(Arm(b, 1), every) for b in BUDGETS]
                                + [(Arm(b, k), core) for k in (2, 3) for b in BUDGETS]))
    identity = Step("identity", tuple((Arm(BY_LABEL[b], 1, identity=True), core)
                                      for b in ("w0", "w32", "full")))
    return [pilot, chart, identity]


# ---------------------------------------------------------------------------
# Scene and words
# ---------------------------------------------------------------------------

def select_scene(shots: list[dict], first: int = SCENE_FIRST, count: int = SCENE_COUNT) -> list[dict]:
    """The `count` consecutive shots from index `first`, in film order. Fails closed on a gap."""
    by_index = {s["index"]: s for s in shots}
    wanted = range(first, first + count)
    missing = [i for i in wanted if i not in by_index]
    if missing:
        raise ValueError(f"scene shots missing from shots.json: {missing[:5]}")
    return [by_index[i] for i in wanted]


def field_text(value) -> str:
    """A description field as text. Gemini sometimes emits lists or dicts."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return " ".join(field_text(v) for v in value)
    if isinstance(value, dict):
        return " ".join(field_text(v) for v in value.values())
    return str(value)


def count_words(desc: dict) -> int:
    return sum(len(field_text(desc.get(f)).split()) for f in VIDEO_FIELDS)


def generic_description(desc: dict) -> dict:
    """The 0-word point: every shot carries the same constant, nothing per shot. `sound`
    and `music` stay as encoded (audio channels, outside this axis)."""
    out = {k: v for k, v in desc.items() if k not in VIDEO_FIELDS}
    out["subjects"] = GENERIC_TEXT
    return out


def needs_rewrite(shot: dict, budget: Budget) -> bool:
    """Only a description longer than the budget is rewritten; a shorter one is kept as encoded."""
    return bool(budget.words) and count_words(shot["description"]) > budget.words


class RewriteError(ValueError):
    pass


def check_rewrite(original: dict, rewritten, budget: int) -> dict:
    """Validate Gemini's rewrite and return it as a description. Raises RewriteError."""
    if not isinstance(rewritten, dict):
        raise RewriteError("not a JSON object")
    extra = set(rewritten) - set(VIDEO_FIELDS)
    if extra:
        raise RewriteError(f"unknown fields {sorted(extra)}")
    clean = {f: field_text(rewritten.get(f)).strip() for f in VIDEO_FIELDS if rewritten.get(f)}
    words = count_words(clean)
    if words > budget:
        raise RewriteError(f"{words} words, budget {budget}")
    if not words:
        raise RewriteError("empty rewrite")
    return {**{k: v for k, v in original.items() if k not in VIDEO_FIELDS}, **clean}


def rewrite_shot(shot: dict, budget: Budget, rewriter, tries: int = 3) -> dict:
    """Call `rewriter(description, words, feedback) -> dict` until a rewrite fits the budget.
    No silent truncation: a rewrite that never fits raises, and the shot stays unwritten."""
    assert budget.words, "the 0-word and full budgets are not rewritten"
    feedback = None
    for _ in range(tries):
        try:
            return check_rewrite(shot["description"], rewriter(shot["description"], budget.words, feedback),
                                 budget.words)
        except RewriteError as e:
            feedback = str(e)
    raise RewriteError(f"shot {shot['index']} at {budget.label}: {feedback} after {tries} tries")


def build_variant(scene: list[dict], budget: Budget, rewrites: dict) -> list[dict]:
    """The scene's shots at one budget. `rewrites` maps shot index -> rewritten description
    (str or int keys: it comes from JSON). Raises KeyError naming a shot still unwritten."""
    out = []
    for shot in scene:
        if budget.generic:
            desc = generic_description(shot["description"])
        elif needs_rewrite(shot, budget):
            idx = shot["index"]
            desc = rewrites[idx] if idx in rewrites else rewrites[str(idx)]
        else:
            desc = shot["description"]
        out.append({**shot, "description": desc})
    return out


def variant_path(run_dir: str, label: str) -> str:
    return os.path.join(run_dir, "variants", label, "shots.json")


def write_variant(run_dir: str, budget: Budget, shots: list[dict]) -> str:
    path = variant_path(run_dir, budget.label)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    manifest.write_json_atomic(path, {
        "format": manifest.PROMPTS_FORMAT_VERSION, "shots": shots, "dialog": [],
        "rd": {"budget": budget.label, "words": budget.words},
    })
    return path


# ---------------------------------------------------------------------------
# Bits per second after xz
# ---------------------------------------------------------------------------

def scene_duration(shots: list[dict]) -> float:
    return shots[-1]["end_s"] - shots[0]["start_s"]


def stream_bytes(shots: list[dict]) -> bytes:
    """What the decoder is sent for the video channel: per shot, its timing and the video
    description fields, as minified JSON with the same keys shots.json uses. The 0-word
    variant is therefore pure timing plus one repeated constant."""
    records = [{"index": s["index"], "start_s": s["start_s"], "end_s": s["end_s"],
                "description": {f: s["description"][f] for f in VIDEO_FIELDS if s["description"].get(f)}}
               for s in shots]
    return json.dumps(records, separators=(",", ":"), sort_keys=True, ensure_ascii=False).encode()


def xz_size(data: bytes) -> int:
    """`xz -9e --check=none -T1` size, in bytes (one block: the CLI's default threading
    splits it and adds headers). The check is dropped: 8 bytes of CRC64 is a visible
    share of a 0-word stream."""
    return len(lzma.compress(data, format=lzma.FORMAT_XZ, check=lzma.CHECK_NONE,
                             preset=9 | lzma.PRESET_EXTREME))


def measure_variant(shots: list[dict]) -> dict:
    raw = stream_bytes(shots)
    xz = xz_size(raw)
    dur = scene_duration(shots)
    words = [count_words(s["description"]) for s in shots]
    return {"shots": len(shots), "duration_s": round(dur, 3), "raw_bytes": len(raw),
            "xz_bytes": xz, "bits_per_s": round(8 * xz / dur, 2),
            "words_total": sum(words), "words_mean": round(sum(words) / len(words), 1),
            "words_per_s": round(sum(words) / dur, 3)}


def av1_command(source: str, shots: list[dict], kbps: int, out_dir: str) -> list[list[str]]:
    """Two-pass libaom encode of the scene at a target rate, video only, 360p. The x-axis
    uses the measured size of the file, not the target."""
    start, dur = shots[0]["start_s"], scene_duration(shots)
    out = os.path.join(out_dir, f"av1_{kbps}k.mkv")
    common = ["ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", source, "-t", f"{dur:.3f}", "-an",
              "-vf", "scale=-2:360", "-c:v", "libaom-av1", "-b:v", f"{kbps}k", "-cpu-used", "4",
              "-passlogfile", os.path.join(out_dir, f"av1_{kbps}k")]
    return [common + ["-pass", "1", "-f", "null", os.devnull], common + ["-pass", "2", out]]


def measure_av1(run_dir: str, duration_s: float) -> dict:
    out = {}
    for kbps in AV1_KBPS:
        path = os.path.join(run_dir, "av1", f"av1_{kbps}k.mkv")
        if os.path.exists(path):
            out[f"av1_{kbps}k"] = {"target_kbps": kbps, "bytes": os.path.getsize(path),
                                   "bits_per_s": round(8 * os.path.getsize(path) / duration_s, 1)}
    return out


# ---------------------------------------------------------------------------
# Decode jobs and projected cost
# ---------------------------------------------------------------------------

def frames_for(duration_s: float) -> int:
    """ltx25_frames without importing the pod stack: frames % 8 == 1, clamped to [25, 121]."""
    n = round((round(duration_s * FPS) - 1) / 8) * 8 + 1
    return max(25, min(MAX_CLIP_FRAMES, n))


def build_jobs(step: Step, variants: dict, prompt_for) -> list[dict]:
    """The step's AB-manifest jobs. `variants`: label -> {index: shot}. `prompt_for(entry,
    identity) -> str` is the real strategy's prompt builder (injected so tests need no pod
    code). `span_s` is the source span the clip covers; the fidelity tool scores its midpoint."""
    jobs = []
    for arm, shot_indices in step.cells:
        for idx in shot_indices:
            shot = variants[arm.budget.label][idx]
            frames = frames_for(shot["duration_s"])
            jobs.append({
                "shot": idx, "arm": arm.name, "seed": arm.seed(idx), "frames": frames,
                "prompt": prompt_for(shot, arm.identity),
                "span_s": [shot["start_s"], round(shot["start_s"] + frames / FPS, 3)],
                "words": count_words(shot["description"]),
            })
    return jobs


def project_cost(jobs: list[dict]) -> dict:
    gpu_s = sum(FIXED_S + PER_VIDEO_S * j["frames"] / FPS for j in jobs)
    model = (gpu_s / 3600 + BOOT_MIN / 60) * RATE_PER_H
    flat = MEASURED_PER_CLIP * len(jobs)
    quoted = max(model, flat)
    # capped_run stops a margin below the cap; leave room for it and for a bad day.
    cap = math.ceil((1.5 * quoted + 0.5) * 2) / 2
    return {"clips": len(jobs), "gpu_minutes": round(gpu_s / 60, 1), "model_usd": round(model, 2),
            "flat_usd": round(flat, 2), "projected_usd": round(quoted, 2), "cap_usd": cap}


def decode_command(run_dir: str, manifest_path: str, cap: float, dry_run: bool = False) -> str:
    decode = f"uv run python decode.py {run_dir} --strategy {STRATEGY} --ab-manifest {manifest_path}"
    if dry_run:
        return decode + " --dry-run"
    return (f"uv run python tools/capped_run.py --cap {cap:g} --margin 0.5 --no-project "
            f"--output-dir {run_dir} -- {decode}")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def load_scene(src: str) -> list[dict]:
    shots, _ = manifest.load_shots(src)
    return select_scene(shots)


def rewrites_path(run_dir: str, label: str) -> str:
    return os.path.join(run_dir, "variants", label, "rewrites.json")


def load_rewrites(run_dir: str, label: str) -> dict:
    path = rewrites_path(run_dir, label)
    if not os.path.exists(path):
        return {"model": None, "descriptions": {}}
    with open(path) as f:
        return json.load(f)


REWRITE_SYSTEM = """You shorten a film-shot description to a strict word budget for a video generator.

You get a JSON object describing one shot (fields: shot_type, camera_movement, subjects, action, lighting, color_palette, mood, setting). Return a JSON object using ONLY those field names, with at most {budget} words in total across all fields (every word of every value counts). Leave out fields you drop.

Rules:
- Keep what a viewer would need to recognise the shot first: who or what is in frame, the composition (shot type), the main action, the place. Drop mood, palette and lighting before those.
- Add nothing the original does not say. Do not name anyone the original does not name.
- Output ONLY the JSON object, no markdown."""


def gemini_rewriter():
    """Build the real rewriter: (description, words, feedback) -> dict. Imports lazily: encode
    pulls in the whole encode stack and `config.ENCODE_MODEL` / backend come from the environment."""
    import config
    from google.genai import types

    import encode

    config.load_env()
    client = encode._gemini_client(os.environ.get("GEMINI_API_KEY", ""))

    def rewrite(desc: dict, words: int, feedback: str | None) -> dict:
        body = {f: desc[f] for f in VIDEO_FIELDS if desc.get(f)}
        text = json.dumps(body, ensure_ascii=False)
        if feedback:
            text += f"\n\nYour last attempt was rejected: {feedback}. Shorten it."
        response = client.models.generate_content(
            model=config.ENCODE_MODEL,
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=text)])],
            config=types.GenerateContentConfig(
                system_instruction=REWRITE_SYSTEM.format(budget=words),
                temperature=0.0, response_mime_type="application/json"),
        )
        try:
            return json.loads((response.text or "").strip())
        except json.JSONDecodeError as e:
            raise RewriteError(f"response not valid JSON: {e}") from e

    return rewrite, config.ENCODE_MODEL


def cmd_rewrite(args) -> None:
    scene = load_scene(args.src)
    todo = {b.label: [s for s in scene if needs_rewrite(s, b)] for b in LLM_BUDGETS}
    pending = {label: [s for s in shots if str(s["index"]) not in load_rewrites(args.run, label)["descriptions"]]
               for label, shots in todo.items()}
    n = sum(len(v) for v in pending.values())
    print(f"{n} Gemini calls pending ({', '.join(f'{k}: {len(v)}' for k, v in pending.items())}); "
          f"{sum(len(v) for v in todo.values()) - n} already written.")
    if not args.go:
        print("Dry run. Re-run with --go to make them (cents; needs a Gemini key or the vertex backend).")
        return
    rewriter, model = gemini_rewriter()
    failed = []
    for label, shots in pending.items():
        data = load_rewrites(args.run, label)
        data["model"] = model
        for shot in shots:
            try:
                data["descriptions"][str(shot["index"])] = rewrite_shot(shot, BY_LABEL[label], rewriter)
            except RewriteError as e:
                failed.append(str(e))
                print(f"  FAILED {e}")
                continue
            os.makedirs(os.path.dirname(rewrites_path(args.run, label)), exist_ok=True)
            manifest.write_json_atomic(rewrites_path(args.run, label), data)   # resume point after every shot
    if failed:
        sys.exit(f"{len(failed)} shots did not fit their budget; re-run to retry them.")


def make_prompt_builder(src: str):
    """The real LTX-2.5 prompt builder over SRC's characters.json. Identity off keeps the
    characters so personal names are still stripped from the text (as in a real decode)
    but gives no shot any identity block."""
    from decode import build_character_shot_map
    from strategies_video import RunPodLtx25Strategy

    with open(manifest.characters_path(src)) as f:
        chars = json.load(f)
    shot_map = build_character_shot_map(chars)

    def strategy(identity: bool):
        s = object.__new__(RunPodLtx25Strategy)   # skip __init__: no RunPod session
        s._init_character_identity(shot_map if identity else {}, chars, STYLE_PREFIX)
        return s

    plain, with_identity = strategy(False), strategy(True)
    return lambda entry, identity: (with_identity if identity else plain).format_prompt(entry)


def cmd_build(args) -> None:
    scene = load_scene(args.src)
    variants, report = {}, {}
    for b in BUDGETS:
        rewrites = load_rewrites(args.run, b.label)["descriptions"] if b.words else {}
        try:
            shots = build_variant(scene, b, rewrites)
        except KeyError as e:
            sys.exit(f"{b.label}: shot {e} has no rewrite yet. Run `rewrite --go` first.")
        write_variant(args.run, b, shots)
        variants[b.label] = {s["index"]: s for s in shots}
        report[b.label] = measure_variant(shots)
    report.update(measure_av1(args.run, scene_duration(scene)))
    manifest.write_json_atomic(os.path.join(args.run, "bitrate.json"), report)
    print(f"{'variant':<9}{'words/shot':>11}{'words/s':>9}{'xz bytes':>10}{'bit/s':>9}")
    for label, m in report.items():
        if "words_mean" in m:
            print(f"{label:<9}{m['words_mean']:>11}{m['words_per_s']:>9}{m['xz_bytes']:>10}{m['bits_per_s']:>9}")
        else:
            print(f"{label:<9}{'-':>11}{'-':>9}{m['bytes']:>10}{m['bits_per_s']:>9}")

    prompt_for = make_prompt_builder(args.src)
    os.makedirs(os.path.join(args.run, "manifests"), exist_ok=True)
    for step in make_steps(scene):
        jobs = build_jobs(step, variants, prompt_for)
        cost = project_cost(jobs)
        manifest.write_json_atomic(os.path.join(args.run, "manifests", f"{step.name}.json"),
                                   {"step": step.name, "style_prefix": STYLE_PREFIX, "cost": cost, "jobs": jobs})
    print(f"Wrote variants, bitrate.json and manifests under {args.run}. Next: `plan {args.run}`.")


def cmd_plan(args) -> None:
    run = args.run
    scene_shots = None
    if args.src:
        scene_shots = load_scene(args.src)
    print("Decode steps (each resumes: finished arm/shot pairs are skipped, so the chart reuses the pilot's clips):")
    for step in ("pilot", "chart", "identity"):
        path = os.path.join(run, "manifests", f"{step}.json")
        with open(path) as f:
            cost = json.load(f)["cost"]
        print(f"\n[{step}] {cost['clips']} clips, ~{cost['gpu_minutes']} GPU-min, projected "
              f"${cost['projected_usd']} (model ${cost['model_usd']}, measured-rate ${cost['flat_usd']}), cap ${cost['cap_usd']:g}")
        print(f"  dry run: {decode_command(run, path, cost['cap_usd'], dry_run=True)}")
        print(f"  run:     {decode_command(run, path, cost['cap_usd'])}")
    if scene_shots:
        print(f"\nAV1 baselines (local ffmpeg, free; run before `build` so bitrate.json lists them):")
        os.makedirs(os.path.join(run, "av1"), exist_ok=True)
        for kbps in AV1_KBPS:
            for cmd in av1_command(args.source, scene_shots, kbps, os.path.join(run, "av1")):
                print("  " + " ".join(cmd))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("rewrite", "build"):
        p = sub.add_parser(name)
        p.add_argument("src", help="SW4 encode dir (shots.json, characters.json)")
        p.add_argument("run", help="experiment output dir, e.g. output/rd_curve")
    sub.choices["rewrite"].add_argument("--go", action="store_true", help="actually call Gemini")
    p = sub.add_parser("plan")
    p.add_argument("run")
    p.add_argument("--src", help="SW4 encode dir; with it, also prints the AV1 commands")
    p.add_argument("--source", default="media/star_wars_iv.mp4", help="source video for the AV1 baselines")
    args = ap.parse_args(argv)
    {"rewrite": cmd_rewrite, "build": cmd_build, "plan": cmd_plan}[args.cmd](args)


if __name__ == "__main__":
    main()
