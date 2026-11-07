"""Build the fixed 20-shot hard benchmark for the video-model pilot (research 0025).

Selects shots from an existing encode by category, composes each prompt through
the REAL production path (RunPodLtx2Strategy.format_prompt: identity mixin,
2-block cap, name stripping), and writes a self-contained benchmark dir:

    <out>/benchmark.json      shots, categories, prompts, target frames
    <out>/keyframes/NNNN.jpg  middle keyframe of each source shot (for scoring)

Every model in the pilot gets byte-identical prompts, so differences come from
the model, not the prompt. Deterministic: same encode in, same 20 shots out.

    uv run python tools/model_pilot/build_benchmark.py \
        /Users/will/projects/lossy/output/star_wars_iv_v2 output/pilot0025
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from decode import build_character_shot_map  # noqa: E402
from strategies_video import RunPodLtx2Strategy  # noqa: E402

# Single-part shots only (LTX caps a part at 121 frames = 4.84s), and long
# enough that motion is judgeable. Split-shot chaining is out of scope here:
# it is a pipeline property already measured in 0023/0024.
MIN_S, MAX_S = 2.0, 4.8

NON_HUMANOIDS = {"c_3po", "r2_d2", "chewbacca", "chief_jawa", "darth_vader"}


def _f(s: dict, key: str) -> str:
    """A description field as text (the encode stores some fields as lists)."""
    v = s.get("description", {}).get(key, "")
    return " ".join(map(str, v)) if isinstance(v, list) else str(v or "")


def _text(s: dict) -> str:
    return " ".join(_f(s, k) for k in ("subjects", "action", "setting", "lighting"))


def _cats(shot_map: dict) -> dict:
    """Category -> (predicate, quota). Order matters: a shot lands in the first
    category it matches that still has room, so categories don't double-count."""

    def chars(s):
        return shot_map.get(s["index"], [])

    return {
        # 0023 defect 5: >=3 stacked identity blocks blended figures together.
        "three_plus_characters": (lambda s: len(chars(s)) >= 3, 3),
        # 0023 defect 6: non-humanoids rendered as people in costume.
        "droids_nonhumanoids": (
            lambda s: bool(NON_HUMANOIDS & set(chars(s)))
            and re.search(r"\b(droid|robot|astromech|fur|furry|wookiee|creature|jawa)\b",
                          _f(s, "subjects"), re.I), 3),
        # Wide vistas: also the letterbox trigger found in 0024.
        "wide_long_shots": (
            lambda s: _f(s, "shot_type") in ("extreme wide", "wide"), 3),
        "dark_interiors": (
            lambda s: re.search(r"\b(dimly lit|dark|low-key|very little light|darkness)\b",
                                _f(s, "lighting"), re.I)
            and re.search(r"\b(interior|corridor|room|inside|chamber|cockpit|hold)\b",
                          _f(s, "setting"), re.I), 3),
        "fast_action": (
            lambda s: re.search(r"\b(explo\w+|firing|fires|blaster fire|lightsaber|duel|"
                                r"dives?|races?|chase|speeding|swoops?|crash\w*)\b",
                                _f(s, "action"), re.I), 3),
        "dialogue_closeups": (
            lambda s: _f(s, "shot_type") in ("close-up", "medium close-up")
            and s.get("dialogue") and chars(s), 3),
    }


# 0023's named failure classes, pinned so the benchmark always contains them.
FAILURE_QUERIES = [
    # Vader rendered in a grey sweatshirt un-chained (0023 'What worked' note).
    ("vader_costume", lambda s, cm: "darth_vader" in cm.get(s["index"], [])
        and _f(s, "shot_type") in ("medium", "medium wide", "medium close-up")),
    # Leia hologram: described from world knowledge as Obi-Wan (0021).
    ("leia_hologram", lambda s, cm: re.search(r"hologra", _f(s, "subjects"), re.I)
        and re.search(r"\b(woman|female|princess)\b", _f(s, "subjects"), re.I)),
]


def _spread(cands: list[dict], n: int) -> list[dict]:
    """Pick n evenly spaced candidates across the film (deterministic)."""
    if len(cands) <= n:
        return cands
    step = len(cands) / n
    return [cands[int(i * step + step / 2)] for i in range(n)]


def build(encode_dir: str, out_dir: str) -> dict:
    with open(os.path.join(encode_dir, "shots.json")) as f:
        shots = json.load(f)["shots"]
    with open(os.path.join(encode_dir, "shot_index.json")) as f:
        src = json.load(f)["source"]
    source = os.path.join(encode_dir, "..", "..", src["path"])
    with open(os.path.join(encode_dir, "characters.json")) as f:
        characters = json.load(f)
    shot_map = build_character_shot_map(characters)
    strat = RunPodLtx2Strategy.__new__(RunPodLtx2Strategy)  # prompt path only, no pod
    strat._init_character_identity(shot_map, characters)

    eligible = [s for s in shots if MIN_S <= s["duration_s"] <= MAX_S and s.get("description")]
    taken: set[int] = set()
    picks: list[tuple[str, dict]] = []

    for name, query in FAILURE_QUERIES:
        cands = [s for s in eligible if s["index"] not in taken and query(s, shot_map)]
        if cands:
            s = _spread(cands, 1)[0]
            picks.append((f"known_failure:{name}", s))
            taken.add(s["index"])

    for cat, (pred, quota) in _cats(shot_map).items():
        cands = [s for s in eligible if s["index"] not in taken and pred(s)]
        for s in _spread(cands, quota):
            picks.append((cat, s))
            taken.add(s["index"])

    os.makedirs(os.path.join(out_dir, "keyframes"), exist_ok=True)
    out = []
    for cat, s in sorted(picks, key=lambda p: p[1]["index"]):
        idx = s["index"]
        # Reference frame straight from the source film at the shot midpoint.
        # The encode's keyframes/ dir is NOT used: its NNNN numbering is stale
        # (predates the Aug re-encode) and shows the wrong shot from ~#405 on.
        kf = os.path.join(out_dir, "keyframes", f"{idx:04d}.jpg")
        mid = (s["start_s"] + s["end_s"]) / 2
        r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{mid:.3f}", "-i", source,
                            "-frames:v", "1", "-vf", "scale=640:-2", kf])
        kf_src = kf if r.returncode == 0 and os.path.exists(kf) else None
        target_s = min(s["duration_s"], MAX_S)
        out.append({
            "index": idx,
            "category": cat,
            "start_s": s["start_s"],
            "duration_s": s["duration_s"],
            "target_s": round(target_s, 3),
            "characters": shot_map.get(idx, []),
            "prompt": strat.format_prompt(s),
            "description": s["description"],
            "dialogue": s.get("dialogue", []),
            "keyframe": f"keyframes/{idx:04d}.jpg" if kf_src else None,
        })

    bench = {"source_encode": os.path.abspath(encode_dir), "n": len(out), "shots": out}
    with open(os.path.join(out_dir, "benchmark.json"), "w") as f:
        json.dump(bench, f, indent=1)
    return bench


def main():
    ap = argparse.ArgumentParser(description="Build the 20-shot video-model pilot benchmark")
    ap.add_argument("encode_dir")
    ap.add_argument("out_dir")
    a = ap.parse_args()
    b = build(a.encode_dir, a.out_dir)
    for s in b["shots"]:
        print(f"{s['index']:5d} {s['duration_s']:5.2f}s {s['category']:32s} "
              f"{','.join(s['characters'])[:40]:40s} {s['description']['subjects'][:70]}")
    print(f"{b['n']} shots -> {a.out_dir}/benchmark.json")


if __name__ == "__main__":
    main()
