"""Build research 0029's trial: 16 Seinfeld shots, arms T (text) and K (keyframe-first).

Usage (from the repo root):
    uv run python docs/research/0029-character-reference-continuity/trial/make_trial_manifest.py RUN_DIR

RUN_DIR holds a copy of the episode's shots.json and characters.json (output/, gitignored).
Writes RUN_DIR/trial_manifest.json (the --ab-manifest jobs; K jobs point at
k_frames/NNNN.png, made later by make_refs.py) and RUN_DIR/trial_plan.json (the
shots, the identity blocks and each K keyframe's editor prompt). Spends nothing.

Both arms get the same text: the photoreal style prefix (0028) and one name-free,
photographic identity block per character, prepended by the real
CharacterIdentityMixin. The only difference is K's first frame.
"""
import json
import os
import re
import sys

sys.path.insert(0, os.getcwd())
from decode import build_character_shot_map  # noqa: E402
from strategies_video import RunPodLtx25Strategy, ltx25_frames  # noqa: E402

STYLE_PREFIX = ("Live-action 1990s American TV sitcom footage of real actors, "
                "photographed on 35mm film, photorealistic skin and lighting.")

# Photographic identity blocks in the style of encode.IDENTITY_DESCRIPTION_SPEC
# (e54fa16). Jerry, George and Elaine are converted from the episode's own
# characters.json text only (wardrobe, role and demeanour dropped, nothing added
# from knowing the actors: 0021). Kramer's is 0028's KRAMER_PHOTO verbatim.
IDENTITY = {
    "jerry_seinfeld": ("A real man in his mid 30s with short dark wavy hair and a lean build. "
                       "Clean-shaven, natural skin texture, ordinary features."),
    "george_costanza": ("A real middle-aged man with thinning hair and glasses. "
                        "Clean-shaven, natural skin texture, ordinary features."),
    "cosmo_kramer": ("A real man in his late 30s, tall and slim, with thick dark curly hair "
                     "brushed upward and back from a high forehead. Long face, prominent straight "
                     "nose, high cheekbones, deep-set dark eyes under a heavy brow, natural skin "
                     "texture and light stubble."),
    "elaine_benes": ("A real woman in her early 30s with dark, voluminous curly hair. "
                     "Natural skin texture, ordinary features."),
}
SHEETED = list(IDENTITY)

SINGLES_PER_CHAR = 3
PAIRS = 4
MIN_S, MAX_S = 1.0, 3.0
# A subjects line naming more than one person is not a clean single-character shot.
CROWD = re.compile(r"\b(two|three|four|several|group|people|men|women|couple|crowd|others?)\b|\band (a|an|another)\b", re.I)

# Cost model fitted in research 0026: ~46 s fixed + ~16 s per video-second per clip,
# A6000 at the SECURE rate, plus one ~20 min pod boot.
RATE_PER_H = 0.59
BOOT_MIN = 20


def _framed(shot: dict, max_s: float = MAX_S) -> bool:
    st = ((shot.get("description") or {}).get("shot_type") or "").lower()
    return ("close" in st or "medium" in st) and MIN_S <= shot["duration_s"] <= max_s


def _even(cands: list[int], k: int) -> list[int]:
    """k indices spread evenly through a sorted candidate list."""
    if len(cands) <= k:
        return cands
    return [cands[round(i * (len(cands) - 1) / (k - 1))] for i in range(k)] if k > 1 else [cands[len(cands) // 2]]


def select(shots: dict, shot_map: dict) -> list[tuple[int, list[str]]]:
    picked = []
    for name in SHEETED:
        solo = [i for i, s in shots.items() if shot_map.get(i) == [name]
                and not CROWD.search((s.get("description") or {}).get("subjects") or "")]
        cands = sorted(i for i in solo if _framed(shots[i]))
        if len(cands) < SINGLES_PER_CHAR:  # George: top up from shots up to 4 s
            cands = sorted(i for i in solo if _framed(shots[i], max_s=4.0))
        picked += [(i, [name]) for i in _even(cands, SINGLES_PER_CHAR)]
    pairs = sorted(i for i, s in shots.items()
                   if len(shot_map.get(i, [])) == 2 and set(shot_map[i]) <= set(SHEETED) and _framed(s))
    picked += [(i, sorted(shot_map[i])) for i in _even(pairs, PAIRS)]
    return sorted(picked)


def keyframe_prompt(text_prompt: str, chars: list[str]) -> str:
    """The editor prompt for K's first frame: the arm's own text plus who the refs are."""
    if len(chars) == 1:
        who = ("The person in this frame is the person shown in the reference sheet (image 1): "
               "same face, hair, age and build.")
    else:
        who = ("The two people in this frame are the person from reference sheet 1 and the person "
               "from reference sheet 2: keep each one's face, hair, age and build.")
    return (f"A single film frame, not a grid. {text_prompt} {who} Ignore the sheets' layout, "
            "grey background and plain clothes; dress and light them as the scene describes.")


def build(run_dir: str) -> tuple[dict, dict]:
    shots = {s["index"]: s for s in json.load(open(f"{run_dir}/shots.json"))["shots"]}
    chars = json.load(open(f"{run_dir}/characters.json"))
    for ch in chars["characters"]:
        if ch["name"] in IDENTITY:
            ch["description"] = IDENTITY[ch["name"]]
    shot_map = build_character_shot_map(chars)

    strat = object.__new__(RunPodLtx25Strategy)  # skip __init__: no RunPod session
    strat._init_character_identity(shot_map, chars, style_prefix=STYLE_PREFIX)

    jobs, plan_shots, gpu_s = [], [], 0.0
    for idx, names in select(shots, shot_map):
        entry = shots[idx]
        prompt = strat.format_prompt(entry)
        frames = ltx25_frames(entry["duration_s"])
        base = {"shot": idx, "seed": idx, "frames": frames, "prompt": prompt}
        jobs.append({**base, "arm": "T"})
        jobs.append({**base, "arm": "K", "start_image": f"k_frames/{idx:04d}.png"})
        gpu_s += 2 * (46 + 16 * frames / 25)
        plan_shots.append({"shot": idx, "characters": names, "duration_s": entry["duration_s"],
                           "shot_type": entry["description"]["shot_type"], "frames": frames,
                           "keyframe_prompt": keyframe_prompt(prompt, names)})
    cost = (gpu_s / 3600 + BOOT_MIN / 60) * RATE_PER_H
    manifest = {"style_prefix": STYLE_PREFIX, "n_clips": len(jobs),
                "est_gpu_minutes": round(gpu_s / 60, 1), "est_cost_usd": round(cost, 2), "jobs": jobs}
    plan = {"style_prefix": STYLE_PREFIX, "identity": IDENTITY, "shots": plan_shots}
    return manifest, plan


if __name__ == "__main__":
    run_dir = sys.argv[1]
    m, p = build(run_dir)
    json.dump(m, open(f"{run_dir}/trial_manifest.json", "w"), indent=1)
    json.dump(p, open(f"{run_dir}/trial_plan.json", "w"), indent=1)
    by = {}
    for s in p["shots"]:
        by.setdefault("+".join(s["characters"]), []).append(s["shot"])
    print(json.dumps(by))
    print(f"{m['n_clips']} clips, ~{m['est_gpu_minutes']} GPU-min + {BOOT_MIN} min boot, ~${m['est_cost_usd']}")
