"""Write the proposed style A/B test as an exact job list (research 0028). Spends nothing.

Usage (from the repo root):
    uv run python docs/research/0028-style-drift/make_ab_manifest.py OUTPUT_DIR OUT_JSON

Every job uses the same seed and frame count as the original runpod-ltx25 clip, so the
existing clip is the free baseline for its shot. Prompts are built with the real
RunPodLtx25Strategy.format_prompt; only the arm's change is applied on top.
"""
import copy
import json
import os
import sys

sys.path.insert(0, os.getcwd())
from decode import build_character_shot_map  # noqa: E402
from strategies_video import RunPodLtx25Strategy, ltx25_frames  # noqa: E402

STYLE_PREFIX = ("Live-action 1990s American TV sitcom footage of real actors, "
                "photographed on 35mm film, photorealistic skin and lighting.")

# Kramer's block rewritten in photographic terms: same geometry, none of the
# caricature cues the analysis flagged (very thin, angular, mop, stands up high,
# sharp pointed, wide expressive mouth, lanky, hunched).
KRAMER_PHOTO = ("A real man in his late 30s, tall and slim, with thick dark curly hair "
                "brushed upward and back from a high forehead. Long face, prominent straight "
                "nose, high cheekbones, deep-set dark eyes under a heavy brow, natural skin "
                "texture and light stubble.")

KRAMER_SHOTS = [79, 98, 106, 152, 246, 256]   # Kramer block only, cartoon at baseline, one part
PAIR_SHOTS = [125, 221]                        # Elaine + George blocks, cartoon at baseline
DETERMINISM_SHOT = 98                          # re-render unchanged: does seed+prompt reproduce?

ARMS = {
    "control_rerender": {"prefix": False, "kramer_photo": False},
    "A_prefix": {"prefix": True, "kramer_photo": False},
    "B_photo_block": {"prefix": False, "kramer_photo": True},
    "C_prefix_and_block": {"prefix": True, "kramer_photo": True},
}

# Cost model fitted in research 0026: ~46 s fixed + ~16 s per video-second per clip,
# A6000 at the SECURE rate, plus one ~20 min pod boot.
RATE_PER_H = 0.59
BOOT_MIN = 20


def build(out_dir: str) -> dict:
    shots = {s["index"]: s for s in json.load(open(f"{out_dir}/shots.json"))["shots"]}
    chars = json.load(open(f"{out_dir}/characters.json"))

    def strat(kramer_photo: bool) -> RunPodLtx25Strategy:
        c = copy.deepcopy(chars)
        if kramer_photo:
            for ch in c["characters"]:
                if ch["name"] == "cosmo_kramer":
                    ch["description"] = KRAMER_PHOTO
        s = object.__new__(RunPodLtx25Strategy)  # skip __init__: no RunPod session
        s._init_character_identity(build_character_shot_map(c), c)
        return s

    plan = [(DETERMINISM_SHOT, "control_rerender")]
    plan += [(i, a) for i in KRAMER_SHOTS for a in ("A_prefix", "B_photo_block", "C_prefix_and_block")]
    plan += [(i, "A_prefix") for i in PAIR_SHOTS]

    jobs, gpu_s = [], 0.0
    for idx, arm in plan:
        cfg = ARMS[arm]
        entry = shots[idx]
        prompt = strat(cfg["kramer_photo"]).format_prompt(entry)
        if cfg["prefix"]:
            prompt = f"{STYLE_PREFIX} {prompt}"
        frames = ltx25_frames(entry["duration_s"])
        gpu_s += 46 + 16 * frames / 25
        jobs.append({"shot": idx, "arm": arm, "seed": idx, "frames": frames,
                     "baseline_clip": f"clips/runpod-ltx25/{idx:04d}.mp4", "prompt": prompt})
    cost = (gpu_s / 3600 + BOOT_MIN / 60) * RATE_PER_H
    return {"style_prefix": STYLE_PREFIX, "kramer_photo_block": KRAMER_PHOTO,
            "n_clips": len(jobs), "est_gpu_minutes": round(gpu_s / 60, 1),
            "est_cost_usd": round(cost, 2), "jobs": jobs}


if __name__ == "__main__":
    m = build(sys.argv[1])
    with open(sys.argv[2], "w") as f:
        json.dump(m, f, indent=1)
    print(f"{m['n_clips']} clips, ~{m['est_gpu_minutes']} GPU-min + {BOOT_MIN} min boot, ~${m['est_cost_usd']}")
