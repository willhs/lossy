"""Build research 0029's arm I (LTX-2.5 Ingredients IC-LoRA) for the trial. Spends nothing.

Usage (from the repo root, after make_trial_manifest.py and make_refs.py --sheets-only):
    uv run python docs/research/0029-character-reference-continuity/trial/make_ingredients_arm.py RUN_DIR

RUN_DIR is the trial's run directory: trial_manifest.json and trial_plan.json from
make_trial_manifest.py, and sheets/<character>.png from make_refs.py (the four klein
sheets, shared with arm K).

For each trial shot it writes:
  RUN_DIR/i_sheets/NNNN.png      the shot's characters' sheets side by side on black,
                                 at the clip's frame size (ingredients.compose_sheet)
and then RUN_DIR/trial_manifest_I.json: the arm's jobs. Same shots, seeds and frame
counts as arms T and K; the prompt is arm T's text inside the adapter's two-part form
(ingredients.format_sheet_prompt), so the only differences from T are the sheet
and the framing around the same words. Render it with:

    decode.py RUN_DIR --strategy runpod-ltx25-ingredients --ab-manifest RUN_DIR/trial_manifest_I.json
"""
import json
import os
import sys

import cv2

sys.path.insert(0, os.getcwd())
from ingredients import SheetPanel, compose_sheet, format_sheet_prompt  # noqa: E402
from strategies_video import RunPodLtx25IngredientsStrategy, RunPodLtx25Strategy  # noqa: E402

# make_refs.py draws each sheet as a 2x2 grid (front, 3/4, profile, full body); the
# panel description says so, as the card's own descriptions say "shown from multiple angles".
VIEWS = (" Shown four times in a 2x2 grid on a plain grey background: front view, three-quarter view, "
         "side profile and full body, in a plain grey t-shirt and dark trousers.")

# RunPodLtx25Strategy.WIDTH x HEIGHT: the sheet must have the clip's aspect (see compose_sheet).
FRAME = (RunPodLtx25Strategy.WIDTH, RunPodLtx25Strategy.HEIGHT)


def build(run_dir: str) -> dict:
    plan = json.load(open(os.path.join(run_dir, "trial_plan.json")))
    t_jobs = {j["shot"]: j for j in json.load(open(os.path.join(run_dir, "trial_manifest.json")))["jobs"]
              if j["arm"] == "T"}
    sheets = {}
    for name in plan["identity"]:
        path = os.path.join(run_dir, "sheets", f"{name}.png")
        img = cv2.imread(path)
        if img is None:
            raise SystemExit(f"missing sheet {path}: run make_refs.py RUN_DIR --sheets-only first "
                             "(four klein calls, about $0.02)")
        sheets[name] = img

    os.makedirs(os.path.join(run_dir, "i_sheets"), exist_ok=True)
    jobs = []
    for s in plan["shots"]:
        panels = [SheetPanel("Character", plan["identity"][c] + VIEWS, sheets[c]) for c in s["characters"]]
        rel = f"i_sheets/{s['shot']:04d}.png"
        cv2.imwrite(os.path.join(run_dir, rel), compose_sheet(panels, *FRAME))
        t = t_jobs[s["shot"]]
        jobs.append({"shot": s["shot"], "arm": "I", "seed": t["seed"], "frames": t["frames"],
                     "prompt": format_sheet_prompt(panels, t["prompt"]), "start_image": rel})
    return {"n_clips": len(jobs), "note": "arm I: Ingredients IC-LoRA, one composite sheet per shot", "jobs": jobs}


if __name__ == "__main__":
    import config
    from runpod_pod import GatedModelError, check_gated_access

    run_dir = sys.argv[1]
    m = build(run_dir)
    json.dump(m, open(os.path.join(run_dir, "trial_manifest_I.json"), "w"), indent=1)
    print(f"{m['n_clips']} arm I jobs -> {run_dir}/trial_manifest_I.json, sheets in {run_dir}/i_sheets/")
    # Last, so run_arm_i.sh stops here (exit 1, no pod, no retry loop) while the gate is closed.
    config.load_env()
    try:
        check_gated_access(RunPodLtx25IngredientsStrategy.PINNED_MODELS, os.environ.get("HF_TOKEN"))
    except GatedModelError as e:
        sys.exit(f"Error: {e}")
    print("HF gate OK: the adapter can be downloaded.")
