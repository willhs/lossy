"""Make arm K's references with FLUX.2 [klein] 4B on fal (research 0029 trial). Spends ~$0.1-0.3.

Usage (from the repo root, after make_trial_manifest.py):
    uv run python docs/research/0029-character-reference-continuity/trial/make_refs.py RUN_DIR [--sheets-only]

1. One 2x2 reference sheet per character (front, 3/4, profile, full body), text-to-image
   from the name-free identity block only -> RUN_DIR/sheets/<name>.png
2. One first frame per trial shot, image edit from that shot's sheets (<=2) and the
   shot's own prompt -> RUN_DIR/k_frames/NNNN.png

The same open weights the note proposes to self-host (Apache-2.0); fal runs them so the
pod only has to do video. Resumable: existing images are kept. Every call's estimated
price is appended to RUN_DIR/refs_cost.jsonl so the trial's $1.25 cap counts it.
"""
import json
import os
import sys
import time

import httpx

sys.path.insert(0, os.getcwd())
import config  # noqa: E402

T2I = "fal-ai/flux-2/klein/4b"
EDIT = "fal-ai/flux-2/klein/4b/edit"
PRICE_T2I_PER_MP = 0.005   # fal page, read 2026-11-15
PRICE_EDIT_PER_MP = 0.01   # fal page; charged on output MP, counted here on inputs too (upper bound)
SHEET_SIZE = (1024, 1024)
FRAME_SIZE = (1280, 704)   # RunPodLtx25Strategy.WIDTH x HEIGHT
SHEET_SEED = 1234

SHEET_PROMPT = (
    "Character reference sheet: one photograph divided into a 2x2 grid on a plain mid-grey studio "
    "background, showing the same real person four times. Top left: front-facing head-and-shoulders "
    "portrait. Top right: three-quarter view head-and-shoulders. Bottom left: side profile "
    "head-and-shoulders. Bottom right: full-body standing pose. Neutral expression, soft even studio "
    "lighting, plain grey t-shirt and dark trousers. The person: {block} Photorealistic 35mm film "
    "photograph, natural skin texture, no text, no labels.")


def _log(run_dir: str, rec: dict) -> None:
    with open(os.path.join(run_dir, "refs_cost.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")


def _mp(w: int, h: int) -> float:
    return w * h / 1e6


def _call(fal_client, model: str, args: dict) -> bytes:
    err = None
    for attempt in range(3):
        try:
            res = fal_client.subscribe(model, arguments=args, with_logs=False)
            if any(res.get("has_nsfw_concepts") or []):
                raise RuntimeError("safety checker blanked the image")
            r = httpx.get(res["images"][0]["url"], follow_redirects=True, timeout=120)
            r.raise_for_status()
            return r.content
        except Exception as e:  # transient API errors
            err = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"{model} failed: {err}")


def main(run_dir: str, sheets_only: bool) -> None:
    config.load_env()
    import fal_client

    plan = json.load(open(os.path.join(run_dir, "trial_plan.json")))
    os.makedirs(os.path.join(run_dir, "sheets"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "k_frames"), exist_ok=True)

    sheet_urls = {}
    for i, (name, block) in enumerate(plan["identity"].items()):
        path = os.path.join(run_dir, "sheets", f"{name}.png")
        if not os.path.exists(path):
            w, h = SHEET_SIZE
            prompt = SHEET_PROMPT.format(block=block)
            data = _call(fal_client, T2I, {"prompt": prompt, "seed": SHEET_SEED + i,
                                           "image_size": {"width": w, "height": h},
                                           "output_format": "png"})
            open(path, "wb").write(data)
            _log(run_dir, {"kind": "sheet", "name": name, "model": T2I, "seed": SHEET_SEED + i,
                           "usd": round(PRICE_T2I_PER_MP * _mp(w, h), 4), "prompt": prompt})
            print(f"sheet {name}")
        if not sheets_only:
            sheet_urls[name] = fal_client.upload_file(path)
    if sheets_only:
        return

    for s in plan["shots"]:
        path = os.path.join(run_dir, "k_frames", f"{s['shot']:04d}.png")
        if os.path.exists(path):
            continue
        refs = [sheet_urls[c] for c in s["characters"]]
        w, h = FRAME_SIZE
        data = _call(fal_client, EDIT, {"prompt": s["keyframe_prompt"], "image_urls": refs,
                                        "seed": s["shot"], "image_size": {"width": w, "height": h},
                                        "output_format": "png"})
        open(path, "wb").write(data)
        mp = _mp(w, h) + len(refs) * _mp(*SHEET_SIZE)
        _log(run_dir, {"kind": "keyframe", "shot": s["shot"], "model": EDIT, "seed": s["shot"],
                       "usd": round(PRICE_EDIT_PER_MP * mp, 4)})
        print(f"keyframe {s['shot']:04d} ({'+'.join(s['characters'])})")

    total = sum(json.loads(line)["usd"] for line in open(os.path.join(run_dir, "refs_cost.jsonl")))
    print(f"fal spend so far (upper bound): ${total:.3f}")


if __name__ == "__main__":
    main(sys.argv[1], "--sheets-only" in sys.argv)
