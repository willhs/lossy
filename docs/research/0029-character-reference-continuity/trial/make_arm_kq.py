"""Build research 0029's bake-off arm KQ: arm K with Qwen-Image-Edit-2511 as the keyframe editor.

Usage (from the repo root; RUN_DIR as for make_refs.py, with sheets/ already there):
    uv run python docs/research/0029-character-reference-continuity/trial/make_arm_kq.py RUN_DIR [--price-only]

Same sheets, same keyframe prompt, same size, same seed as arm K (make_refs.py); only the
editor differs (fal-ai/qwen-image-edit-2511 instead of fal-ai/flux-2/klein/4b/edit). Writes
RUN_DIR/kq_frames/NNNN.png and RUN_DIR/trial_manifest_KQ.json (arm K's jobs, start_image
swapped), rendered with:

    decode.py RUN_DIR --strategy runpod-ltx25 --ab-manifest RUN_DIR/trial_manifest_KQ.json

Every call's estimated price is appended to RUN_DIR/refs_cost.jsonl (fal bills $0.03 per
output megapixel; input megapixels are counted too, as an upper bound).
"""
import json
import os
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
import make_refs  # noqa: E402

EDIT_QWEN = "fal-ai/qwen-image-edit-2511"
PRICE_PER_MP = 0.03   # fal page, read 2026-11-20


def estimate(plan: dict) -> float:
    w, h = make_refs.FRAME_SIZE
    total = 0.0
    for s in plan["shots"]:
        mp = make_refs._mp(w, h) + len(s["characters"]) * make_refs._mp(*make_refs.SHEET_SIZE)
        total += PRICE_PER_MP * mp
    return total


def main(run_dir: str, price_only: bool) -> None:
    plan = json.load(open(os.path.join(run_dir, "trial_plan.json")))
    if price_only:
        print(f"arm KQ keyframes, upper bound: ${estimate(plan):.2f} "
              f"(output only: ${PRICE_PER_MP * make_refs._mp(*make_refs.FRAME_SIZE) * len(plan['shots']):.2f})")
        return
    config.load_env()
    import fal_client

    os.makedirs(os.path.join(run_dir, "kq_frames"), exist_ok=True)
    sheet_urls = {}
    for name in plan["identity"]:
        sheet_urls[name] = fal_client.upload_file(os.path.join(run_dir, "sheets", f"{name}.png"))
    w, h = make_refs.FRAME_SIZE
    for s in plan["shots"]:
        path = os.path.join(run_dir, "kq_frames", f"{s['shot']:04d}.png")
        if os.path.exists(path):
            continue
        refs = [sheet_urls[c] for c in s["characters"]]
        data = make_refs._call(fal_client, EDIT_QWEN, {
            "prompt": s["keyframe_prompt"], "image_urls": refs, "seed": s["shot"],
            "image_size": {"width": w, "height": h}, "output_format": "png"})
        open(path, "wb").write(data)
        mp = make_refs._mp(w, h) + len(refs) * make_refs._mp(*make_refs.SHEET_SIZE)
        make_refs._log(run_dir, {"kind": "keyframe", "arm": "KQ", "shot": s["shot"], "model": EDIT_QWEN,
                                 "seed": s["shot"], "usd": round(PRICE_PER_MP * mp, 4)})
        print(f"keyframe {s['shot']:04d} ({'+'.join(s['characters'])})")

    base = json.load(open(os.path.join(run_dir, "trial_manifest.json")))
    jobs = [{**j, "arm": "KQ", "start_image": f"kq_frames/{int(j['shot']):04d}.png"}
            for j in base["jobs"] if j["arm"] == "K"]
    out = {"style_prefix": base["style_prefix"], "n_clips": len(jobs), "jobs": jobs}
    json.dump(out, open(os.path.join(run_dir, "trial_manifest_KQ.json"), "w"), indent=2)
    print(f"{len(jobs)} arm KQ jobs -> {run_dir}/trial_manifest_KQ.json")


if __name__ == "__main__":
    main(sys.argv[1], "--price-only" in sys.argv)
