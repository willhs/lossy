"""Lay out a score dir for trial/score.py from a Star Wars IV K-vs-T decode (research 0029, SW4 paid check).

score.py was written for the Seinfeld trial and reads a run dir of its own shape:
trial_plan.json, trial_manifest.json, sheets/, k_frames/, source_kf/ and
clips/runpod-ltx25-ab/{T,K}/NNNN.mp4. This builds that from the plain decode output dirs
instead (`decode.py --strategy runpod-ltx25` and `--strategy runpod-ltx25-keyframe` on the
same window). Free: no network, no GPU.

A shot is scored only if the K arm really started it from a drawn keyframe (kf_NNNN.png
exists) and it has one or two characters. Text-fallback shots decode the same way in both
arms, so they say nothing about the keyframe and are left out. A multi-part clip is
joined into one file.

    uv run python docs/research/0029-character-reference-continuity/trial/make_sw4_score_inputs.py \
        output/sw4_v2 SCORE_DIR --start-index 684 --limit 30 --source-keyframes SRC_KEYFRAMES_DIR \
        [--style-prefix "..."]
    uv run python docs/research/0029-character-reference-continuity/trial/score.py SCORE_DIR models/face
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.getcwd())
import manifest  # noqa: E402
from decode import build_character_shot_map  # noqa: E402
from keyframes import plan_route  # noqa: E402
from strategies_video import RunPodLtx25KeyframeStrategy  # noqa: E402

ARM_DIR = {"T": "runpod-ltx25", "K": "runpod-ltx25-keyframe"}


def joined_clip(clips: str, shot: int, dst: str) -> bool:
    """Copy shot's clip to dst, joining its parts if it was split. False if no clip."""
    single = os.path.join(clips, f"{shot:04d}.mp4")
    parts = sorted(glob.glob(os.path.join(clips, f"{shot:04d}-[0-9][0-9].mp4")))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(single):
        shutil.copy(single, dst)
        return True
    if not parts:
        return False
    lst = dst + ".txt"
    with open(lst, "w") as f:
        f.writelines(f"file '{os.path.abspath(p)}'\n" for p in parts)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst,
                    "-c:v", "libx264", "-crf", "16", "-an", dst], check=True)
    os.remove(lst)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir")
    ap.add_argument("score_dir")
    ap.add_argument("--start-index", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--source-keyframes", help="the encode's keyframes/ dir (NNNN-PP.jpg), for the real-footage ceiling")
    ap.add_argument("--style-prefix", default=None)
    args = ap.parse_args()

    prompts, _ = manifest.load_shots(args.output_dir)
    if args.start_index:
        prompts = [p for p in prompts if p["index"] >= args.start_index]
    if args.limit:
        prompts = prompts[:args.limit]
    with open(manifest.characters_path(args.output_dir)) as f:
        characters_data = json.load(f)
    by_name = {c["name"]: c for c in characters_data["characters"]}

    probe = object.__new__(RunPodLtx25KeyframeStrategy)   # the same text prompt the T arm was given
    probe._init_character_identity(build_character_shot_map(characters_data), characters_data,
                                   args.style_prefix or manifest.load_style_prefix(args.output_dir))
    kf_dir = manifest.keyframes_dir(args.output_dir)
    out = args.score_dir
    shots, jobs, identity = [], [], {}
    for e in prompts:
        route = plan_route(e["index"], probe._character_shot_map.get(e["index"], []), probe._characters_by_name,
                           probe._cap_to_most_prominent)
        kf = os.path.join(kf_dir, f"kf_{e['index']:04d}.png")
        if not (route.keyframe and 1 <= len(route.characters) <= 2 and os.path.exists(kf)):
            continue
        t_clip = joined_clip(os.path.join(args.output_dir, "clips", ARM_DIR["T"]), e["index"],
                             os.path.join(out, "clips", "runpod-ltx25-ab", "T", f"{e['index']:04d}.mp4"))
        k_clip = joined_clip(os.path.join(args.output_dir, "clips", ARM_DIR["K"]), e["index"],
                             os.path.join(out, "clips", "runpod-ltx25-ab", "K", f"{e['index']:04d}.mp4"))
        if not (t_clip and k_clip):
            print(f"shot {e['index']}: a clip is missing (T {t_clip}, K {k_clip}); left out")
            continue
        os.makedirs(os.path.join(out, "k_frames"), exist_ok=True)
        shutil.copy(kf, os.path.join(out, "k_frames", f"{e['index']:04d}.png"))
        if args.source_keyframes:
            srcs = sorted(glob.glob(os.path.join(args.source_keyframes, f"{e['index']:04d}-*.jpg")))
            if srcs:
                os.makedirs(os.path.join(out, "source_kf"), exist_ok=True)
                shutil.copy(srcs[len(srcs) // 2], os.path.join(out, "source_kf", f"{e['index']:04d}.jpg"))
        for c in route.characters:
            identity[c] = by_name[c]["description"].strip()
        shots.append({"shot": e["index"], "characters": list(route.characters), "duration_s": e["duration_s"],
                      "shot_type": e.get("shot_type")})
        text = probe.format_prompt(e)
        jobs += [{"arm": a, "shot": e["index"], "prompt": text} for a in ("T", "K")]
    for c in identity:
        sheet = os.path.join(args.output_dir, "characters", f"{c}_sheet.png")
        os.makedirs(os.path.join(out, "sheets"), exist_ok=True)
        shutil.copy(sheet, os.path.join(out, "sheets", f"{c}.png"))
    json.dump({"style_prefix": args.style_prefix or manifest.load_style_prefix(args.output_dir),
               "identity": identity, "shots": shots}, open(os.path.join(out, "trial_plan.json"), "w"), indent=1)
    json.dump({"jobs": jobs}, open(os.path.join(out, "trial_manifest.json"), "w"), indent=1)
    print(f"{len(shots)} scorable shots, characters: {', '.join(identity)}; written to {out}")


if __name__ == "__main__":
    main()
