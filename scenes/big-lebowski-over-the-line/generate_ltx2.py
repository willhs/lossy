#!/usr/bin/env python3
"""One-off LTX-2 comparison render of the same 6-shot manifest as scene.json.

Drives RunPodLtx2Strategy directly rather than through decode.py's full
shots.json/characters.json pipeline -- this scene has no per-character
identity map, just self-contained per-shot prompts, so the low-level
_generate_one_clip/_upload_start_frame calls are enough. keep_pod defaults to
False (terminate on exit) to avoid the idle-pod billing the 0023 rehearsal
flagged as its single largest avoidable cost.

Clips are named s{shot:02d}_p{part:02d}.mp4 and skipped if already present in
CLIPS_DIR, so a run that dies partway (e.g. a pod timeout) can be re-invoked
and only the missing shots regenerate -- same resume contract as the fal
generate.py script.

Usage:
    cd lossy
    .venv/bin/python3 scenes/big-lebowski-over-the-line/generate_ltx2.py
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def load_env(path):
    for line in open(path):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v.strip().strip('"').strip("'"))


load_env(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env"))

from strategies_video import RunPodLtx2Strategy

HERE = os.path.dirname(os.path.abspath(__file__))
CLIPS_DIR = os.path.join(HERE, "work-ltx2")
OUTPUT = os.path.join(HERE, "over-the-line-ltx2.mp4")
TARGET_S = 4.8  # < LTX-2's 121-frame/25fps native max (~4.84s) -- one part per call, no internal split

os.makedirs(CLIPS_DIR, exist_ok=True)

with open(os.path.join(HERE, "scene.json")) as f:
    spec = json.load(f)


def entry_for(prompt: str) -> dict:
    return {"description": {"action": prompt}}


def clip_path(si: int, pi: int) -> str:
    return os.path.join(CLIPS_DIR, f"s{si:02d}_p{pi:02d}.mp4")


def main():
    strategy = RunPodLtx2Strategy(output_dir=CLIPS_DIR, keep_pod=False)
    seed = spec.get("seed", 4200)
    pod_ready = False
    all_paths = []  # (si, pi, path) in manifest order, for later stitching

    for si, shot in enumerate(spec["shots"]):
        base_path = clip_path(si, 0)
        frames = strategy._target_frames(TARGET_S)

        if os.path.exists(base_path):
            print(f"shot {si + 1} part 0: skip (exists) {base_path}")
        else:
            if not pod_ready:
                strategy._ensure_pod()
                pod_ready = True
            prompt = strategy.format_prompt(entry_for(shot["prompt"]))
            print(f"shot {si + 1} part 0 seed={seed}")
            result = strategy._generate_one_clip(
                prompt, CLIPS_DIR, os.path.basename(base_path), frames, seed)
            if result is None:
                print(f"  FAILED shot {si + 1} part 0, skipping its continuations")
                seed += len(shot.get("continue", [])) + 1
                continue
        all_paths.append((si, 0, base_path))
        seed += 1
        prev_path = base_path

        for ci, cont_prompt in enumerate(shot.get("continue", [])):
            part_path = clip_path(si, ci + 1)
            if os.path.exists(part_path):
                print(f"shot {si + 1} part {ci + 1}: skip (exists) {part_path}")
                all_paths.append((si, ci + 1, part_path))
                seed += 1
                prev_path = part_path
                continue
            if not pod_ready:
                strategy._ensure_pod()
                pod_ready = True
            cont_formatted = strategy.format_prompt(entry_for(cont_prompt))
            print(f"shot {si + 1} part {ci + 1} (I2V chained) seed={seed}")
            start_image = strategy._upload_start_frame(prev_path, si, ci)
            result = strategy._generate_one_clip(
                cont_formatted, CLIPS_DIR, os.path.basename(part_path), frames, seed,
                start_image=start_image)
            if result is None:
                print(f"  FAILED shot {si + 1} part {ci + 1}, stopping this shot's chain")
                seed += 1
                break
            all_paths.append((si, ci + 1, part_path))
            seed += 1
            prev_path = part_path

    if pod_ready:
        strategy.mark_clean_exit()

    if not all_paths:
        sys.exit("No clips available -- nothing to stitch")

    missing_shots = sorted({si + 1 for si in range(len(spec["shots"]))} - {si + 1 for si, _, _ in all_paths})
    if missing_shots:
        print(f"\nWARNING: shots with no clip at all: {missing_shots} -- stitching without them")

    clip_paths = [p for _, _, p in all_paths]
    silent = os.path.join(CLIPS_DIR, "_silent.mp4")
    inputs = []
    for c in clip_paths:
        inputs += ["-i", c]
    fc = "".join(f"[{i}:v]" for i in range(len(clip_paths))) + f"concat=n={len(clip_paths)}:v=1:a=0,format=yuv420p[v]"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *inputs,
                    "-filter_complex", fc, "-map", "[v]",
                    "-c:v", "libx264", "-crf", "28", "-preset", "slow",
                    "-movflags", "+faststart", "-an", silent], check=True)
    subprocess.run(["cp", silent, OUTPUT], check=True)
    dur = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", OUTPUT]
    ).decode().strip()
    print(f"\nDONE -> {OUTPUT} ({dur}s, {len(clip_paths)} clips)")


if __name__ == "__main__":
    main()
