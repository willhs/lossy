#!/usr/bin/env python3
"""
lossy decoder: prompt manifest → video clips → reconstructed film.

Usage:
    python decode.py output/star_wars_iv_v2 [--start-index 7] [--limit 20]
    python decode.py output/star_wars_iv_v2 --stitch [--start-index 7]
"""

import argparse
import json
import os
import subprocess
import sys
import time


def load_env():
    """Load .env file if present."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path) as ef:
            for line in ef:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


def format_prompt(entry: dict) -> str:
    """Convert a structured prompt entry into a flat text prompt for video generation.

    Combines the description fields into a single cinematic prompt string
    that video generation models respond well to.
    """
    desc = entry["description"]

    parts = []

    # Lead with shot type and camera movement
    shot_type = desc.get("shot_type", "")
    camera = desc.get("camera_movement", "")
    if shot_type and camera:
        parts.append(f"Cinematic {shot_type} shot, {camera}.")
    elif shot_type:
        parts.append(f"Cinematic {shot_type} shot.")

    # Core content: action is the most important for video gen
    action = desc.get("action", "")
    if action:
        parts.append(action)

    # Subjects if not already covered by action
    subjects = desc.get("subjects", "")
    if subjects:
        if isinstance(subjects, list):
            subjects = ", ".join(subjects)
        # Only add if action doesn't already describe subjects well
        if len(str(action)) < 50:
            parts.append(subjects)

    # Visual style
    lighting = desc.get("lighting", "")
    if lighting:
        parts.append(lighting)

    palette = desc.get("color_palette", "")
    if palette:
        if isinstance(palette, list):
            palette = ", ".join(palette)
        parts.append(f"Color palette: {palette}.")

    mood = desc.get("mood", "")
    if mood:
        parts.append(f"Mood: {mood}.")

    setting = desc.get("setting", "")
    if setting:
        parts.append(setting)

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Video generation via Replicate
# ---------------------------------------------------------------------------

def generate_clip(prompt: str, output_path: str, seed: int | None = None) -> bool:
    """Generate a video clip via Replicate Wan 2.2 and save to output_path.

    Returns True on success, False on failure.
    """
    import httpx
    import replicate

    try:
        input_params = {
            "prompt": prompt,
            "num_frames": 81,
            "aspect_ratio": "16:9",
            "frames_per_second": 16,
        }
        if seed is not None:
            input_params["seed"] = seed

        output = replicate.run("wan-video/wan-2.2-t2v-fast", input=input_params)

        # Output is a FileOutput — download it
        if hasattr(output, "url"):
            url = output.url
        elif isinstance(output, str):
            url = output
        else:
            url = str(output)

        resp = httpx.get(url, follow_redirects=True)
        resp.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(resp.content)
        return True

    except Exception as e:
        print(f"  Error generating clip: {e}")
        return False


# ---------------------------------------------------------------------------
# Decode loop
# ---------------------------------------------------------------------------

def run_decode(args):
    """Main decode loop: read prompts, generate clips, track progress."""
    output_dir = args.output_dir
    prompts_path = os.path.join(output_dir, "prompts.json")

    if not os.path.exists(prompts_path):
        print(f"Error: {prompts_path} not found. Run encoder first.")
        sys.exit(1)

    with open(prompts_path) as f:
        prompts = json.load(f)

    # Apply start index and limit
    if args.start_index:
        prompts = [p for p in prompts if p["index"] >= args.start_index]
        print(f"Starting from shot index {args.start_index}")

    if args.limit:
        prompts = prompts[:args.limit]
        print(f"Processing {len(prompts)} shots")

    clips_dir = os.path.join(output_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    # Track progress for resume
    progress_path = os.path.join(output_dir, "decode_progress.json")
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            progress = json.load(f)
    else:
        progress = {"completed": [], "failed": [], "total_cost_estimate": 0.0}

    completed_set = set(progress["completed"])
    total = len(prompts)
    generated = 0
    errors = 0

    print(f"Generating {total} clips ({len(completed_set)} already done)...")

    for entry in prompts:
        idx = entry["index"]

        if idx in completed_set:
            continue

        clip_path = os.path.join(clips_dir, f"{idx:04d}.mp4")

        # Skip if file already exists (belt and suspenders with progress.json)
        if os.path.exists(clip_path):
            completed_set.add(idx)
            progress["completed"].append(idx)
            continue

        prompt_text = format_prompt(entry)
        print(f"  Shot {idx} ({generated + 1}/{total - len(completed_set)} remaining)...")

        success = generate_clip(prompt_text, clip_path, seed=idx)

        if success:
            generated += 1
            progress["completed"].append(idx)
            progress["total_cost_estimate"] += 0.05
            completed_set.add(idx)
        else:
            errors += 1
            progress["failed"].append(idx)
            # Retry once after a short wait
            print(f"  Retrying shot {idx} in 10s...")
            time.sleep(10)
            success = generate_clip(prompt_text, clip_path, seed=idx)
            if success:
                generated += 1
                progress["completed"].append(idx)
                progress["total_cost_estimate"] += 0.05
                completed_set.add(idx)
                # Remove from failed since retry worked
                progress["failed"] = [f for f in progress["failed"] if f != idx]

        # Save progress every 5 clips
        if generated % 5 == 0:
            with open(progress_path, "w") as f:
                json.dump(progress, f, indent=2)

        if errors > 20:
            print("Too many errors, saving progress and stopping.")
            break

    # Final save
    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2)

    print(f"\nDone. Generated {generated} clips.")
    print(f"  Total: {len(progress['completed'])} completed, {len(progress['failed'])} failed")
    print(f"  Estimated cost: ${progress['total_cost_estimate']:.2f}")


# ---------------------------------------------------------------------------
# FFmpeg stitcher
# ---------------------------------------------------------------------------

def stitch_clips(args):
    """Concatenate all clips into a single reconstructed video.

    Speed-adjusts each clip to match original shot duration using FFmpeg's
    setpts filter.
    """
    output_dir = args.output_dir
    clips_dir = os.path.join(output_dir, "clips")
    prompts_path = os.path.join(output_dir, "prompts.json")

    if not os.path.exists(clips_dir):
        print(f"Error: {clips_dir} not found. Run decode first.")
        sys.exit(1)

    with open(prompts_path) as f:
        prompts = json.load(f)

    if args.start_index:
        prompts = [p for p in prompts if p["index"] >= args.start_index]

    # Collect existing clips in order
    clip_entries = []
    for entry in prompts:
        idx = entry["index"]
        clip_path = os.path.join(clips_dir, f"{idx:04d}.mp4")
        if os.path.exists(clip_path):
            clip_entries.append((clip_path, entry))

    if not clip_entries:
        print("No clips found to stitch.")
        sys.exit(1)

    print(f"Stitching {len(clip_entries)} clips...")

    # Speed-adjust each clip to match original duration, write to temp dir
    adjusted_dir = os.path.join(output_dir, "adjusted")
    os.makedirs(adjusted_dir, exist_ok=True)

    clip_duration = 81 / 16  # ~5.0625s at default settings
    concat_list = []

    for clip_path, entry in clip_entries:
        idx = entry["index"]
        original_duration = entry["duration_s"]
        adjusted_path = os.path.join(adjusted_dir, f"{idx:04d}.mp4")

        if not os.path.exists(adjusted_path):
            # setpts factor: >1 slows down (longer), <1 speeds up (shorter)
            speed_factor = original_duration / clip_duration

            subprocess.run(
                [
                    "ffmpeg", "-i", clip_path,
                    "-filter:v", f"setpts={speed_factor}*PTS",
                    "-an", "-y", adjusted_path,
                ],
                capture_output=True,
            )

        concat_list.append(adjusted_path)

    # Write concat list file
    concat_file = os.path.join(output_dir, "concat.txt")
    with open(concat_file, "w") as f:
        for path in concat_list:
            f.write(f"file '{os.path.abspath(path)}'\n")

    # Concatenate
    output_path = os.path.join(output_dir, "reconstructed.mp4")
    subprocess.run(
        [
            "ffmpeg", "-f", "concat", "-safe", "0",
            "-i", concat_file,
            "-c", "copy", "-y", output_path,
        ],
        capture_output=True,
    )

    print(f"Reconstructed film saved to {output_path}")

    # Report stats
    total_original = sum(e["duration_s"] for _, e in clip_entries)
    print(f"  Original duration: {total_original:.1f}s ({total_original / 60:.1f}min)")
    print(f"  Clips used: {len(clip_entries)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="lossy decoder: prompts → video")
    parser.add_argument("output_dir", help="Output directory containing prompts.json")
    parser.add_argument("--start-index", type=int, default=None,
                        help="Skip shots before this index (e.g., 7 to skip credits)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N shots (after start-index)")
    parser.add_argument("--stitch", action="store_true",
                        help="Only run the stitching step (skip generation)")
    args = parser.parse_args()

    load_env()

    if args.stitch:
        stitch_clips(args)
    else:
        run_decode(args)


if __name__ == "__main__":
    main()
