---
id: plan-0002
type: spec
purpose: "Implementation plan for the decoder: prompts.json → video clips → reconstructed film."
tags: ["plan", "decoder", "video-generation", "replicate"]
related: ["./task.md", "../../research/0003-prompt-to-video/research.md"]
created: 2026-03-14
updated: 2026-03-14
---

# Prompts-to-Video v1 Implementation Plan

## Overview

Build `decode.py` — a standalone CLI tool that reads the prompt manifest (prompts.json), generates a video clip per shot via Replicate's Wan 2.2 API, and stitches clips into a single reconstructed film using FFmpeg.

**Primary Goal**: Reconstruct Star Wars EP IV from text prompts for under $60.

**Approach**: Follow the same patterns as `encode.py` — stateless CLI, disk-based I/O, resume support, incremental saves. Separate file (`decode.py`) since the decoder is a distinct pipeline stage with different dependencies.

## Current State Analysis

### Input Format

`output/star_wars_iv_v2/prompts.json` — 1,161 entries, each containing:

```json
{
  "index": 1,
  "start_s": 28.779,
  "end_s": 31.406,
  "duration_s": 2.628,
  "camera_motion_detected": "zoom",
  "dialogue": null,
  "description": {
    "shot_type": "medium wide",
    "camera_movement": "slow zoom out",
    "subjects": "The 'STAR WARS' logo...",
    "action": "The camera slowly zooms out...",
    "lighting": "The logo is brightly lit...",
    "color_palette": "Primarily black and yellow...",
    "mood": "Epic, iconic, adventurous, classic.",
    "setting": "Deep space, with a field of stars visible."
  }
}
```

### Key Constraints

- **Wan 2.2 Fast**: 81-100 frames, 480p (832x480), 16fps default = ~5.06s per clip
- **Duration mismatch**: Original shots range from <1s to 28s. All clips will be ~5s.
- **Cost**: ~$0.05/clip on Replicate at 480p

## Desired End State

- `decode.py` CLI that reads prompts.json and generates clips
- `output/star_wars_iv_v2/clips/` directory with one MP4 per shot
- `output/star_wars_iv_v2/reconstructed.mp4` — full stitched film
- Resume support — can stop and restart without re-generating clips
- Start-offset flag to skip credits/crawl

## What We're NOT Doing

- Audio/soundtrack reconstruction
- Image-to-video conditioning (using keyframes as starting frames)
- Multiple video gen model support (Wan 2.2 only for v1)
- Quality optimization or prompt engineering iterations
- Comparator/side-by-side view (separate task)

---

## Phase 1: Core Decoder

### Overview

Build the prompt formatter, Replicate API integration, and clip download with resume support.

### Tasks

#### 1. Create `decode.py` with prompt formatter

- [x] Create `decode.py` with imports, .env loading, and `format_prompt()` function

```python
#!/usr/bin/env python3
"""
lossy decoder: prompt manifest → video clips → reconstructed film.

Usage:
    python decode.py output/star_wars_iv_v2 [--start-index 7] [--limit 20]
"""

import argparse
import json
import os
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
    duration = entry.get("duration_s", 5.0)

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
```

#### 2. Add Replicate API integration

- [x] Add `generate_clip()` function that calls Replicate and downloads the output

```python
def generate_clip(prompt: str, output_path: str, seed: int | None = None) -> bool:
    """Generate a video clip via Replicate Wan 2.2 and save to output_path.

    Returns True on success, False on failure.
    """
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

        # Output is a FileOutput URL — download it
        import urllib.request
        if hasattr(output, "url"):
            url = output.url
        elif isinstance(output, str):
            url = output
        else:
            url = str(output)

        urllib.request.urlretrieve(url, output_path)
        return True

    except Exception as e:
        print(f"  Error generating clip: {e}")
        return False
```

#### 3. Add decode loop with resume support

- [x] Add `run_decode()` function with progress tracking, resume, and incremental saves

```python
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
```

#### 4. Add CLI entry point

- [x] Add argparse CLI matching encode.py patterns

```python
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
```

### Success Criteria

#### Automated Verification:
- [x] Run: `python decode.py --help` — prints usage without errors
- [x] Run: `python -c "from decode import format_prompt; print('ok')"` — imports cleanly

#### Manual Verification:
- [x] Manual: `format_prompt()` produces sensible video-gen prompts when tested on a few entries from prompts.json

---

## Phase 2: FFmpeg Stitcher

### Overview

Concatenate generated clips into a single video, with optional speed-adjustment to match original shot durations.

### Tasks

#### 1. Add stitch_clips function

- [x] Add `stitch_clips()` to decode.py

```python
import subprocess


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
```

### Success Criteria

#### Automated Verification:
- [ ] Run: `python decode.py output/star_wars_iv_v2 --stitch --start-index 7` — produces reconstructed.mp4 (after clips exist)

#### Manual Verification:
- [ ] Manual: Play reconstructed.mp4 and verify it's a sequence of generated clips at varying speeds

---

## Phase 3: Test Batch

### Overview

Run the decoder on 10-20 clips to verify everything works end-to-end before committing to the full run.

### Tasks

#### 1. Install replicate package

- [x] Run: `pip install replicate` in the project venv

#### 2. Set Replicate API token

- [x] Add `REPLICATE_API_TOKEN` to `.env` (Replicate SDK reads this env var automatically)

#### 3. Run test batch

- [x] Run: `python decode.py output/star_wars_iv_v2 --start-index 10 --limit 15`

#### 4. Review results

- [x] Manual: Check `output/star_wars_iv_v2/clips/` — 15 MP4 files exist
- [x] Manual: Watch a few clips — recognizable scenes. Character/robot shots (C-3PO, R2-D2) look good. Abstract space shots less convincing.
- [x] Manual: Check `decode_progress.json` — cost estimate should be ~$0.75

#### 5. Test stitching

- [x] Run: `python decode.py output/star_wars_iv_v2 --stitch --start-index 10 --limit 15`
- [x] Manual: Play stitched output. Known issue: very short clips show as stills in VLC after speed-adjust (setpts factor too low produces sub-frame durations).

#### 6. Verify resume

- [x] Run: `python decode.py output/star_wars_iv_v2 --start-index 10 --limit 15` again (resume confirmed — skipped shot 10)
- [x] Verify it skips all 15 already-generated clips and exits quickly

---

## Final Checklist

- [x] `decode.py` runs end-to-end on a test batch
- [x] Resume works (re-running skips completed clips)
- [x] FFmpeg stitcher produces a playable video
- [x] Cost for test batch matches estimates ($0.75 for 15 clips)
- [x] Progress tracking works (decode_progress.json updated correctly)

## References

- Task: `docs/work/tasks/0002-prompts-to-video-v1/task.md`
- Research: `docs/research/0003-prompt-to-video/research.md`
- Architecture: `docs/design/architecture.md`
- Encoder (pattern reference): `encode.py`
