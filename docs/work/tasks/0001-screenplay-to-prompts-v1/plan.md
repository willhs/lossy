---
id: plan-0001
type: spec
purpose: "Implementation plan for two-stage encoding pipeline: shot detection to prompt generation."
tags: ["plan", "encoder", "pipeline"]
related: ["./task.md", "research/shot-to-prompt-landscape.md"]
---

# Screenplay-to-Prompts v1 Implementation Plan

## Overview

Refactor `encode.py` into a two-stage pipeline. Stage 1 detects shots and extracts multi-frame keyframes. Stage 2 enriches shots with subtitles and camera motion metadata, then sends frames to Gemini Flash-Lite to generate descriptive prompts. All intermediate outputs persist to disk so stages can be re-run independently.

**Primary Goal**: Generate a prompt manifest for Star Wars EP IV where every shot has a structured text description.

**Approach**: Disk-based pipeline with subcommands. Test on small subsets (first 20 shots) before running full film.

## Current State Analysis

### Key Discoveries
- `encode.py:18-38` — `detect_shots()` runs PySceneDetect AdaptiveDetector, returns scene list
- `encode.py:41-54` — `export_keyframes()` saves 1 frame per shot (middle frame) via `save_images(num_images=1)`
- `encode.py:57-87` — `build_manifest()` creates JSON with `description: null` for each shot
- `output/star_wars_iv/manifest.json` — 2,070 shots already detected, keyframes already exported
- `output/star_wars_iv/keyframes/` — 2,070 single JPGs (full resolution, naming: `star_wars_iv-Scene-NNNN-01.jpg`)
- `media/star_wars_iv.mp4` — 632MB, 1080p, ~2h, subtitle stream at index 2
- `tools/verify_shots.py` — serves manifest + keyframes for visual inspection
- Python 3.11, ffmpeg 7.1.1, existing venv with scenedetect installed

## Desired End State

```
output/star_wars_iv/
├── manifest.json          # shot list with timestamps (stage 1 output)
├── keyframes/             # 4-8 frames per shot at 512px (stage 1 output)
│   ├── 0001-01.jpg
│   ├── 0001-02.jpg
│   ├── ...
├── subtitles.srt          # extracted subtitles (stage 2 intermediate)
├── camera_motion.json     # per-shot camera motion labels (stage 2 intermediate)
└── prompts.json           # final prompt manifest (stage 2 output)
```

CLI:
```bash
python encode.py stage1 media/star_wars_iv.mp4 -o output/star_wars_iv
python encode.py stage2 output/star_wars_iv --limit 20   # test on first 20 shots
python encode.py stage2 output/star_wars_iv               # full run
```

## What We're NOT Doing

- Screenplay scraping/alignment from IMSDb (v2)
- Audio classification / YAMNet mood detection (v2)
- Local model support / ollama fallback (v2)
- Provider abstraction beyond Gemini (v2)
- WhisperX transcription (subtitles already embedded)
- Fine-tuning or multi-pass description

---

## Phase 1: Pipeline CLI & Stage 1 Enhancement

### Overview
Refactor `encode.py` from a single script into a subcommand CLI. Enhance keyframe extraction to save multiple frames per shot at 512px resolution using ffmpeg (more control than PySceneDetect's `save_images`).

### Tasks

#### 1. Refactor encode.py to subcommand CLI
- [x] Rewrite `encode.py` with `argparse` subcommands: `stage1` and `stage2`

```python
#!/usr/bin/env python3
"""
lossy encoder: video → shot manifest → prompt manifest.

Usage:
    python encode.py stage1 media/star_wars_iv.mp4 -o output/star_wars_iv
    python encode.py stage2 output/star_wars_iv [--limit 20]
"""

import argparse
import json
import os
import subprocess
import sys
import time

from scenedetect import open_video, SceneManager, AdaptiveDetector, ContentDetector


def detect_shots(video_path: str, detector: str = "adaptive", threshold: float | None = None):
    """Detect shot boundaries in a video file."""
    video = open_video(video_path)

    manager = SceneManager()
    if detector == "adaptive":
        manager.add_detector(AdaptiveDetector(adaptive_threshold=threshold or 3.0))
    elif detector == "content":
        manager.add_detector(ContentDetector(threshold=threshold or 27.0))
    else:
        raise ValueError(f"Unknown detector: {detector}")

    print(f"Detecting shots with {detector} detector...")
    start = time.time()
    manager.detect_scenes(video, show_progress=True)
    elapsed = time.time() - start

    scenes = manager.get_scene_list()
    print(f"Found {len(scenes)} shots in {elapsed:.1f}s")

    return scenes


def frames_for_duration(duration_s: float) -> int:
    """Determine how many frames to extract based on shot duration."""
    if duration_s < 1.0:
        return 2
    elif duration_s < 3.0:
        return 4
    elif duration_s < 8.0:
        return 6
    else:
        return 8


def extract_keyframes(video_path: str, scenes: list, output_dir: str):
    """Extract multiple uniformly-sampled frames per shot at 512px using ffmpeg."""
    keyframes_dir = os.path.join(output_dir, "keyframes")
    os.makedirs(keyframes_dir, exist_ok=True)

    total = len(scenes)
    for i, (start, end) in enumerate(scenes):
        start_s = start.get_seconds()
        end_s = end.get_seconds()
        duration_s = end_s - start_s
        n_frames = frames_for_duration(duration_s)

        # Uniform sampling: evenly spaced timestamps within the shot
        if n_frames == 1:
            timestamps = [start_s + duration_s / 2]
        else:
            step = duration_s / (n_frames + 1)
            timestamps = [start_s + step * (j + 1) for j in range(n_frames)]

        for j, ts in enumerate(timestamps):
            out_path = os.path.join(keyframes_dir, f"{i + 1:04d}-{j + 1:02d}.jpg")
            if os.path.exists(out_path):
                continue
            subprocess.run(
                [
                    "ffmpeg", "-ss", f"{ts:.3f}", "-i", video_path,
                    "-vframes", "1",
                    "-vf", "scale='if(gt(iw,ih),512,-2)':'if(gt(ih,iw),512,-2)'",
                    "-q:v", "2", "-y", out_path,
                ],
                capture_output=True,
            )

        if (i + 1) % 100 == 0 or i + 1 == total:
            print(f"  Extracted frames for {i + 1}/{total} shots")


def build_manifest(scenes, video_path: str) -> dict:
    """Build a scene manifest JSON from detected shots."""
    manifest = {
        "source": {
            "file": os.path.basename(video_path),
            "path": video_path,
        },
        "shot_count": len(scenes),
        "scenes": [],
    }

    for i, (start, end) in enumerate(scenes):
        duration_s = (end - start).get_seconds()
        n_frames = frames_for_duration(duration_s)
        keyframe_files = [f"{i + 1:04d}-{j + 1:02d}.jpg" for j in range(n_frames)]

        scene_entry = {
            "index": i,
            "start_timecode": start.get_timecode(),
            "end_timecode": end.get_timecode(),
            "start_s": round(start.get_seconds(), 3),
            "end_s": round(end.get_seconds(), 3),
            "duration_s": round(duration_s, 3),
            "start_frame": start.get_frames(),
            "end_frame": end.get_frames(),
            "keyframes": keyframe_files,
        }
        manifest["scenes"].append(scene_entry)

    return manifest


def run_stage1(args):
    """Stage 1: Detect shots, extract keyframes, build manifest."""
    if not os.path.exists(args.video):
        print(f"Error: {args.video} not found")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    scenes = detect_shots(args.video, args.detector, args.threshold)
    if not scenes:
        print("No shots detected. Try lowering the threshold.")
        sys.exit(1)

    print(f"\nExtracting keyframes (512px, adaptive frame count)...")
    extract_keyframes(args.video, scenes, args.output)

    manifest = build_manifest(scenes, args.video)
    manifest_path = os.path.join(args.output, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest saved to {manifest_path}")
    print(f"  {manifest['shot_count']} shots detected")
    durations = [s["duration_s"] for s in manifest["scenes"]]
    print(f"  Average shot duration: {sum(durations) / len(durations):.2f}s")
    print(f"  Shortest: {min(durations):.2f}s / Longest: {max(durations):.2f}s")


def run_stage2(args):
    """Stage 2: Generate prompts from shots. (Implemented in Phase 2-3)"""
    print("Stage 2 not yet implemented.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="lossy encoder: video → prompts")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Stage 1
    s1 = subparsers.add_parser("stage1", help="Detect shots and extract keyframes")
    s1.add_argument("video", help="Path to video file")
    s1.add_argument("--output", "-o", default="output", help="Output directory")
    s1.add_argument("--detector", "-d", default="adaptive", choices=["adaptive", "content"])
    s1.add_argument("--threshold", "-t", type=float, default=None)
    s1.set_defaults(func=run_stage1)

    # Stage 2
    s2 = subparsers.add_parser("stage2", help="Generate prompts from detected shots")
    s2.add_argument("output_dir", help="Output directory from stage 1")
    s2.add_argument("--limit", type=int, default=None, help="Process only first N shots")
    s2.add_argument("--provider", default="gemini", choices=["gemini"], help="Vision API provider")
    s2.set_defaults(func=run_stage2)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
```

#### 2. Test stage 1 on Star Wars IV
- [x] Run: `python encode.py stage1 media/star_wars_iv.mp4 -o output/star_wars_iv_v2` — verify shot detection and multi-frame keyframe extraction
- [ ] Manual: Spot-check a few keyframe sets (e.g., `output/star_wars_iv_v2/keyframes/0005-*.jpg`) — confirm multiple frames per shot, 512px resolution, visually distinct frames within each shot

### Success Criteria
- [x] Run: `python encode.py stage1 --help` — shows stage1 subcommand help
- [x] Run: `python encode.py stage2 --help` — shows stage2 subcommand help
- [x] Run: `ls output/star_wars_iv_v2/keyframes/0010-*.jpg | wc -l` — returns 4-8 (multi-frame extraction working)
- [x] Run: `python3 -c "import json; m=json.load(open('output/star_wars_iv_v2/manifest.json')); print(len(m['scenes'][0]['keyframes']))"` — returns >1

---

## Phase 2: Subtitle & Optical Flow Preprocessing

### Overview
Extract subtitles from the video and detect camera motion via optical flow. Both produce per-shot metadata that gets included in the vision API prompt.

### Tasks

#### 1. Subtitle extraction and alignment
- [x] Add `extract_subtitles()` function to `encode.py`

```python
import re


def extract_subtitles(video_path: str, output_dir: str) -> str | None:
    """Extract subtitles from video using ffmpeg. Returns path to SRT file or None."""
    srt_path = os.path.join(output_dir, "subtitles.srt")
    if os.path.exists(srt_path):
        print(f"Subtitles already extracted: {srt_path}")
        return srt_path

    # Check if subtitle streams exist
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "s",
         "-show_entries", "stream=index", "-of", "csv=p=0", video_path],
        capture_output=True, text=True,
    )
    if not result.stdout.strip():
        print("No subtitle streams found in video.")
        return None

    # Extract first subtitle stream
    subprocess.run(
        ["ffmpeg", "-i", video_path, "-map", "0:s:0", "-y", srt_path],
        capture_output=True,
    )
    print(f"Subtitles extracted to {srt_path}")
    return srt_path


def parse_srt(srt_path: str) -> list[dict]:
    """Parse SRT file into list of {start_s, end_s, text} dicts."""
    with open(srt_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    entries = []
    # SRT blocks separated by blank lines
    blocks = re.split(r"\n\s*\n", content.strip())
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        # Line 2: timestamp line "00:01:23,456 --> 00:01:25,789"
        ts_match = re.match(
            r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})",
            lines[1],
        )
        if not ts_match:
            continue
        g = [int(x) for x in ts_match.groups()]
        start_s = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
        end_s = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
        text = " ".join(lines[2:]).strip()
        # Strip HTML tags sometimes found in SRT
        text = re.sub(r"<[^>]+>", "", text)
        entries.append({"start_s": start_s, "end_s": end_s, "text": text})

    return entries


def align_subtitles_to_shots(subtitles: list[dict], scenes: list[dict]) -> dict[int, list[str]]:
    """Map shot indices to overlapping subtitle text. Returns {shot_index: [dialogue lines]}."""
    dialogue_map = {}
    for scene in scenes:
        shot_start = scene["start_s"]
        shot_end = scene["end_s"]
        lines = []
        for sub in subtitles:
            # Check overlap: sub overlaps shot if sub.start < shot.end AND sub.end > shot.start
            if sub["start_s"] < shot_end and sub["end_s"] > shot_start:
                lines.append(sub["text"])
        if lines:
            dialogue_map[scene["index"]] = lines
    return dialogue_map
```

#### 2. Camera motion detection via optical flow
- [x] Add `detect_camera_motion()` function to `encode.py`

```python
import cv2
import numpy as np


def classify_motion(angles: np.ndarray, magnitudes: np.ndarray, threshold: float = 2.0) -> str:
    """Classify camera motion from optical flow statistics."""
    avg_mag = np.mean(magnitudes)
    if avg_mag < threshold:
        return "static"

    # Compute dominant direction from flow angles
    # Convert angles to unit vectors and average
    dx = np.mean(np.cos(angles) * magnitudes)
    dy = np.mean(np.sin(angles) * magnitudes)
    dominant_angle = np.degrees(np.arctan2(dy, dx)) % 360

    # Check for zoom: divergent/convergent flow
    # (simplified: high magnitude variance across frame regions suggests zoom)
    mag_std = np.std(magnitudes)
    if mag_std > avg_mag * 0.8:
        return "zoom"

    # Classify pan/tilt by dominant angle
    if 315 <= dominant_angle or dominant_angle < 45:
        return "pan right"
    elif 45 <= dominant_angle < 135:
        return "tilt down"
    elif 135 <= dominant_angle < 225:
        return "pan left"
    elif 225 <= dominant_angle < 315:
        return "tilt up"

    return "moving"


def detect_camera_motion(video_path: str, scenes: list[dict], output_dir: str) -> dict:
    """Detect camera motion for each shot using Farneback optical flow."""
    motion_path = os.path.join(output_dir, "camera_motion.json")
    if os.path.exists(motion_path):
        print(f"Camera motion already detected: {motion_path}")
        with open(motion_path) as f:
            return json.load(f)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    motion_labels = {}

    total = len(scenes)
    for i, scene in enumerate(scenes):
        start_frame = scene["start_frame"]
        end_frame = scene["end_frame"]
        n_frames = end_frame - start_frame

        # Sample every 5th frame, minimum 2 frames
        step = max(1, n_frames // 10)
        sample_frames = list(range(start_frame, end_frame, step))[:20]

        if len(sample_frames) < 2:
            motion_labels[str(i)] = "static"
            continue

        all_angles = []
        all_mags = []
        prev_gray = None

        for frame_idx in sample_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            # Downscale for speed
            small = cv2.resize(frame, (320, 180))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
                )
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                all_angles.append(ang.flatten())
                all_mags.append(mag.flatten())

            prev_gray = gray

        if all_angles:
            angles = np.concatenate(all_angles)
            mags = np.concatenate(all_mags)
            motion_labels[str(i)] = classify_motion(angles, mags)
        else:
            motion_labels[str(i)] = "static"

        if (i + 1) % 100 == 0 or i + 1 == total:
            print(f"  Camera motion: {i + 1}/{total} shots")

    cap.release()

    with open(motion_path, "w") as f:
        json.dump(motion_labels, f, indent=2)
    print(f"Camera motion saved to {motion_path}")

    return motion_labels
```

#### 3. Wire preprocessing into stage2 skeleton
- [x] Update `run_stage2()` to load manifest, run subtitle extraction, run optical flow, and save intermediate results

```python
def run_stage2(args):
    """Stage 2: Enrich shots with metadata, generate prompts via vision API."""
    output_dir = args.output_dir
    manifest_path = os.path.join(output_dir, "manifest.json")

    if not os.path.exists(manifest_path):
        print(f"Error: {manifest_path} not found. Run stage1 first.")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    video_path = manifest["source"]["path"]
    scenes = manifest["scenes"]

    if args.limit:
        scenes = scenes[:args.limit]
        print(f"Processing first {args.limit} shots (of {manifest['shot_count']})")

    # Step 1: Extract subtitles
    srt_path = extract_subtitles(video_path, output_dir)
    subtitles = parse_srt(srt_path) if srt_path else []
    dialogue_map = align_subtitles_to_shots(subtitles, scenes)
    print(f"  {len(dialogue_map)} shots have dialogue")

    # Step 2: Camera motion detection
    print("Detecting camera motion...")
    motion_labels = detect_camera_motion(video_path, scenes, output_dir)

    # Step 3: Generate prompts via vision API
    print("Generating prompts...")
    prompts = generate_prompts(scenes, dialogue_map, motion_labels, output_dir, args.provider)

    # Save prompt manifest
    prompts_path = os.path.join(output_dir, "prompts.json")
    with open(prompts_path, "w") as f:
        json.dump(prompts, f, indent=2)

    print(f"\nPrompts saved to {prompts_path}")
    print(f"  {len(prompts)} shots described")
```

#### 4. Test preprocessing on first 20 shots
- [ ] Run: `python encode.py stage2 output/star_wars_iv_v2 --limit 20` — verify subtitle extraction and camera motion detection complete (will fail at `generate_prompts` which is Phase 3)
- [ ] Manual: Check `output/star_wars_iv_v2/subtitles.srt` — confirm subtitle content looks correct
- [ ] Manual: Check `output/star_wars_iv_v2/camera_motion.json` — confirm motion labels are plausible (opening crawl should be "static" or "tilt up", space battle shots should show motion)

### Success Criteria
- [ ] Run: `python3 -c "import json; m=json.load(open('output/star_wars_iv_v2/camera_motion.json')); print(len(m))"` — returns shot count
- [ ] Run: `wc -l output/star_wars_iv_v2/subtitles.srt` — returns non-zero line count

---

## Phase 3: Gemini Vision API Integration

### Overview
Send extracted keyframes plus metadata context (dialogue, camera motion) to Gemini 2.5 Flash-Lite. Parse structured JSON responses into the prompt manifest.

### Tasks

#### 1. Install google-genai SDK
- [x] Run: `pip install google-genai`

#### 2. Add Gemini prompt generation
- [x] Add `generate_prompts()` function to `encode.py`

```python
import base64
from google import genai
from google.genai import types


SYSTEM_PROMPT = """You are a film analysis expert. Given frames from a single shot of a film, describe the shot for use as a video generation prompt.

Return a JSON object with these fields:
- "shot_type": one of "extreme wide", "wide", "medium wide", "medium", "medium close-up", "close-up", "extreme close-up", "insert"
- "camera_movement": description of camera motion (e.g., "static", "slow pan left", "tracking forward", "handheld")
- "subjects": who/what is in the shot and what they are doing
- "action": what happens during the shot (describe the motion/change from start to end)
- "lighting": description of lighting quality and direction
- "color_palette": dominant colors
- "mood": emotional tone or atmosphere
- "setting": location/environment description

Be specific and cinematic. Describe what changes between frames, not just what's visible in one frame. Output ONLY valid JSON, no markdown."""


def load_frame_as_part(frame_path: str) -> types.Part:
    """Load a JPEG frame as a Gemini API Part."""
    with open(frame_path, "rb") as f:
        data = f.read()
    return types.Part.from_bytes(data=data, mime_type="image/jpeg")


def generate_prompts(
    scenes: list[dict],
    dialogue_map: dict[int, list[str]],
    motion_labels: dict,
    output_dir: str,
    provider: str,
) -> list[dict]:
    """Generate descriptive prompts for each shot via vision API."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        print("Get one at https://aistudio.google.com/apikey")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    keyframes_dir = os.path.join(output_dir, "keyframes")

    # Load existing prompts for resume support
    prompts_path = os.path.join(output_dir, "prompts.json")
    if os.path.exists(prompts_path):
        with open(prompts_path) as f:
            existing = json.load(f)
        existing_indices = {p["index"] for p in existing}
    else:
        existing = []
        existing_indices = set()

    prompts = list(existing)
    total = len(scenes)
    errors = 0

    for scene in scenes:
        idx = scene["index"]
        if idx in existing_indices:
            continue

        # Load keyframe images
        frame_files = scene.get("keyframes", [])
        parts = []
        for fname in frame_files:
            fpath = os.path.join(keyframes_dir, fname)
            if os.path.exists(fpath):
                parts.append(load_frame_as_part(fpath))

        if not parts:
            print(f"  Shot {idx}: no keyframes found, skipping")
            continue

        # Build user prompt with metadata context
        context_lines = [f"Shot {idx + 1} of {total}. Duration: {scene['duration_s']:.1f}s."]

        camera = motion_labels.get(str(idx), "unknown")
        context_lines.append(f"Detected camera motion: {camera}.")

        dialogue = dialogue_map.get(idx, [])
        if dialogue:
            context_lines.append(f"Dialogue during this shot: \"{' / '.join(dialogue)}\"")

        context_lines.append(f"These are {len(parts)} uniformly-sampled frames from the shot, in chronological order.")
        context_lines.append("Analyze the frames and return the JSON description.")

        user_content = parts + [types.Part.from_text("\n".join(context_lines))]

        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-preview-05-20",
                contents=[types.Content(role="user", parts=user_content)],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.3,
                    response_mime_type="application/json",
                ),
            )

            # Parse JSON response
            text = response.text.strip()
            description = json.loads(text)

            prompt_entry = {
                "index": idx,
                "start_s": scene["start_s"],
                "end_s": scene["end_s"],
                "duration_s": scene["duration_s"],
                "camera_motion_detected": camera,
                "dialogue": dialogue if dialogue else None,
                "description": description,
            }
            prompts.append(prompt_entry)

            if (len(prompts) - len(existing)) % 10 == 0:
                print(f"  Described {len(prompts)}/{total} shots")
                # Incremental save
                with open(prompts_path, "w") as f:
                    json.dump(prompts, f, indent=2)

        except Exception as e:
            errors += 1
            print(f"  Shot {idx}: error - {e}")
            if errors > 5:
                print("Too many errors, saving progress and stopping.")
                break

    return prompts
```

#### 3. Test on first 20 shots
- [ ] Run: `GEMINI_API_KEY=... python encode.py stage2 output/star_wars_iv_v2 --limit 20` — should produce prompts for first 20 shots
- [ ] Manual: Review `output/star_wars_iv_v2/prompts.json` — check that descriptions are accurate and structured correctly. Verify:
  - Opening crawl shot has appropriate description (text crawl, static/tilt)
  - Space battle shots mention ships, movement
  - Dialogue shots include the subtitle text in context
  - Camera motion labels are roughly correct

### Success Criteria
- [ ] Run: `python3 -c "import json; p=json.load(open('output/star_wars_iv_v2/prompts.json')); print(len(p))"` — returns 20
- [ ] Run: `python3 -c "import json; p=json.load(open('output/star_wars_iv_v2/prompts.json')); print(list(p[0]['description'].keys()))"` — returns expected JSON keys
- [ ] Manual: User confirms description quality is acceptable for first 20 shots

---

## Phase 4: End-to-End Verification

### Overview
Run the full pipeline on Star Wars IV. Verify output quality and cost.

### Tasks

#### 1. Full stage 2 run
- [ ] Run: `GEMINI_API_KEY=... python encode.py stage2 output/star_wars_iv_v2` — process all ~2,070 shots
- [ ] Monitor API cost in Google AI Studio dashboard — should be under $5

#### 2. Quality spot-check
- [ ] Manual: Review 10 randomly sampled prompts from the full manifest — check accuracy of descriptions, camera motion, dialogue alignment
- [ ] Manual: Check edge cases:
  - Shot 0 (opening crawl, ~29s) — should describe scrolling text, space background
  - Very short shots (<1s) — should still have reasonable descriptions
  - Shots with no dialogue — `dialogue` field should be null
  - Final shots (credits) — should describe credits/end

#### 3. Summary statistics
- [ ] Run: `python3 -c "
import json
p = json.load(open('output/star_wars_iv_v2/prompts.json'))
print(f'Total prompts: {len(p)}')
with_dialogue = sum(1 for x in p if x.get('dialogue'))
print(f'With dialogue: {with_dialogue}')
motion_types = {}
for x in p:
    m = x.get('camera_motion_detected', 'unknown')
    motion_types[m] = motion_types.get(m, 0) + 1
for k, v in sorted(motion_types.items(), key=lambda x: -x[1]):
    print(f'  {k}: {v}')
"` — verify distribution looks reasonable

### Success Criteria
- [ ] All ~2,070 shots have prompts
- [ ] API cost was under $5
- [ ] Manual: User approves quality of sampled descriptions

---

## Final Checklist

- [ ] All phases complete
- [ ] Stage 1 and Stage 2 can be run independently
- [ ] `output/star_wars_iv_v2/prompts.json` contains structured descriptions for all shots
- [ ] Pipeline handles edge cases (long shots, short shots, no-dialogue shots)

## References

- Task: `docs/work/tasks/0001-screenplay-to-prompts-v1/task.md`
- Research: `docs/research/shot-to-prompt-landscape.md`
- ADR: `docs/design/adr/003-pyscenedetect-for-shot-detection.md`
