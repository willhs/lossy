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
import re
import subprocess
import sys
import time

import cv2
import numpy as np
from scenedetect import open_video, SceneManager, AdaptiveDetector, ContentDetector


# ---------------------------------------------------------------------------
# Stage 1: Shot detection & keyframe extraction
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Stage 2: Subtitle extraction
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Stage 2: Camera motion detection via optical flow
# ---------------------------------------------------------------------------

def classify_motion(angles: np.ndarray, magnitudes: np.ndarray, threshold: float = 2.0) -> str:
    """Classify camera motion from optical flow statistics."""
    avg_mag = np.mean(magnitudes)
    if avg_mag < threshold:
        return "static"

    # Compute dominant direction from flow angles
    dx = np.mean(np.cos(angles) * magnitudes)
    dy = np.mean(np.sin(angles) * magnitudes)
    dominant_angle = np.degrees(np.arctan2(dy, dx)) % 360

    # Check for zoom: divergent/convergent flow
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
    motion_labels = {}

    total = len(scenes)
    for i, scene in enumerate(scenes):
        start_frame = scene["start_frame"]
        end_frame = scene["end_frame"]
        n_frames = end_frame - start_frame

        # Sample every 10th of the shot's frames, cap at 20
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


# ---------------------------------------------------------------------------
# Audio: YAMNet classification and per-shot aggregation
# ---------------------------------------------------------------------------

# YAMNet hop between successive classification frames
YAMNET_HOP_S = 0.48
YAMNET_WINDOW_S = 0.96


def class_to_bucket(class_index: int) -> str:
    """Map a YAMNet class index (0-520) to one of 6 high-level audio buckets.

    Buckets: speech, music, effects, ambient, silence, other.
    Mapping follows AudioSet ontology groupings in yamnet_class_map.csv.
    """
    # Speech: human vocal sounds (0-23), body/crowd sounds (33-66)
    # Excludes singing (24-32) which goes to music
    if class_index <= 23 or 33 <= class_index <= 66:
        return "speech"
    # Music: singing (24-32), instruments and genres (132-276)
    if 24 <= class_index <= 32 or 132 <= class_index <= 276:
        return "music"
    # Ambient: natural environment — wind, water, thunder, fire (277-293)
    if 277 <= class_index <= 293:
        return "ambient"
    # Effects: vehicles, engines, doors, tools, weapons, impacts, liquids,
    # mechanical sounds, alarms, electronic tones (294-493)
    if 294 <= class_index <= 493:
        return "effects"
    # Silence
    if class_index == 494:
        return "silence"
    # Other: animals (67-131), signal processing artifacts (495-520)
    return "other"


def aggregate_shot_audio(
    scores: np.ndarray,
    class_names: list[str],
    shot_start_s: float,
    shot_end_s: float,
    top_n: int = 5,
) -> dict:
    """Aggregate YAMNet scores for a single shot into bucket + top labels.

    Args:
        scores: Full YAMNet output, shape [N, 521].
        class_names: List of 521 class display names.
        shot_start_s: Shot start time in seconds.
        shot_end_s: Shot end time in seconds.
        top_n: Number of top class labels to include.

    Returns:
        {"bucket": str, "labels": [str]}
    """
    # Find YAMNet frames overlapping this shot
    frame_indices = []
    for i in range(len(scores)):
        frame_start = i * YAMNET_HOP_S
        frame_end = frame_start + YAMNET_WINDOW_S
        if frame_start < shot_end_s and frame_end > shot_start_s:
            frame_indices.append(i)

    if not frame_indices:
        return {"bucket": "silence", "labels": []}

    # Mean scores across overlapping frames -> [521]
    mean_scores = scores[frame_indices].mean(axis=0)

    # Sum mean scores per bucket
    bucket_scores = {
        "speech": 0.0, "music": 0.0, "effects": 0.0,
        "ambient": 0.0, "silence": 0.0, "other": 0.0,
    }
    for idx, score in enumerate(mean_scores):
        bucket_scores[class_to_bucket(idx)] += float(score)

    # Dominant bucket
    bucket = max(bucket_scores, key=bucket_scores.get)

    # Top-N labels by mean score, skip near-zero
    top_indices = np.argsort(mean_scores)[::-1][:top_n]
    labels = [class_names[i] for i in top_indices if mean_scores[i] > 0.01]

    return {"bucket": bucket, "labels": labels}


def extract_audio(video_path: str, output_dir: str) -> str:
    """Extract audio track to mono 16kHz WAV. Returns path to WAV file."""
    wav_path = os.path.join(output_dir, "audio.wav")
    if os.path.exists(wav_path):
        print(f"Audio already extracted: {wav_path}")
        return wav_path

    print("Extracting audio track...")
    result = subprocess.run(
        [
            "ffmpeg", "-i", video_path,
            "-ac", "1", "-ar", "16000", "-vn",
            "-y", wav_path,
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Error extracting audio: {result.stderr}")
        sys.exit(1)

    print(f"Audio extracted to {wav_path}")
    return wav_path


def run_yamnet(audio_path: str, output_dir: str) -> tuple[np.ndarray, list[str]]:
    """Run YAMNet classification on audio file. Returns (scores, class_names).

    Scores shape: [N, 521] where N is the number of ~0.48s frames.
    Caches raw scores to yamnet_scores.npz for resume support.
    """
    import tensorflow_hub as hub
    import tensorflow as tf
    import csv

    print("Loading YAMNet model...")
    model = hub.load("https://tfhub.dev/google/yamnet/1")

    class_map_path = model.class_map_path().numpy()
    class_names = []
    with tf.io.gfile.GFile(class_map_path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_names.append(row["display_name"])

    # Check cache
    scores_path = os.path.join(output_dir, "yamnet_scores.npz")
    if os.path.exists(scores_path):
        print(f"YAMNet scores already computed: {scores_path}")
        data = np.load(scores_path)
        return data["scores"], class_names

    # Load audio
    from scipy.io import wavfile

    print("Running YAMNet classification...")
    sample_rate, wav_data = wavfile.read(audio_path)
    waveform = wav_data.astype(np.float32) / 32768.0

    start = time.time()
    scores, embeddings, spectrogram = model(waveform)
    scores = scores.numpy()
    elapsed = time.time() - start
    print(f"  Classified {len(scores)} frames in {elapsed:.1f}s")

    # Cache scores
    np.savez_compressed(scores_path, scores=scores)
    print(f"  Scores cached to {scores_path}")

    return scores, class_names


def detect_audio_labels(
    video_path: str, scenes: list[dict], output_dir: str,
) -> dict[str, dict]:
    """Classify audio for each shot using YAMNet. Returns {shot_index: {bucket, labels}}.

    Caches per-shot labels to audio_labels.json (same pattern as camera_motion.json).
    """
    labels_path = os.path.join(output_dir, "audio_labels.json")
    if os.path.exists(labels_path):
        print(f"Audio labels already detected: {labels_path}")
        with open(labels_path) as f:
            return json.load(f)

    # Extract audio and run YAMNet
    wav_path = extract_audio(video_path, output_dir)
    scores, class_names = run_yamnet(wav_path, output_dir)

    # Aggregate per shot
    audio_labels = {}
    total = len(scenes)
    for i, scene in enumerate(scenes):
        audio_labels[str(scene["index"])] = aggregate_shot_audio(
            scores, class_names, scene["start_s"], scene["end_s"],
        )
        if (i + 1) % 100 == 0 or i + 1 == total:
            print(f"  Audio labels: {i + 1}/{total} shots")

    with open(labels_path, "w") as f:
        json.dump(audio_labels, f, indent=2)
    print(f"Audio labels saved to {labels_path}")

    return audio_labels


# ---------------------------------------------------------------------------
# Stage 2: Gemini vision API prompt generation
# ---------------------------------------------------------------------------

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
- "sound": description of the soundtrack — what you'd expect to hear based on the visuals and the detected audio labels. Describe music style/mood, sound effects, ambient sounds, and atmosphere. If silence or near-silence, say so.

Be specific and cinematic. Describe what changes between frames, not just what's visible in one frame. Output ONLY valid JSON, no markdown."""


def generate_prompts(
    scenes: list[dict],
    dialogue_map: dict[int, list[str]],
    motion_labels: dict,
    audio_labels: dict,
    output_dir: str,
    provider: str,
) -> list[dict]:
    """Generate descriptive prompts for each shot via vision API."""
    from google import genai
    from google.genai import types

    # Load from .env if present
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path) as ef:
            for line in ef:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

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
                with open(fpath, "rb") as f:
                    data = f.read()
                parts.append(types.Part.from_bytes(data=data, mime_type="image/jpeg"))

        if not parts:
            print(f"  Shot {idx}: no keyframes found, skipping")
            continue

        # Build user prompt with metadata context
        context_lines = [f"Shot {idx + 1} of {total}. Duration: {scene['duration_s']:.1f}s."]

        camera = motion_labels.get(str(idx), "unknown")
        context_lines.append(f"Detected camera motion: {camera}.")

        audio = audio_labels.get(str(idx), {})
        if audio.get("labels"):
            audio_desc = f"{audio['bucket']} ({', '.join(audio['labels'])})"
            context_lines.append(f"Detected audio: {audio_desc}.")

        dialogue = dialogue_map.get(idx, [])
        if dialogue:
            context_lines.append(f"Dialogue during this shot: \"{' / '.join(dialogue)}\"")

        context_lines.append(f"These are {len(parts)} uniformly-sampled frames from the shot, in chronological order.")
        context_lines.append("Analyze the frames and return the JSON description.")

        user_content = parts + [types.Part.from_text(text="\n".join(context_lines))]

        try:
            response = client.models.generate_content(
                model="gemini-3.1-flash-lite-preview",
                contents=[types.Content(role="user", parts=user_content)],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.3,
                    response_mime_type="application/json",
                ),
            )

            # Parse JSON response
            text = response.text
            if not text:
                print(f"  Shot {idx}: empty response, skipping")
                continue
            text = text.strip()
            description = json.loads(text)

            prompt_entry = {
                "index": idx,
                "start_s": scene["start_s"],
                "end_s": scene["end_s"],
                "duration_s": scene["duration_s"],
                "camera_motion_detected": camera,
                "audio_detected": audio_labels.get(str(idx)),
                "dialogue": dialogue if dialogue else None,
                "description": description,
            }
            prompts.append(prompt_entry)

            new_count = len(prompts) - len(existing)
            if new_count % 10 == 0:
                print(f"  Described {len(prompts)}/{total} shots")
                # Incremental save
                with open(prompts_path, "w") as f:
                    json.dump(prompts, f, indent=2)

        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                # Rate limited — wait and retry
                print(f"  Rate limited at shot {idx}, waiting 30s...")
                time.sleep(30)
                try:
                    response = client.models.generate_content(
                        model="gemini-3.1-flash-lite-preview",
                        contents=[types.Content(role="user", parts=user_content)],
                        config=types.GenerateContentConfig(
                            system_instruction=SYSTEM_PROMPT,
                            temperature=0.3,
                            response_mime_type="application/json",
                        ),
                    )
                    text = response.text.strip()
                    description = json.loads(text)
                    prompt_entry = {
                        "index": idx,
                        "start_s": scene["start_s"],
                        "end_s": scene["end_s"],
                        "duration_s": scene["duration_s"],
                        "camera_motion_detected": camera,
                        "audio_detected": audio_labels.get(str(idx)),
                        "dialogue": dialogue if dialogue else None,
                        "description": description,
                    }
                    prompts.append(prompt_entry)
                except Exception as e2:
                    errors += 1
                    print(f"  Shot {idx}: retry failed - {e2}")
            else:
                errors += 1
                print(f"  Shot {idx}: error - {e}")

            if errors > 50:
                print("Too many errors, saving progress and stopping.")
                break

    return prompts


# ---------------------------------------------------------------------------
# CLI entry points
# ---------------------------------------------------------------------------

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

    # Step 3: Audio classification
    print("Classifying audio...")
    audio_labels = detect_audio_labels(video_path, scenes, output_dir)

    # Step 4: Generate prompts via vision API
    print("Generating prompts...")
    prompts = generate_prompts(scenes, dialogue_map, motion_labels, audio_labels, output_dir, args.provider)

    # Save prompt manifest
    prompts_path = os.path.join(output_dir, "prompts.json")
    with open(prompts_path, "w") as f:
        json.dump(prompts, f, indent=2)

    print(f"\nPrompts saved to {prompts_path}")
    print(f"  {len(prompts)} shots described")


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
