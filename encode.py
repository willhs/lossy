#!/usr/bin/env python3
"""
lossy encoder: video → shot manifest → prompt manifest.

Usage:
    python encode.py stage1 media/star_wars_iv.mp4 -o output/star_wars_iv
    python encode.py stage2 output/star_wars_iv [--limit 20]
    python encode.py stage3 output/star_wars_iv [--tmdb-id 11]
    python encode.py stage4 output/star_wars_iv  # speaker attribution -> speakers.json
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time

import cv2
import numpy as np
from scenedetect import open_video, SceneManager, AdaptiveDetector, ContentDetector

import manifest
from config import ENCODE_MODEL, load_env
from manifest import _run_ffmpeg

# Gemini Flash Lite pricing (per token)
GEMINI_INPUT_COST = 0.075 / 1_000_000   # $0.075 per 1M input tokens
GEMINI_OUTPUT_COST = 0.30 / 1_000_000   # $0.30 per 1M output tokens


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
            _run_ffmpeg(
                [
                    "ffmpeg", "-ss", f"{ts:.3f}", "-i", video_path,
                    "-vframes", "1",
                    "-vf", "scale='if(gt(iw,ih),512,-2)':'if(gt(ih,iw),512,-2)'",
                    "-q:v", "2", "-y", out_path,
                ],
                f"keyframe extraction for shot {i + 1} frame {j + 1}",
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
    _run_ffmpeg(
        ["ffmpeg", "-i", video_path, "-map", "0:s:0", "-y", srt_path],
        "subtitle extraction",
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


def align_subtitles_to_shots(subtitles: list[dict], scenes: list[dict]) -> dict[int, list[dict]]:
    """Map shot indices to overlapping subtitle entries with timing.

    Returns {shot_index: [{text, start_s, end_s}]} where start_s/end_s
    are offsets relative to the shot start time.
    """
    dialogue_map = {}
    for scene in scenes:
        shot_start = scene["start_s"]
        shot_end = scene["end_s"]
        entries = []
        for sub in subtitles:
            if sub["start_s"] < shot_end and sub["end_s"] > shot_start:
                # Clamp to shot boundaries, convert to shot-relative offset
                rel_start = max(0, sub["start_s"] - shot_start)
                rel_end = min(shot_end - shot_start, sub["end_s"] - shot_start)
                entries.append({
                    "text": sub["text"],
                    "start_s": round(rel_start, 3),
                    "end_s": round(rel_end, 3),
                })
        if entries:
            dialogue_map[scene["index"]] = entries
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


def extract_shot_audio(wav_path: str, start_s: float, end_s: float) -> bytes | None:
    """Extract a shot's audio segment as WAV bytes via ffmpeg stdout pipe."""
    result = subprocess.run(
        [
            "ffmpeg", "-ss", str(start_s), "-t", str(end_s - start_s),
            "-i", wav_path, "-f", "wav", "-ac", "1", "-ar", "16000", "pipe:1", "-y",
        ],
        capture_output=True,
    )
    return result.stdout if result.returncode == 0 and result.stdout else None


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

    # Work around macOS Python SSL certificate issue
    try:
        import certifi
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except ImportError:
        pass

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

# What a canonical character description has to be, in one place.
#
# These descriptions are the ONLY identity signal the video model gets, now
# that character names are stripped before the prompt is sent
# (prompt_format.strip_character_names). So every word has to be something a
# diffusion model can actually draw.
#
# The earlier wording asked for "distinctive facial features and bearing",
# and "bearing" is what invited the unrenderable half. Han came back as "a
# smuggler and pilot: cocky and cynical in manner, quick-moving and
# physically confident" -- roughly half the description describing personality
# and profession, which the model silently dropped, falling back on a generic
# handsome lead. Luke's "open, boyish face that reads clearly from eager
# determination to alarm" is an acting note, not a face.
#
# Naming the character was previously papering over this: the name pointed at
# a face the model had memorised, so a vague description did not matter. With
# names gone the description has to carry the whole load, and specificity has
# to come from geometry -- face shape, brow, nose, jaw, skin -- rather than
# from adjectives about temperament.
IDENTITY_DESCRIPTION_SPEC = (
    "a canonical IDENTITY description (~50-80 words) that a text-to-image "
    "model could draw from cold. Give concrete, visible geometry: age range, "
    "gender, skin tone, hair colour and cut, eye colour, build, face shape, "
    "brow, nose, jaw, mouth, and any distinctive marks, scars or asymmetries. "
    "Write only what is VISIBLE IN A STILL FRAME. Do NOT include personality, "
    "temperament, profession, role, mood, backstory or how the character "
    "moves or behaves -- a model cannot draw 'cocky', 'cynical', 'confident' "
    "or 'quick-moving', and those words crowd out the ones it can use. "
    "Prefer 'asymmetric half-smile, heavy brow, broad straight nose' over "
    "'sardonic and rugged'. "
    "Do NOT describe clothing or costume for human characters -- this one "
    "description is prepended to every shot they appear in, so a costume "
    "named here is wrong everywhere they wear something else. Exception: when "
    "the costume or shell IS the character and never changes (droids, masked "
    "or armoured figures, non-human creatures), describe it."
)


# Every Gemini call gets a deadline.
#
# The client was built with no http_options, so generate_content had no
# timeout and could block forever. Re-encoding the film hung on shot ~400 and
# sat there for 21 hours: process alive, 0% CPU, 5s of CPU consumed, nothing
# written, no error. Indistinguishable from "still working" to anything
# watching progress, and it burns wall-clock rather than money, so nothing
# else catches it either.
#
# 180s is generous for a describe call that normally takes a few seconds --
# it is a hang detector, not a latency budget. The loop already retries and
# tolerates a failed shot, so a timeout costs one shot, not the run.
GEMINI_TIMEOUT_MS = 180_000


def _gemini_client(api_key: str):
    """Gemini client with a request deadline. Use instead of genai.Client."""
    from google import genai
    from google.genai import types

    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_MS),
    )


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
- "sound": description of the non-speech soundtrack — music, sound effects, ambient sounds, and atmosphere only. Do NOT include dialogue, voices, or speech — those are handled by a separate system. If silence or near-silence, say so.

Be specific and cinematic. Describe what changes between frames, not just what's visible in one frame.

You may also be given the previous shot's setting/lighting/color continuity as context. Use it only to keep the location and visual style consistent across the cut. It deliberately excludes character identity — do not infer, reuse, or guess who a person is from it. Identify who/what is in THIS shot from these frames alone; if you don't recognize someone, describe them by visible appearance rather than guessing a name.

Output ONLY valid JSON, no markdown."""

MUSIC_SYSTEM_PROMPT = """You are a film analysis expert. Given frames and audio from a single shot of a film, describe the shot for use as a video generation prompt.

Return a JSON object with these fields:
- "shot_type": one of "extreme wide", "wide", "medium wide", "medium", "medium close-up", "close-up", "extreme close-up", "insert"
- "camera_movement": description of camera motion (e.g., "static", "slow pan left", "tracking forward", "handheld")
- "subjects": who/what is in the shot and what they are doing
- "action": what happens during the shot (describe the motion/change from start to end)
- "lighting": description of lighting quality and direction
- "color_palette": dominant colors
- "mood": emotional tone or atmosphere
- "setting": location/environment description
- "sound": description of the non-speech soundtrack — music, sound effects, ambient sounds, and atmosphere only. Do NOT include dialogue, voices, or speech — those are handled by a separate system. If silence or near-silence, say so.
- "music": description of the music/score heard in this shot — mood, estimated tempo (e.g., ~80bpm), instrumentation, and how the music functions in the scene (e.g., "builds tension", "underscores triumph"). Describe only the musical elements; exclude SFX and ambient sounds. Listen carefully to the audio provided.

Be specific and cinematic. Describe what changes between frames, not just what's visible in one frame.

You may also be given the previous shot's setting/lighting/color continuity as context. Use it only to keep the location and visual style consistent across the cut. It deliberately excludes character identity — do not infer, reuse, or guess who a person is from it. Identify who/what is in THIS shot from these frames alone; if you don't recognize someone, describe them by visible appearance rather than guessing a name.

Output ONLY valid JSON, no markdown."""

SEGMENT_THRESHOLD_S = 8.0   # shots longer than this get temporal_segments generated
SEGMENT_DURATION_S = 12.0  # one segment per this many seconds of shot duration

SEGMENT_SYSTEM_PROMPT = """You are a film analysis expert. Given frames from a specific temporal portion of a longer film shot, describe what happens in THESE frames for use as a video generation prompt.

The full shot description is provided as context. Your task: describe this specific portion as if it were its own shot, using the same JSON schema. Focus on what specifically happens or changes in these frames.

Return a JSON object with these fields: shot_type, camera_movement, subjects, action, lighting, color_palette, mood, setting, sound.

Output ONLY valid JSON, no markdown."""


def generate_temporal_segments(
    scene: dict,
    main_description: dict,
    frame_files: list[str],
    keyframes_dir: str,
    client,
    n_segments: int | None = None,
) -> list[dict]:
    """Generate N segment descriptions for a long shot by dividing keyframes evenly.

    Each segment is described using its keyframes plus the whole-shot description
    as context, so Gemini can focus on temporal specifics while staying coherent.

    n_segments defaults to ceil(duration_s / SEGMENT_DURATION_S), capped at keyframe count.

    Returns a list of n_segments description dicts (same JSON schema as main description),
    or [] on failure or too few keyframes.
    """
    from google.genai import types

    n = len(frame_files)
    if n < 2:
        return []

    if n_segments is None:
        n_segments = max(2, math.ceil(scene["duration_s"] / SEGMENT_DURATION_S))
    n_segments = min(n_segments, n)

    # Split frames into n_segments groups as evenly as possible
    groups = [frame_files[i * n // n_segments : (i + 1) * n // n_segments] for i in range(n_segments)]

    descriptions = []
    for seg_idx, files in enumerate(groups):
        parts = []
        for fname in files:
            fpath = os.path.join(keyframes_dir, fname)
            if os.path.exists(fpath):
                with open(fpath, "rb") as f:
                    data = f.read()
                parts.append(types.Part.from_bytes(data=data, mime_type="image/jpeg"))

        if not parts:
            return []

        label = f"segment {seg_idx + 1} of {n_segments}"
        context = (
            f"Full shot duration: {scene['duration_s']:.1f}s. "
            f"You are describing {label} ({len(files)} of {n} keyframes).\n"
            f"Full shot description for context: {json.dumps(main_description)}\n"
            f"Describe what specifically happens in these {len(parts)} frames."
        )
        user_content = parts + [types.Part.from_text(text=context)]

        try:
            response = client.models.generate_content(
                model=ENCODE_MODEL,
                contents=[types.Content(role="user", parts=user_content)],
                config=types.GenerateContentConfig(
                    system_instruction=SEGMENT_SYSTEM_PROMPT,
                    temperature=0.3,
                    response_mime_type="application/json",
                ),
            )
            text = response.text
            if not text:
                return []
            descriptions.append(json.loads(text.strip()))
        except Exception as e:
            print(f"  Segment description failed ({label}): {e}")
            return []

    return descriptions if len(descriptions) == n_segments else []


# Fields safe to carry forward as continuity context. Deliberately excludes
# "subjects"/"action" (and "mood"/"sound"/"music") — those are where Gemini's
# world-knowledge bias hallucinates character identity, and once a wrong name
# rides forward it gets echoed back as "consistent" by the next shot too.
CONTINUITY_CONTEXT_FIELDS = ("setting", "lighting", "color_palette")


def _continuity_context(description: dict | None) -> dict:
    """Extract only the non-identity continuity fields from a shot description."""
    if not description:
        return {}
    return {k: description[k] for k in CONTINUITY_CONTEXT_FIELDS if description.get(k)}


def _retry_without_poisoned_frame(client, types, user_content, config, idx):
    """Retry a blocked describe call, dropping one keyframe at a time.

    A single frame can poison a whole shot. Shot 451 of Star Wars IV -- the
    Tusken Raider standing over Luke -- came back empty on every encode the
    project has ever run, and its absence is the cause of the known "index
    diverges from list position from 451 onward" defect. The block is
    `BlockedReason.OTHER`, which relaxing safety_settings does NOT lift, so it
    read as permanently un-describable.

    It is not. Only the first of its four keyframes trips the filter; frames
    2, 3 and 4 each describe fine on their own. Dropping the offending frame
    keeps the shot, at the cost of describing it from slightly less coverage
    -- which is strictly better than losing it and silently renumbering every
    shot after it.

    Tries the largest subsets first (drop exactly one frame), so the retained
    description is built from as much of the shot as possible.
    """
    images = [p for p in user_content if getattr(p, "inline_data", None) is not None]
    if len(images) < 2:
        return None

    for dropped, image in enumerate(images):
        attempt = [p for p in user_content if p is not image]
        try:
            response = client.models.generate_content(
                model=ENCODE_MODEL,
                contents=[types.Content(role="user", parts=attempt)],
                config=config,
            )
        except Exception:
            continue
        if response.text:
            print(f"  Shot {idx}: recovered by dropping keyframe {dropped + 1} "
                  f"of {len(images)}")
            return response
    return None


def _describe_shot(client, types, idx, scene, camera, audio_labels, dialogue,
                   frame_files, keyframes_dir, system_prompt, user_content,
                   encode_costs) -> dict | None:
    """Call Gemini for one shot, track cost, and build its shots.json entry.

    Returns ``None`` if Gemini returned an empty response. Raises on API
    errors so the caller can decide whether to retry (e.g. on a 429).
    """
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=0.3,
        response_mime_type="application/json",
    )
    response = client.models.generate_content(
        model=ENCODE_MODEL,
        contents=[types.Content(role="user", parts=user_content)],
        config=config,
    )
    text = response.text
    if not text:
        response = _retry_without_poisoned_frame(
            client, types, user_content, config, idx)
        text = response.text if response is not None else None
    if not text:
        return None
    description = json.loads(text.strip())

    usage = getattr(response, "usage_metadata", None)
    if usage:
        in_tok = getattr(usage, "prompt_token_count", 0) or 0
        out_tok = getattr(usage, "candidates_token_count", 0) or 0
        shot_cost = in_tok * GEMINI_INPUT_COST + out_tok * GEMINI_OUTPUT_COST
        encode_costs["total_input_tokens"] += in_tok
        encode_costs["total_output_tokens"] += out_tok
        encode_costs["cost_estimate"] += shot_cost
        encode_costs["per_shot"].append(
            {"index": idx, "input_tokens": in_tok, "output_tokens": out_tok, "cost": round(shot_cost, 6)})

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

    # Generate temporal segments for long shots so decode can vary prompts per split part
    if scene["duration_s"] >= SEGMENT_THRESHOLD_S:
        segs = generate_temporal_segments(scene, description, frame_files, keyframes_dir, client)
        if segs:
            prompt_entry["temporal_segments"] = segs

    return prompt_entry


def generate_prompts(
    scenes: list[dict],
    dialogue_map: dict[int, list[str]],
    motion_labels: dict,
    audio_labels: dict,
    output_dir: str,
    provider: str,
    dialog: list[dict] | None = None,
    wav_path: str | None = None,
) -> list[dict]:
    """Generate descriptive prompts for each shot via vision API."""
    from google import genai
    from google.genai import types

    dialog = dialog or []

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        print("Get one at https://aistudio.google.com/apikey")
        sys.exit(1)

    client = _gemini_client(api_key)
    keyframes_dir = os.path.join(output_dir, "keyframes")

    # Load existing shots for resume support (v2 shots.json, per the manifest contract)
    try:
        existing, _ = manifest.load_shots(output_dir)
    except FileNotFoundError:
        existing = []
    existing_indices = {p["index"] for p in existing}
    existing_by_index = {p["index"]: p for p in existing}

    prompts = list(existing)
    total = len(scenes)
    errors = 0
    prev_description = None

    # Cost tracking
    encode_costs = {"model": ENCODE_MODEL, "per_shot": [],
                    "total_input_tokens": 0, "total_output_tokens": 0, "cost_estimate": 0.0}
    costs_path = os.path.join(output_dir, "encode_costs.json")

    for scene in scenes:
        idx = scene["index"]
        if idx in existing_indices:
            prev_description = existing_by_index[idx].get("description")
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
            texts = [d["text"] if isinstance(d, dict) else d for d in dialogue]
            context_lines.append(f"Dialogue during this shot: \"{' / '.join(texts)}\"")

        is_music_shot = audio.get("bucket") == "music"

        # For music shots, prepend raw audio so Gemini can describe the score
        audio_parts = []
        if is_music_shot and wav_path and os.path.exists(wav_path):
            shot_audio = extract_shot_audio(wav_path, scene["start_s"], scene["end_s"])
            if shot_audio:
                audio_parts = [types.Part.from_bytes(data=shot_audio, mime_type="audio/wav")]

        context_lines.append(f"These are {len(parts)} uniformly-sampled frames from the shot, in chronological order.")
        if is_music_shot and audio_parts:
            context_lines.append("The audio for this shot is also provided. Use it to fill the 'music' field.")
        prev_continuity = _continuity_context(prev_description)
        if prev_continuity:
            context_lines.append(f"Previous shot's setting/lighting continuity (context only): {json.dumps(prev_continuity)}")
        context_lines.append("Analyze the frames and return the JSON description.")

        user_content = audio_parts + parts + [types.Part.from_text(text="\n".join(context_lines))]
        system_prompt = MUSIC_SYSTEM_PROMPT if is_music_shot else SYSTEM_PROMPT

        try:
            prompt_entry = _describe_shot(
                client, types, idx, scene, camera, audio_labels, dialogue,
                frame_files, keyframes_dir, system_prompt, user_content, encode_costs)
            if prompt_entry is None:
                print(f"  Shot {idx}: empty response, skipping")
                continue
            prompts.append(prompt_entry)
            prev_description = prompt_entry["description"]

            new_count = len(prompts) - len(existing)
            if new_count % 10 == 0:
                print(f"  Described {len(prompts)}/{total} shots")
                # Incremental save, through the same v2 contract as the final save
                manifest.save_shots(output_dir, prompts, dialog)

        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                # Rate limited — wait and retry
                print(f"  Rate limited at shot {idx}, waiting 30s...")
                time.sleep(30)
                try:
                    prompt_entry = _describe_shot(
                        client, types, idx, scene, camera, audio_labels, dialogue,
                        frame_files, keyframes_dir, system_prompt, user_content, encode_costs)
                    if prompt_entry is None:
                        errors += 1
                        print(f"  Shot {idx}: empty response on retry, skipping")
                    else:
                        prompts.append(prompt_entry)
                        prev_description = prompt_entry["description"]
                except Exception as e2:
                    errors += 1
                    print(f"  Shot {idx}: retry failed - {e2}")
            else:
                errors += 1
                print(f"  Shot {idx}: error - {e}")

            if errors > 50:
                print("Too many errors, saving progress and stopping.")
                break

    # Write encode cost data
    encode_costs["cost_estimate"] = round(encode_costs["cost_estimate"], 6)
    with open(costs_path, "w") as f:
        json.dump(encode_costs, f, indent=2)
    if encode_costs["per_shot"]:
        print(f"  Encode cost: ${encode_costs['cost_estimate']:.4f} "
              f"({encode_costs['total_input_tokens']} in / {encode_costs['total_output_tokens']} out tokens)")

    return prompts


# ---------------------------------------------------------------------------
# Stage 3: Character registry
# ---------------------------------------------------------------------------


def _normalize_character_name(name):
    """Strip TMDB annotations like '(voice)', '(uncredited)', and ' / alternate'."""
    name = re.sub(r'\s*\([^)]*\)', '', name).strip()
    return name.split(' / ')[0].strip()


def _to_name_key(display_name):
    """Convert a character display name to a lowercase_underscore identifier.

    This is the single source of truth for character identity. Anything that
    stores or looks up a character by id must derive it from display_name here,
    never trust an id supplied by the model.
    """
    normalized = _normalize_character_name(display_name)
    return re.sub(r'[^a-z0-9]+', '_', normalized.lower()).strip('_')


def _character_dedupe_key(name):
    """Punctuation-insensitive identity key, for matching ids that disagree only
    on separators.

    `_to_name_key` turns "C-3PO" into `c_3po`, but the stage-3 model is asked for
    its own short id and freely returns `c3po` for the same character. Comparing
    those as exact strings created two registry entries for one character, each
    with its own locked description and a disjoint half of the shots — so decode
    rendered the droid two ways and voice casting gave it two voices. Collapsing
    every non-alphanumeric makes both forms compare equal.
    """
    return re.sub(r'[^a-z0-9]', '', (name or '').lower())


def _collapse_duplicate_characters(chars_list):
    """Merge registry entries that describe the same character.

    Two entries collide when their display names match ignoring punctuation
    ("C-3PO" vs "C 3PO") or when one entry's id matches another's. On collision:
    keep the canonical id derived from display_name, union the shot lists, and
    keep the longest description. Order is preserved so output stays stable.
    """
    merged = {}
    for char in chars_list:
        display = char.get("display_name") or char.get("name", "")
        key = _character_dedupe_key(display) or _character_dedupe_key(char.get("name", ""))
        if not key:
            continue
        if key not in merged:
            entry = dict(char)
            entry["name"] = _to_name_key(display) or char.get("name", "")
            entry["shots"] = sorted(set(char.get("shots", [])))
            merged[key] = entry
            continue
        existing = merged[key]
        existing["shots"] = sorted(set(existing["shots"]) | set(char.get("shots", [])))
        if len(char.get("description", "")) > len(existing.get("description", "")):
            existing["description"] = char["description"]
    return list(merged.values())


def fetch_tmdb_cast(tmdb_id, api_key, media_type="movie"):
    """Fetch top 30 cast members from TMDB API.

    Returns list of dicts: {display_name, actor, name_key}.
    Raises urllib.error.URLError or ValueError on failure.
    """
    import urllib.request
    url = (
        f"https://api.themoviedb.org/3/{media_type}/{tmdb_id}/credits"
        f"?api_key={api_key}&language=en-US"
    )
    with urllib.request.urlopen(url, timeout=10) as resp:
        data = json.loads(resp.read())
    cast = data.get("cast", [])[:30]
    result = []
    for c in cast:
        raw = c.get("character", "")
        if not raw:
            continue
        display = _normalize_character_name(raw)
        if display:
            result.append({
                "display_name": display,
                "actor": c.get("name", ""),
                "name_key": _to_name_key(display),
            })
    return result


def _text_match_cast(cast_entries, subjects_by_shot):
    """Word-boundary match cast character names against shot subject strings.

    Returns:
        pre_assignments: dict mapping shot_idx -> list of matching cast_entry dicts
        unmatched: list of shot strings where no cast member was found
    """
    patterns = [
        (entry, re.compile(r'\b' + re.escape(entry['display_name']) + r'\b', re.IGNORECASE))
        for entry in cast_entries
    ]
    pre_assignments = {}
    unmatched = []
    for shot_text in subjects_by_shot:
        m = re.match(r'Shot (\d+):', shot_text)
        if not m:
            unmatched.append(shot_text)
            continue
        shot_idx = int(m.group(1))
        matched = [entry for entry, pat in patterns if pat.search(shot_text)]
        if matched:
            pre_assignments[shot_idx] = matched
        else:
            unmatched.append(shot_text)
    return pre_assignments, unmatched


def _make_supervised_system_prompt(cast_entries):
    """Build a Gemini system prompt that includes the TMDB cast list."""
    cast_lines = "\n".join(
        f"- {e['display_name']} (played by {e['actor']})"
        for e in cast_entries
    )
    return (
        "You are a film analysis expert. This film has the following confirmed cast:\n\n"
        f"{cast_lines}\n\n"
        "Given subject descriptions from shots where cast members weren't explicitly named, "
        "identify which cast member appears in each shot.\n\n"
        "Return a JSON object with a \"characters\" array. Each entry has:\n"
        "- \"name\": lowercase identifier matching the cast list (e.g., \"luke_skywalker\") — underscores, no spaces\n"
        "- \"display_name\": character's name as listed in credits\n"
        f"- \"description\": {IDENTITY_DESCRIPTION_SPEC}\n"
        "- \"shots\": list of shot indices (integers) where this character appears\n\n"
        "Rules:\n"
        "- Only match subjects to the provided cast list — do not invent unlisted characters\n"
        "- Merge all descriptions of the same character across shots\n"
        "- Include characters found in even 1 shot (pre-text-matched shots will supplement the count)\n"
        "- Limit to the 15 most prominent cast members found\n\n"
        "Output ONLY valid JSON."
    )


def _run_supervised_stage3(client, cast_entries, subjects_by_shot, prompts):
    """TMDB-seeded two-step character discovery: text match then supervised Gemini."""
    from google.genai import types

    BATCH_SIZE = 200

    # Step 1: text match
    pre_assignments, unmatched = _text_match_cast(cast_entries, subjects_by_shot)
    print(f"  Text-matched: {len(pre_assignments)} shots")
    print(f"  Sending to Gemini: {len(unmatched)} unmatched shots")

    # Step 2: supervised Gemini in batches on unmatched shots
    system_prompt = _make_supervised_system_prompt(cast_entries)
    all_chars = {}  # name_key -> character dict

    for batch_start in range(0, len(unmatched), BATCH_SIZE):
        batch = unmatched[batch_start:batch_start + BATCH_SIZE]
        user_text = (
            f"Subject descriptions from {len(batch)} shots:\n\n" + "\n".join(batch)
        )
        response = client.models.generate_content(
            model=ENCODE_MODEL,
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=user_text)])],
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.3,
                response_mime_type="application/json",
            ),
        )
        try:
            batch_data = json.loads(response.text.strip())
        except (json.JSONDecodeError, AttributeError):
            batch_num = batch_start // BATCH_SIZE + 1
            print(f"  Warning: batch {batch_num} response not valid JSON, skipping")
            continue

        for char in batch_data.get("characters", []):
            # Identity comes from display_name, not the model's own id — the id
            # is unstable across batches ("c3po" one batch, "c_3po" the next).
            display = char.get("display_name") or char.get("name", "")
            key = _character_dedupe_key(display) or _character_dedupe_key(char.get("name", ""))
            if not key:
                continue
            if key not in all_chars:
                all_chars[key] = {
                    **char,
                    "name": _to_name_key(display) or char.get("name", ""),
                    "shots": list(char.get("shots", [])),
                }
            else:
                existing = all_chars[key]
                existing["shots"] = sorted(set(existing["shots"]) | set(char.get("shots", [])))
                if len(char.get("description", "")) > len(existing.get("description", "")):
                    existing["description"] = char["description"]

    # Step 3: merge pre-assigned shots into character list
    chars_list = list(all_chars.values())
    for shot_idx, cast_matches in pre_assignments.items():
        for entry in cast_matches:
            key = entry["name_key"]
            # Match on the dedupe key: the TMDB path derives `c_3po` from the
            # credited name while the model may have registered `c3po` for the
            # same character. An exact-string compare here silently appended a
            # duplicate stub instead of merging.
            dkey = _character_dedupe_key(entry["display_name"]) or _character_dedupe_key(key)
            found = next(
                (c for c in chars_list
                 if _character_dedupe_key(c.get("display_name") or c["name"]) == dkey
                 or _character_dedupe_key(c["name"]) == dkey),
                None,
            )
            if found:
                if shot_idx not in found["shots"]:
                    found["shots"].append(shot_idx)
                    found["shots"].sort()
            else:
                # Character seen only via text match — add stub, fill description below
                chars_list.append({
                    "name": key,
                    "display_name": entry["display_name"],
                    "description": "",
                    "shots": [shot_idx],
                })

    # Step 4: collapse any remaining same-character duplicates, then filter to
    # characters appearing in >= 2 shots.
    #
    # The merges above should prevent duplicates, but this is the last gate
    # before the registry becomes the locked identity source for decode and
    # voice casting, so enforce uniqueness here rather than trusting upstream.
    # Union the shots and keep the longest description, matching step 2's rule.
    chars_list = _collapse_duplicate_characters(chars_list)
    chars_list = [c for c in chars_list if len(c["shots"]) >= 2]

    # Step 5: generate descriptions for text-match-only stubs
    stubs = [c for c in chars_list if not c.get("description")]
    if stubs:
        subjects_map = {
            e["index"]: e.get("description", {}).get("subjects", "")
            for e in prompts
        }
        for stub in stubs:
            shot_subjects = [
                f"Shot {s}: {subjects_map[s]}"
                for s in stub["shots"]
                if subjects_map.get(s)
            ]
            if not shot_subjects:
                continue
            desc_prompt = (
                f"Character: {stub['display_name']}\n"
                f"Appears in these shots:\n" + "\n".join(shot_subjects[:20]) + "\n\n"
                f"Write {IDENTITY_DESCRIPTION_SPEC}"
            )
            resp = client.models.generate_content(
                model=ENCODE_MODEL,
                contents=[types.Content(role="user", parts=[types.Part.from_text(text=desc_prompt)])],
                config=types.GenerateContentConfig(temperature=0.3),
            )
            stub["description"] = resp.text.strip()

    return {"characters": chars_list}


STAGE3_SYSTEM_PROMPT = """You are a film analysis expert. Given a list of subject descriptions from every shot of a film, identify the distinct named characters and create a canonical appearance description for each.

Return a JSON object with a "characters" array. Each character entry has:
- "name": a short identifier (e.g., "luke", "han_solo", "vader") — lowercase, underscores, no spaces
- "display_name": the character's name as it would appear in credits (e.g., "Luke Skywalker")
- "description": __IDENTITY_SPEC__
- "shots": list of shot indices (integers) where this character appears

Rules:
- Merge different descriptions of the same character across shots (e.g., "a young man with blond hair" and "Luke, wearing a white tunic" are the same person)
- Only include characters who appear in at least 2 shots
- Limit to the 5 most prominent characters (by shot count)
- If a character cannot be identified by name, use a descriptive identifier (e.g., "tall_officer", "bartender")

Output ONLY valid JSON."""

# One definition of what a description must be, shared by all three stage-3
# prompts so they cannot drift apart.
STAGE3_SYSTEM_PROMPT = STAGE3_SYSTEM_PROMPT.replace(
    "__IDENTITY_SPEC__", IDENTITY_DESCRIPTION_SPEC)


REFINE_SYSTEM_PROMPT = """You are a film analysis expert. You are given a character registry and a list of unassigned shots (shots not yet linked to any character). For each unassigned shot, determine if any of the registered characters appear in it based on the subject description.

Return a JSON object with an "assignments" array. Each entry has:
- "shot": the shot index (integer)
- "characters": list of character names (from the registry) that appear in this shot

Rules:
- Only assign characters from the provided registry — do not invent new ones.
- Match based on appearance, role, and context clues (e.g., "a young man in desert robes" is likely the same as a registered character described as a young man with blond hair in desert clothing).
- If no registered character matches, omit that shot from the output.
- Be generous with matching — it's better to include a plausible match than to miss one.

Output ONLY valid JSON."""


# Fields a human curates by hand, which stage 3 must not blow away when it
# regenerates the registry.
CURATED_CHARACTER_FIELDS = ("keep_name",)


def _preserve_curated_fields(characters_path: str, characters_data: dict) -> None:
    """Carry hand-set fields from the existing registry into the new one.

    Stage 3 rewrites characters.json wholesale, so anything a human set by
    hand vanishes on the next run. `keep_name` -- which marks the characters
    whose name is their design rather than a person, and so must survive the
    prompt-time name strip -- was lost exactly this way the first time the
    registry was regenerated. Silently, and the damage only shows up as
    stormtroopers rendering as generic soldiers several dollars later.
    """
    if not os.path.exists(characters_path):
        return
    try:
        with open(characters_path) as f:
            existing = json.load(f)
    except (OSError, json.JSONDecodeError):
        return

    curated = {
        c["name"]: {k: c[k] for k in CURATED_CHARACTER_FIELDS if k in c}
        for c in existing.get("characters", []) if c.get("name")
    }
    kept = 0
    for char in characters_data.get("characters", []):
        fields = curated.get(char.get("name"))
        if fields:
            char.update(fields)
            kept += 1
    if kept:
        print(f"  Preserved hand-set fields on {kept} character(s)")


def _refine_shot_assignments(client, characters_data, prompts, subjects_by_shot):
    """Second pass: assign unassigned shots to existing characters."""
    from google.genai import types

    characters = characters_data.get("characters", [])
    if not characters:
        return characters_data

    assigned_shots = set()
    for char in characters:
        assigned_shots.update(char.get("shots", []))

    unassigned = []
    for entry in prompts:
        idx = entry["index"]
        subjects = entry.get("description", {}).get("subjects", "")
        if subjects and idx not in assigned_shots:
            unassigned.append(f"Shot {idx}: {subjects}")

    if not unassigned:
        print("  All shots with subjects are assigned.")
        return characters_data

    registry_text = "\n".join(
        f"- {c['name']} ({c['display_name']}): {c['description']}"
        for c in characters
    )
    user_text = (
        f"Character registry:\n{registry_text}\n\n"
        f"Unassigned shots ({len(unassigned)}):\n" + "\n".join(unassigned)
    )

    print(f"  Refining: {len(unassigned)} unassigned shots...")
    response = client.models.generate_content(
        model=ENCODE_MODEL,
        contents=[types.Content(role="user", parts=[types.Part.from_text(text=user_text)])],
        config=types.GenerateContentConfig(
            system_instruction=REFINE_SYSTEM_PROMPT,
            temperature=0.3,
            response_mime_type="application/json",
            max_output_tokens=65536,
        ),
    )

    try:
        refinement = json.loads(response.text.strip())
    except (json.JSONDecodeError, AttributeError):
        print("  Warning: refinement response was not valid JSON, skipping.")
        return characters_data

    # Build name -> character index lookup
    char_idx = {c["name"]: i for i, c in enumerate(characters)}
    new_assignments = 0

    for assignment in refinement.get("assignments", []):
        shot_idx = assignment.get("shot")
        for char_name in assignment.get("characters", []):
            if char_name in char_idx:
                ci = char_idx[char_name]
                if shot_idx not in characters[ci]["shots"]:
                    characters[ci]["shots"].append(shot_idx)
                    new_assignments += 1

    if new_assignments:
        # Sort shot lists
        for char in characters:
            char["shots"].sort()
        print(f"  Refinement added {new_assignments} shot assignments:")
        for char in characters:
            print(f"    {char['display_name']}: {len(char['shots'])} shots")

    return characters_data


def run_stage3(args):
    """Stage 3: Build character registry from shots.json subjects."""
    from google import genai
    from google.genai import types

    output_dir = args.output_dir
    shots_path = manifest.shots_path(output_dir)

    if not os.path.exists(shots_path):
        print(f"Error: {shots_path} not found. Run stage2 first.")
        sys.exit(1)

    with open(shots_path) as f:
        raw = json.load(f)

    # Support both v2 format ({"format": "v2", "shots": [...]}) and v1 (flat array)
    prompts = raw["shots"] if isinstance(raw, dict) and raw.get("format") == "v2" else raw

    # Collect subjects with shot indices
    subjects_by_shot = []
    for entry in prompts:
        subjects = entry.get("description", {}).get("subjects", "")
        if subjects:
            subjects_by_shot.append(f"Shot {entry['index']}: {subjects}")

    if not subjects_by_shot:
        print("No subjects found in shots.json")
        sys.exit(1)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not set")
        sys.exit(1)

    client = _gemini_client(api_key)

    # --- TMDB supervised path ---
    tmdb_id = getattr(args, 'tmdb_id', None)
    characters_data = None
    if tmdb_id:
        tmdb_api_key = os.environ.get("TMDB_API_KEY")
        if not tmdb_api_key:
            print("Warning: --tmdb-id set but TMDB_API_KEY not found; using unsupervised mode")
        else:
            try:
                media_type = getattr(args, 'tmdb_type', 'movie') or 'movie'
                cast_entries = fetch_tmdb_cast(tmdb_id, tmdb_api_key, media_type)
                print(f"TMDB: fetched {len(cast_entries)} cast members")
                print(f"Analyzing {len(subjects_by_shot)} shots (supervised)...")
                characters_data = _run_supervised_stage3(
                    client, cast_entries, subjects_by_shot, prompts
                )
            except Exception as e:
                print(f"Warning: TMDB fetch failed ({e}); falling back to unsupervised mode")

    # --- Unsupervised path (fallback or no --tmdb-id) ---
    if characters_data is None:
        user_text = (
            f"Film has {len(prompts)} shots. "
            f"Subject descriptions from each shot:\n\n"
            + "\n".join(subjects_by_shot)
        )

        print(f"Analyzing subjects across {len(prompts)} shots...")
        response = client.models.generate_content(
            model=ENCODE_MODEL,
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=user_text)])],
            config=types.GenerateContentConfig(
                system_instruction=STAGE3_SYSTEM_PROMPT,
                temperature=0.3,
                response_mime_type="application/json",
                max_output_tokens=65536,
            ),
        )

        characters_data = json.loads(response.text.strip())

    # Validate structure
    characters = characters_data.get("characters", [])
    print(f"Found {len(characters)} characters:")
    for char in characters:
        print(f"  {char['display_name']} ({char['name']}): {len(char['shots'])} shots")

    # Refine: assign unassigned shots to characters via second pass
    characters_data = _refine_shot_assignments(
        client, characters_data, prompts, subjects_by_shot
    )
    characters = characters_data.get("characters", [])

    characters_path = manifest.characters_path(output_dir)
    _preserve_curated_fields(characters_path, characters_data)
    with open(characters_path, "w") as f:
        json.dump(characters_data, f, indent=2)

    print(f"\nCharacter registry saved to {characters_path}")


# ---------------------------------------------------------------------------
# Stage 4: Speaker attribution (dialog line -> character)
# ---------------------------------------------------------------------------
#
# Writes a decode-side sidecar (speakers.json) so shots.json stays untouched.
# Every assigned character name is gated against characters.json's registry
# (or the literal "narrator") — Gemini is never allowed to invent a name.
# This is the concrete mitigation for the character-naming instability
# documented in docs/research/0021-training-data-contamination/research.md.

SPEAKER_ATTRIBUTION_SYSTEM_PROMPT = """You are a film dialogue editor. You are given a batch of subtitle lines from a film, each with the characters known to be on-screen around that moment (candidates), plus neighboring lines for conversational context.

For each line, decide who is speaking.

Return a JSON object with an "assignments" array. Each entry has:
- "line": the line index (integer)
- "character": the speaker — MUST be exactly one of that line's candidate names, or the literal string "narrator" if none of the candidates is plausible (e.g. off-screen narration, or a candidate list that doesn't include the true speaker)
- "confidence": a float from 0.0 to 1.0

Rules:
- Never invent a character name. Only use names from that specific line's candidate list, or "narrator".
- Use the surrounding lines to track turn-taking in a conversation (speakers usually alternate).
- If a line has exactly one candidate, that candidate is almost certainly the speaker (high confidence) unless the content clearly contradicts it.
- If a line has no candidates at all, use "narrator".

Output ONLY valid JSON."""


def _build_shot_character_map(characters: list[dict]) -> dict[int, list[str]]:
    """Reverse characters.json's per-character shot lists into shot -> [names]."""
    shot_map: dict[int, list[str]] = {}
    for char in characters:
        for shot_idx in char.get("shots", []):
            shot_map.setdefault(shot_idx, []).append(char["name"])
    return shot_map


def _candidate_characters_for_line(line: dict, shots: list[dict],
                                    shot_character_map: dict[int, list[str]]) -> list[str]:
    """Union of character names from every shot overlapping this dialog line's timestamp."""
    candidates: list[str] = []
    for shot in shots:
        if shot["start_s"] >= line["end_s"] or shot["end_s"] <= line["start_s"]:
            continue
        for name in shot_character_map.get(shot["index"], []):
            if name not in candidates:
                candidates.append(name)
    return candidates


def run_stage4(args):
    """Stage 4: Attribute each dialog line to a character via Gemini."""
    from google import genai
    from google.genai import types

    output_dir = args.output_dir
    shots_file = manifest.shots_path(output_dir)
    characters_file = manifest.characters_path(output_dir)

    if not os.path.exists(shots_file):
        print(f"Error: {shots_file} not found. Run stage2 first.")
        sys.exit(1)
    if not os.path.exists(characters_file):
        print(f"Error: {characters_file} not found. Run stage3 first.")
        sys.exit(1)

    shots, dialog = manifest.load_shots(output_dir)
    with open(characters_file) as f:
        characters = json.load(f).get("characters", [])

    if not dialog:
        print("No dialog lines in shots.json; nothing to attribute.")
        sys.exit(0)

    valid_names = {c["name"] for c in characters}
    descriptions = {c["name"]: c["description"] for c in characters}
    shot_character_map = _build_shot_character_map(characters)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not set")
        sys.exit(1)
    client = _gemini_client(api_key)

    BATCH_SIZE = 50
    assignments: dict[int, dict] = {}

    print(f"Attributing {len(dialog)} dialog lines across {len(characters)} characters...")

    for batch_start in range(0, len(dialog), BATCH_SIZE):
        batch_indices = range(batch_start, min(batch_start + BATCH_SIZE, len(dialog)))
        lines_text = []
        for i in batch_indices:
            line = dialog[i]
            candidates = _candidate_characters_for_line(line, shots, shot_character_map)
            candidate_desc = ", ".join(
                f"{name} ({descriptions.get(name, '')[:60]})" for name in candidates
            ) or "(none on screen)"
            lines_text.append(
                f"Line {i}: \"{line['text']}\"\n  Candidates: {candidate_desc}"
            )

        user_text = "Subtitle lines (in order):\n\n" + "\n".join(lines_text)

        response = client.models.generate_content(
            model=ENCODE_MODEL,
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=user_text)])],
            config=types.GenerateContentConfig(
                system_instruction=SPEAKER_ATTRIBUTION_SYSTEM_PROMPT,
                temperature=0.2,
                response_mime_type="application/json",
            ),
        )

        try:
            batch_data = json.loads(response.text.strip())
        except (json.JSONDecodeError, AttributeError):
            batch_num = batch_start // BATCH_SIZE + 1
            print(f"  Warning: batch {batch_num} response not valid JSON, skipping")
            continue

        for entry in batch_data.get("assignments", []):
            line_idx = entry.get("line")
            character = entry.get("character")
            if line_idx is None:
                continue
            if character not in valid_names and character != "narrator":
                print(f"  Warning: line {line_idx} got unregistered name "
                      f"'{character}', falling back to narrator")
                character = "narrator"
            assignments[line_idx] = {
                "character": character,
                "confidence": entry.get("confidence", 0.0),
            }

    speakers_data = {
        "format": "v1",
        "model": ENCODE_MODEL,
        "assignments": {str(idx): val for idx, val in sorted(assignments.items())},
    }
    speakers_file = manifest.speakers_path(output_dir)
    with open(speakers_file, "w") as f:
        json.dump(speakers_data, f, indent=2)

    unattributed = len(dialog) - len(assignments)
    print(f"\nSpeaker attribution saved to {speakers_file}")
    print(f"  {len(assignments)} lines attributed, {unattributed} unattributed (batch failures)")


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

    print("\nExtracting keyframes (512px, adaptive frame count)...")
    extract_keyframes(args.video, scenes, args.output)

    shot_manifest = build_manifest(scenes, args.video)
    shot_index_path = manifest.shot_index_path(args.output)
    with open(shot_index_path, "w") as f:
        json.dump(shot_manifest, f, indent=2)

    print(f"\nShot index saved to {shot_index_path}")
    print(f"  {shot_manifest['shot_count']} shots detected")
    durations = [s["duration_s"] for s in shot_manifest["scenes"]]
    print(f"  Average shot duration: {sum(durations) / len(durations):.2f}s")
    print(f"  Shortest: {min(durations):.2f}s / Longest: {max(durations):.2f}s")


def run_stage2(args):
    """Stage 2: Enrich shots with metadata, generate prompts via vision API."""
    output_dir = args.output_dir
    shot_index_path = manifest.shot_index_path(output_dir)

    if not os.path.exists(shot_index_path):
        print(f"Error: {shot_index_path} not found. Run stage1 first.")
        sys.exit(1)

    with open(shot_index_path) as f:
        shot_manifest = json.load(f)

    video_path = shot_manifest["source"]["path"]
    scenes = shot_manifest["scenes"]

    if args.limit:
        scenes = scenes[:args.limit]
        print(f"Processing first {args.limit} shots (of {shot_manifest['shot_count']})")

    # Step 1: Extract subtitles
    srt_path = extract_subtitles(video_path, output_dir)
    subtitles = parse_srt(srt_path) if srt_path else []
    dialogue_map = align_subtitles_to_shots(subtitles, scenes)
    print(f"  {len(dialogue_map)} shots have dialogue ({len(subtitles)} total lines)")

    # Step 2: Camera motion detection
    print("Detecting camera motion...")
    motion_labels = detect_camera_motion(video_path, scenes, output_dir)

    # Step 3: Audio classification
    print("Classifying audio...")
    audio_labels = detect_audio_labels(video_path, scenes, output_dir)
    wav_path = os.path.join(output_dir, "audio.wav")

    # Step 4: Generate prompts via vision API
    print("Generating prompts...")
    # dialog is a global subtitle timeline (one entry per SRT line, at its
    # original timestamp) so that speech generation places each line exactly
    # once, even for shots that span subtitle boundaries.
    prompts = generate_prompts(scenes, dialogue_map, motion_labels, audio_labels, output_dir,
                                args.provider, dialog=subtitles, wav_path=wav_path)

    shots_path = manifest.save_shots(output_dir, prompts, subtitles)

    print(f"\nShots saved to {shots_path}")
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

    # Stage 3
    s3 = subparsers.add_parser("stage3", help="Build character registry from prompts")
    s3.add_argument("output_dir", help="Output directory from stage 2")
    s3.add_argument(
        "--tmdb-id",
        help="TMDB movie/TV ID for cast seeding (e.g. 11 for Star Wars IV)",
    )
    s3.add_argument(
        "--tmdb-type",
        choices=["movie", "tv"],
        default="movie",
        help="TMDB media type (default: movie)",
    )
    s3.set_defaults(func=run_stage3)

    # Stage 4
    s4 = subparsers.add_parser("stage4", help="Attribute dialog lines to characters (speakers.json)")
    s4.add_argument("output_dir", help="Output directory from stage 3")
    s4.set_defaults(func=run_stage4)

    args = parser.parse_args()
    load_env()
    args.func(args)


if __name__ == "__main__":
    main()
