#!/usr/bin/env python3
"""
lossy encoder: detect shots in a video and extract keyframes + metadata.

Usage:
    python encode.py media/star_wars_iv.mp4 --output output/star_wars_iv
"""

import argparse
import json
import os
import sys
import time

from scenedetect import open_video, SceneManager, AdaptiveDetector, ContentDetector, save_images


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

    return scenes, video


def export_keyframes(scenes, video, output_dir: str):
    """Save representative keyframes for each shot."""
    keyframes_dir = os.path.join(output_dir, "keyframes")
    os.makedirs(keyframes_dir, exist_ok=True)

    print(f"Saving keyframes to {keyframes_dir}/...")
    image_map = save_images(
        scenes,
        video,
        num_images=1,  # 1 frame per shot (middle frame)
        output_dir=keyframes_dir,
        image_extension="jpg",
    )
    return image_map


def build_manifest(scenes, video_path: str, image_map: dict) -> dict:
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
        # image_map keys are scene indices (0-based)
        keyframe_files = image_map.get(i, [])
        keyframe = keyframe_files[0] if keyframe_files else None

        scene_entry = {
            "index": i,
            "start_timecode": start.get_timecode(),
            "end_timecode": end.get_timecode(),
            "start_s": round(start.get_seconds(), 3),
            "end_s": round(end.get_seconds(), 3),
            "duration_s": round((end - start).get_seconds(), 3),
            "start_frame": start.get_frames(),
            "end_frame": end.get_frames(),
            "keyframe": keyframe,
            "description": None,  # filled later by vision model
        }
        manifest["scenes"].append(scene_entry)

    return manifest


def main():
    parser = argparse.ArgumentParser(description="lossy encoder: video → scene manifest")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--output", "-o", default="output", help="Output directory")
    parser.add_argument("--detector", "-d", default="adaptive", choices=["adaptive", "content"])
    parser.add_argument("--threshold", "-t", type=float, default=None, help="Detection threshold")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: {args.video} not found")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    # Detect shots
    scenes, video = detect_shots(args.video, args.detector, args.threshold)

    if not scenes:
        print("No shots detected. Try lowering the threshold.")
        sys.exit(1)

    # Extract keyframes
    image_map = export_keyframes(scenes, video, args.output)

    # Build and save manifest
    manifest = build_manifest(scenes, args.video, image_map)

    manifest_path = os.path.join(args.output, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest saved to {manifest_path}")
    print(f"  {manifest['shot_count']} shots detected")
    if manifest["scenes"]:
        durations = [s["duration_s"] for s in manifest["scenes"]]
        print(f"  Average shot duration: {sum(durations) / len(durations):.2f}s")
        print(f"  Shortest: {min(durations):.2f}s / Longest: {max(durations):.2f}s")


if __name__ == "__main__":
    main()
