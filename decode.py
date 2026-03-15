#!/usr/bin/env python3
"""
lossy decoder: prompt manifest -> video clips -> reconstructed film.

Usage:
    python decode.py output/star_wars_iv_v2 [--start-index 7] [--limit 20]
    python decode.py output/star_wars_iv_v2 --stitch [--start-index 7]
    python decode.py output/star_wars_iv_v2 --strategy fal-seedance [--start-index 10]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass


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


# ---------------------------------------------------------------------------
# Strategy pattern for video generation backends
# ---------------------------------------------------------------------------


@dataclass
class ClipResult:
    """Result of generating a single video clip."""
    path: str
    actual_duration_s: float
    cost: float


class GenerationStrategy:
    """Base class for video generation backends."""

    name: str = "base"

    def generate(
        self,
        prompt: str,
        clips_dir: str,
        shot_index: int,
        target_duration_s: float,
        seed: int | None = None,
    ) -> list[ClipResult]:
        """Generate clip(s) for a shot.

        Returns a list because long shots may be split into multiple clips.
        """
        raise NotImplementedError


class ReplicateWanStrategy(GenerationStrategy):
    """Replicate Wan 2.2 Fast -- fixed ~5.06s clips at $0.05 each."""

    name = "replicate-wan"
    CLIP_DURATION = 81 / 16  # ~5.0625s

    def generate(
        self,
        prompt: str,
        clips_dir: str,
        shot_index: int,
        target_duration_s: float,
        seed: int | None = None,
    ) -> list[ClipResult]:
        import httpx
        import replicate

        clip_path = os.path.join(clips_dir, f"{shot_index:04d}.mp4")

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

            if hasattr(output, "url"):
                url = output.url
            elif isinstance(output, str):
                url = output
            else:
                url = str(output)

            resp = httpx.get(url, follow_redirects=True)
            resp.raise_for_status()
            with open(clip_path, "wb") as f:
                f.write(resp.content)

            return [ClipResult(path=clip_path, actual_duration_s=self.CLIP_DURATION, cost=0.05)]

        except Exception as e:
            print(f"  Error generating clip: {e}")
            return []


class FalSeedanceStrategy(GenerationStrategy):
    """fal.ai Seedance 1.0 Pro Fast -- 2-12s duration control at ~$0.10/clip (480p)."""

    name = "fal-seedance"
    MIN_DURATION = 2
    MAX_DURATION = 12
    COST_PER_SECOND_480P = 0.02  # Approximate: ~$0.10 for 5s at 480p

    def _target_durations(self, target_s: float) -> list[int]:
        """Split target duration into a list of integer durations within 2-12s range.

        Examples:
            3.7s  -> [4]
            11.0s -> [11]
            15.0s -> [12, 3]
            19.0s -> [12, 7]
            25.0s -> [12, 12, 2]  (remainder clamped to min 2)
        """
        if target_s <= self.MAX_DURATION:
            clamped = max(self.MIN_DURATION, min(self.MAX_DURATION, round(target_s)))
            return [clamped]

        # Split into MAX_DURATION chunks plus remainder
        parts = []
        remaining = target_s
        while remaining > self.MAX_DURATION:
            parts.append(self.MAX_DURATION)
            remaining -= self.MAX_DURATION
        # Remainder
        remainder = max(self.MIN_DURATION, round(remaining))
        parts.append(remainder)
        return parts

    def generate(
        self,
        prompt: str,
        clips_dir: str,
        shot_index: int,
        target_duration_s: float,
        seed: int | None = None,
    ) -> list[ClipResult]:
        import fal_client
        import httpx

        durations = self._target_durations(target_duration_s)
        results = []

        for part_idx, duration in enumerate(durations):
            # Single clip: 0010.mp4, split clips: 0010-01.mp4, 0010-02.mp4
            if len(durations) == 1:
                clip_name = f"{shot_index:04d}.mp4"
            else:
                clip_name = f"{shot_index:04d}-{part_idx + 1:02d}.mp4"

            clip_path = os.path.join(clips_dir, clip_name)

            try:
                arguments = {
                    "prompt": prompt,
                    "aspect_ratio": "16:9",
                    "resolution": "480p",
                    "duration": str(duration),
                }
                if seed is not None:
                    # Vary seed across parts so they don't look identical
                    arguments["seed"] = seed + part_idx

                result = fal_client.subscribe(
                    "fal-ai/bytedance/seedance/v1/pro/fast/text-to-video",
                    arguments=arguments,
                    with_logs=False,
                )

                video_url = result["video"]["url"]
                resp = httpx.get(video_url, follow_redirects=True)
                resp.raise_for_status()
                with open(clip_path, "wb") as f:
                    f.write(resp.content)

                cost = duration * self.COST_PER_SECOND_480P
                results.append(ClipResult(
                    path=clip_path,
                    actual_duration_s=float(duration),
                    cost=cost,
                ))

            except Exception as e:
                print(f"  Error generating clip {clip_name}: {e}")
                return []  # Fail the whole shot if any part fails

        return results


# ---------------------------------------------------------------------------
# Prompt formatting (strategy-independent)
# ---------------------------------------------------------------------------


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
# Decode loop
# ---------------------------------------------------------------------------


def run_decode(args, strategy: GenerationStrategy):
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

    # Initialize clips metadata in progress if not present
    if "clips" not in progress:
        progress["clips"] = {}

    completed_set = set(progress["completed"])
    total = len(prompts)
    generated = 0
    errors = 0

    print(f"Generating {total} clips via {strategy.name} ({len(completed_set)} already done)...")

    for entry in prompts:
        idx = entry["index"]

        if idx in completed_set:
            continue

        # Skip if already generated (check for primary clip file or split parts)
        primary_clip = os.path.join(clips_dir, f"{idx:04d}.mp4")
        part_clip = os.path.join(clips_dir, f"{idx:04d}-01.mp4")
        if os.path.exists(primary_clip) or os.path.exists(part_clip):
            completed_set.add(idx)
            if idx not in progress["completed"]:
                progress["completed"].append(idx)
            continue

        prompt_text = format_prompt(entry)
        print(f"  Shot {idx} ({generated + 1}/{total - len(completed_set)} remaining)...")

        results = strategy.generate(prompt_text, clips_dir, idx, entry["duration_s"], seed=idx)

        if results:
            generated += 1
            progress["completed"].append(idx)
            completed_set.add(idx)
            clip_cost = sum(r.cost for r in results)
            progress["total_cost_estimate"] += clip_cost
            progress["clips"][str(idx)] = [
                {"path": os.path.basename(r.path), "duration_s": r.actual_duration_s}
                for r in results
            ]
        else:
            errors += 1
            progress["failed"].append(idx)
            # Retry once after a short wait
            print(f"  Retrying shot {idx} in 10s...")
            time.sleep(10)
            results = strategy.generate(prompt_text, clips_dir, idx, entry["duration_s"], seed=idx)
            if results:
                generated += 1
                progress["completed"].append(idx)
                completed_set.add(idx)
                clip_cost = sum(r.cost for r in results)
                progress["total_cost_estimate"] += clip_cost
                progress["clips"][str(idx)] = [
                    {"path": os.path.basename(r.path), "duration_s": r.actual_duration_s}
                    for r in results
                ]
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


def _probe_duration(clip_path: str) -> float:
    """Get video duration via ffprobe. Fallback for clips without metadata."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "csv=p=0",
                clip_path,
            ],
            capture_output=True,
            text=True,
        )
        return float(result.stdout.strip())
    except (ValueError, subprocess.SubprocessError):
        return 81 / 16  # Last resort fallback


def stitch_clips(args):
    """Concatenate all clips into a single reconstructed video.

    Speed-adjusts each clip to match original shot duration using FFmpeg's
    setpts filter. Uses clip metadata from decode_progress.json when available,
    falls back to ffprobe for clips without metadata.
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

    # Load clip metadata from progress
    progress_path = os.path.join(output_dir, "decode_progress.json")
    clips_meta = {}
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            progress = json.load(f)
        clips_meta = progress.get("clips", {})

    # Collect existing clips in order
    clip_entries = []
    for entry in prompts:
        idx = entry["index"]
        idx_str = str(idx)

        if idx_str in clips_meta:
            # Use metadata -- handles both single clips and splits
            for clip_info in clips_meta[idx_str]:
                clip_path = os.path.join(clips_dir, clip_info["path"])
                if os.path.exists(clip_path):
                    clip_entries.append((clip_path, entry["duration_s"], clip_info["duration_s"]))
        else:
            # Backwards compat: single clip, probe or assume old default
            clip_path = os.path.join(clips_dir, f"{idx:04d}.mp4")
            if os.path.exists(clip_path):
                clip_duration = _probe_duration(clip_path)
                clip_entries.append((clip_path, entry["duration_s"], clip_duration))

    if not clip_entries:
        print("No clips found to stitch.")
        sys.exit(1)

    print(f"Stitching {len(clip_entries)} clips...")

    # Speed-adjust each clip to match original duration, write to temp dir
    adjusted_dir = os.path.join(output_dir, "adjusted")
    os.makedirs(adjusted_dir, exist_ok=True)

    concat_list = []
    for clip_path, original_duration, actual_duration in clip_entries:
        basename = os.path.splitext(os.path.basename(clip_path))[0]
        adjusted_path = os.path.join(adjusted_dir, f"{basename}.mp4")

        if not os.path.exists(adjusted_path):
            # For split clips, don't speed-adjust -- they're already duration-matched
            if "-" in basename:
                speed_factor = 1.0
            else:
                speed_factor = original_duration / actual_duration if actual_duration > 0 else 1.0

            if abs(speed_factor - 1.0) < 0.05:
                # Close enough -- just copy
                subprocess.run(["cp", clip_path, adjusted_path], capture_output=True)
            else:
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
    total_original = sum(orig_dur for _, orig_dur, _ in clip_entries)
    print(f"  Original duration: {total_original:.1f}s ({total_original / 60:.1f}min)")
    print(f"  Clips used: {len(clip_entries)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="lossy decoder: prompts -> video")
    parser.add_argument("output_dir", help="Output directory containing prompts.json")
    parser.add_argument("--start-index", type=int, default=None,
                        help="Skip shots before this index (e.g., 10 to skip credits)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N shots (after start-index)")
    parser.add_argument("--stitch", action="store_true",
                        help="Only run the stitching step (skip generation)")
    parser.add_argument("--strategy", choices=["replicate-wan", "fal-seedance"],
                        default="replicate-wan",
                        help="Video generation backend (default: replicate-wan)")
    args = parser.parse_args()

    load_env()

    if args.stitch:
        stitch_clips(args)
    else:
        strategies = {
            "replicate-wan": ReplicateWanStrategy,
            "fal-seedance": FalSeedanceStrategy,
        }
        strategy = strategies[args.strategy]()
        run_decode(args, strategy)


if __name__ == "__main__":
    main()
