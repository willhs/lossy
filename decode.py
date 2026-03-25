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


from clip_types import ClipResult, AudioClipResult, SpeechClipResult  # noqa: E402


from prompt_format import (  # noqa: E402 -- re-export for backwards compat
    CAMERA_TERMS,
    format_prompt,
    _format_prompt_wan,
    _format_prompt_seedance,
)
from strategies_video import (  # noqa: E402 -- re-export for backwards compat
    GenerationStrategy,
    ReplicateWanStrategy,
    FalSeedanceStrategy,
    FalSeedanceProStrategy,
    RunPodWanStrategy,
)


from strategies_audio import (  # noqa: E402 -- re-export for backwards compat
    AudioStrategy,
    ElevenLabsStrategy,
    MMAudioStrategy,
    RunPodMMAudioStrategy,
    SpeechStrategy,
)


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

    clips_dir = os.path.join(output_dir, "clips", strategy.name)
    os.makedirs(clips_dir, exist_ok=True)

    # Track progress for resume (per-strategy)
    progress_path = os.path.join(output_dir, f"decode_progress_{strategy.name}.json")
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

        prompt_text = strategy.format_prompt(entry)
        print(f"  Shot {idx} ({generated + 1}/{total - len(completed_set)} remaining)...")

        results = strategy.generate(prompt_text, clips_dir, idx, entry["duration_s"], seed=idx, entry=entry)

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
            results = strategy.generate(prompt_text, clips_dir, idx, entry["duration_s"], seed=idx, entry=entry)
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

    # Finish pipelined audio (last shot + wait for thread)
    if hasattr(strategy, 'finish_audio'):
        strategy.finish_audio()

    # Final save
    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2)

    # Save audio progress if pipelined audio was used
    if hasattr(strategy, 'get_audio_results'):
        audio_results, audio_failures = strategy.get_audio_results()
        if audio_results or audio_failures:
            audio_progress_path = os.path.join(output_dir, "audio_progress_runpod-mmaudio-pipelined.json")
            audio_progress = {
                "completed": sorted(audio_results.keys()),
                "failed": audio_failures,
                "skipped": [],
                "total_cost_estimate": 0.0,
                "clips": {
                    str(idx): [
                        {"path": os.path.basename(r.path), "duration_s": r.actual_duration_s}
                        for r in results_list
                    ]
                    for idx, results_list in audio_results.items()
                },
            }
            with open(audio_progress_path, "w") as f:
                json.dump(audio_progress, f, indent=2)
            print(f"  Pipelined audio: {len(audio_results)} completed, {len(audio_failures)} failed")

    # Signal clean exit for --keep-pod support
    if hasattr(strategy, 'mark_clean_exit'):
        strategy.mark_clean_exit()

    print(f"\nDone. Generated {generated} clips.")
    print(f"  Total: {len(progress['completed'])} completed, {len(progress['failed'])} failed")
    print(f"  Estimated cost: ${progress['total_cost_estimate']:.2f}")


# ---------------------------------------------------------------------------
# Audio generation loop
# ---------------------------------------------------------------------------


def run_audio(args, strategy: AudioStrategy):
    """Audio generation loop: read prompts, generate audio clips, track progress."""
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

    audio_dir = os.path.join(output_dir, "audio", strategy.name)
    os.makedirs(audio_dir, exist_ok=True)

    # Progress tracking (per audio strategy)
    progress_path = os.path.join(output_dir, f"audio_progress_{strategy.name}.json")
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            progress = json.load(f)
    else:
        progress = {"completed": [], "failed": [], "skipped": [], "total_cost_estimate": 0.0, "clips": {}}

    completed_set = set(progress["completed"])
    skipped_set = set(progress.get("skipped", []))
    total = len(prompts)
    generated = 0
    skipped = 0
    errors = 0

    to_generate = total - len(completed_set) - len(skipped_set)
    print(f"Generating audio for {total} shots via {strategy.name} "
          f"({len(completed_set)} done, {len(skipped_set)} skipped)...")

    for entry in prompts:
        idx = entry["index"]

        if idx in completed_set or idx in skipped_set:
            continue

        # Extract sound description
        sound = entry.get("description", {}).get("sound")
        if not sound:
            print(f"  Shot {idx}: no sound description, skipping")
            skipped += 1
            progress["skipped"].append(idx)
            skipped_set.add(idx)
            continue

        # Skip if already generated
        ext = ".mp3" if isinstance(strategy, ElevenLabsStrategy) else ".flac"
        primary_clip = os.path.join(audio_dir, f"{idx:04d}{ext}")
        part_clip = os.path.join(audio_dir, f"{idx:04d}-01{ext}")
        if os.path.exists(primary_clip) or os.path.exists(part_clip):
            completed_set.add(idx)
            if idx not in progress["completed"]:
                progress["completed"].append(idx)
            continue

        print(f"  Shot {idx} ({generated + 1}/{to_generate} remaining)...")

        results = strategy.generate(sound, audio_dir, idx, entry["duration_s"], seed=idx)

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
            print(f"  Retrying shot {idx} in 10s...")
            time.sleep(10)
            results = strategy.generate(sound, audio_dir, idx, entry["duration_s"], seed=idx)
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

    print(f"\nDone. Generated {generated} audio clips, skipped {skipped}.")
    print(f"  Total: {len(progress['completed'])} completed, "
          f"{len(progress.get('skipped', []))} skipped (no sound), "
          f"{len(progress['failed'])} failed")
    print(f"  Estimated cost: ${progress['total_cost_estimate']:.2f}")


# ---------------------------------------------------------------------------
# Speech generation loop
# ---------------------------------------------------------------------------


def run_speech(args, strategy: SpeechStrategy):
    """Speech generation loop: read prompts, generate TTS clips, track progress."""
    output_dir = args.output_dir
    prompts_path = os.path.join(output_dir, "prompts.json")

    if not os.path.exists(prompts_path):
        print(f"Error: {prompts_path} not found. Run encoder first.")
        sys.exit(1)

    with open(prompts_path) as f:
        prompts = json.load(f)

    if args.start_index:
        prompts = [p for p in prompts if p["index"] >= args.start_index]
    if args.limit:
        prompts = prompts[:args.limit]

    speech_dir = os.path.join(output_dir, "speech")
    os.makedirs(speech_dir, exist_ok=True)

    progress_path = os.path.join(output_dir, "speech_progress.json")
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            progress = json.load(f)
    else:
        progress = {"completed": [], "skipped": [], "failed": [],
                     "total_cost_estimate": 0.0, "clips": {}}

    completed_set = set(progress["completed"])
    skipped_set = set(progress.get("skipped", []))
    generated = 0
    skipped = 0
    errors = 0

    # Count total dialogue lines
    total_lines = sum(
        len(e.get("dialogue") or [])
        for e in prompts
        if isinstance(e.get("dialogue"), list)
           and e.get("dialogue")
           and isinstance(e["dialogue"][0], dict)
    )

    print(f"Generating speech for {len(prompts)} shots ({total_lines} lines) "
          f"via ElevenLabs TTS ({len(completed_set)} shots done)...")

    for entry in prompts:
        idx = entry["index"]
        if idx in completed_set or idx in skipped_set:
            continue

        dialogue = entry.get("dialogue")
        if not dialogue or not isinstance(dialogue, list):
            skipped += 1
            progress["skipped"].append(idx)
            skipped_set.add(idx)
            continue

        # Handle both enriched (dict) and legacy (str) formats
        if not isinstance(dialogue[0], dict):
            print(f"  Shot {idx}: dialogue not enriched (plain strings), skipping")
            skipped += 1
            progress["skipped"].append(idx)
            skipped_set.add(idx)
            continue

        shot_clips = []
        shot_ok = True
        for line_idx, line in enumerate(dialogue):
            clip_path = os.path.join(speech_dir, f"{idx:04d}-{line_idx:02d}.mp3")
            if os.path.exists(clip_path):
                shot_clips.append({
                    "path": os.path.basename(clip_path),
                    "offset_s": line["start_s"],
                })
                continue

            result = strategy.generate(
                line["text"], speech_dir, idx, line_idx, line["start_s"]
            )
            if result:
                shot_clips.append({
                    "path": os.path.basename(result.path),
                    "duration_s": result.duration_s,
                    "offset_s": result.offset_s,
                })
                progress["total_cost_estimate"] += result.cost
            else:
                errors += 1
                shot_ok = False
                break

        if shot_ok and shot_clips:
            generated += 1
            progress["completed"].append(idx)
            completed_set.add(idx)
            progress["clips"][str(idx)] = shot_clips
        elif not shot_ok:
            progress["failed"].append(idx)

        if generated % 5 == 0:
            with open(progress_path, "w") as f:
                json.dump(progress, f, indent=2)

        if errors > 20:
            print("Too many errors, saving progress and stopping.")
            break

    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2)

    print(f"\nDone. Generated speech for {generated} shots, skipped {skipped}.")
    print(f"  Estimated cost: ${progress['total_cost_estimate']:.2f}")


from stitch import stitch_clips  # noqa: E402,F811 -- re-export for backwards compat


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
    parser.add_argument("--strategy", choices=["replicate-wan", "fal-seedance", "fal-seedance-pro", "runpod-wan"],
                        default="replicate-wan",
                        help="Video generation backend (default: replicate-wan)")
    parser.add_argument("--audio", action="store_true",
                        help="Generate audio clips (instead of video)")
    parser.add_argument("--audio-strategy", choices=["elevenlabs", "mmaudio", "runpod-mmaudio"],
                        default="elevenlabs",
                        help="Audio generation backend (default: elevenlabs)")
    parser.add_argument("--keep-pod", action="store_true",
                        help="Keep RunPod pod alive after decode (for subsequent audio stage)")
    parser.add_argument("--speech", action="store_true",
                        help="Generate speech/dialogue clips (instead of video)")
    parser.add_argument("--speech-voice", default="Roger",
                        help="ElevenLabs voice name for speech (default: Roger)")
    parser.add_argument("--concurrent-audio", action="store_true",
                        help="Generate audio concurrently with video on RunPod (requires >=24GB VRAM)")
    args = parser.parse_args()

    load_env()

    if args.stitch:
        stitch_clips(args)
    elif args.audio:
        audio_strategies = {
            "elevenlabs": lambda: ElevenLabsStrategy(),
            "mmaudio": lambda: MMAudioStrategy(),
            "runpod-mmaudio": lambda: RunPodMMAudioStrategy(output_dir=args.output_dir),
        }
        audio_strategy = audio_strategies[args.audio_strategy]()
        run_audio(args, audio_strategy)
    elif args.speech:
        speech_strategy = SpeechStrategy(voice=args.speech_voice)
        run_speech(args, speech_strategy)
    else:
        strategies = {
            "replicate-wan": lambda: ReplicateWanStrategy(),
            "fal-seedance": lambda: FalSeedanceStrategy(),
            "fal-seedance-pro": lambda: FalSeedanceProStrategy(),
            "runpod-wan": lambda: RunPodWanStrategy(
                output_dir=args.output_dir,
                keep_pod=getattr(args, "keep_pod", False),
                concurrent_audio=getattr(args, "concurrent_audio", False),
            ),
        }
        strategy = strategies[args.strategy]()
        run_decode(args, strategy)


if __name__ == "__main__":
    main()
