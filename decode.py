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
    vary_prompt_for_part,
    _format_prompt_wan,
    _format_prompt_seedance,
)
from strategies_video import (  # noqa: E402 -- re-export for backwards compat
    GenerationStrategy,
    ReplicateWanStrategy,
    FalSeedanceStrategy,
    FalSeedanceProStrategy,
    RunPodWanStrategy,
    RunPodWan22Strategy,
    RunPodWanEnrichedStrategy,
    RunPodVaceStrategy,
)


from strategies_audio import (  # noqa: E402 -- re-export for backwards compat
    AudioStrategy,
    ElevenLabsStrategy,
    MMAudioStrategy,
    RunPodMMAudioStrategy,
    SpeechStrategy,
    filter_speech_from_sound,
)


# ---------------------------------------------------------------------------
# Portrait generation
# ---------------------------------------------------------------------------


def generate_portraits(characters_path: str, output_dir: str) -> dict:
    """Generate canonical portrait images for each character.

    Returns {character_name: portrait_path} mapping.
    """
    import fal_client
    import httpx

    with open(characters_path) as f:
        characters_data = json.load(f)

    characters_dir = os.path.join(output_dir, "characters")
    os.makedirs(characters_dir, exist_ok=True)

    portraits = {}
    for char in characters_data.get("characters", []):
        name = char["name"]
        portrait_path = os.path.join(characters_dir, f"{name}.png")

        if os.path.exists(portrait_path):
            print(f"  Portrait exists: {name}")
            portraits[name] = portrait_path
            continue

        prompt = (
            f"Professional portrait photograph of {char['description']}. "
            f"Clean background, studio lighting, sharp focus, photorealistic, "
            f"head and shoulders framing, neutral expression."
        )

        print(f"  Generating portrait: {char['display_name']}...")
        try:
            result = fal_client.subscribe(
                "fal-ai/flux/schnell",
                arguments={
                    "prompt": prompt,
                    "image_size": "square_hd",
                    "num_images": 1,
                },
                with_logs=False,
            )

            image_url = result["images"][0]["url"]
            resp = httpx.get(image_url, follow_redirects=True)
            resp.raise_for_status()
            with open(portrait_path, "wb") as f:
                f.write(resp.content)

            portraits[name] = portrait_path
            print(f"    Saved: {portrait_path}")

        except Exception as e:
            print(f"    Error generating portrait for {name}: {e}")

    return portraits


def build_character_shot_map(characters_data: dict) -> dict:
    """Build reverse mapping from shot index to character names."""
    shot_map = {}
    for char in characters_data.get("characters", []):
        for shot_idx in char.get("shots", []):
            shot_map.setdefault(shot_idx, []).append(char["name"])
    return shot_map


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
        raw = json.load(f)
    prompts = raw["shots"] if isinstance(raw, dict) and raw.get("format") == "v2" else raw

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
        raw = json.load(f)
    prompts = raw["shots"] if isinstance(raw, dict) and raw.get("format") == "v2" else raw

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
    """Speech generation loop: read prompts, generate TTS clips, track progress.

    Supports two prompts.json formats:
    - v2 ({"format": "v2", "shots": [...], "dialog": [...]}): generates one TTS
      clip per global dialog line, avoiding repetition across shot boundaries.
    - v1 (flat array): legacy per-shot dialog generation (backward compat).
    """
    output_dir = args.output_dir
    prompts_path = os.path.join(output_dir, "prompts.json")

    if not os.path.exists(prompts_path):
        print(f"Error: {prompts_path} not found. Run encoder first.")
        sys.exit(1)

    with open(prompts_path) as f:
        raw = json.load(f)

    speech_dir = os.path.join(output_dir, "speech")
    os.makedirs(speech_dir, exist_ok=True)

    is_v2 = isinstance(raw, dict) and raw.get("format") == "v2"

    if is_v2:
        _run_speech_v2(args, strategy, raw["dialog"], speech_dir, output_dir)
    else:
        _run_speech_v1(args, strategy, raw, speech_dir, output_dir)


def _run_speech_v2(args, strategy: SpeechStrategy, dialog: list[dict],
                   speech_dir: str, output_dir: str):
    """v2: generate one TTS clip per global dialog line at its SRT timestamp."""
    progress_path = os.path.join(output_dir, "speech_progress.json")
    progress = {"format": "v2", "completed": [], "failed": [],
                "total_cost_estimate": 0.0, "clips": {}}
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            loaded = json.load(f)
        if loaded.get("format") == "v2":
            progress = loaded

    completed_set = set(progress["completed"])
    generated = 0
    errors = 0

    print(f"Generating speech for {len(dialog)} dialog lines via ElevenLabs TTS "
          f"({len(completed_set)} done)...")

    for i, line in enumerate(dialog):
        if i in completed_set:
            continue

        expected_path = os.path.join(speech_dir, f"{i:04d}-00.mp3")
        if os.path.exists(expected_path):
            progress["clips"][str(i)] = {
                "path": os.path.basename(expected_path),
                "start_s": line["start_s"],
                "end_s": line["end_s"],
            }
            progress["completed"].append(i)
            completed_set.add(i)
            generated += 1
            with open(progress_path, "w") as f:
                json.dump(progress, f, indent=2)
            continue

        result = strategy.generate(line["text"], speech_dir, i, 0, line["start_s"])
        if result:
            progress["clips"][str(i)] = {
                "path": os.path.basename(result.path),
                "start_s": line["start_s"],
                "end_s": line["end_s"],
            }
            progress["total_cost_estimate"] += result.cost
            progress["completed"].append(i)
            completed_set.add(i)
            generated += 1
        else:
            errors += 1
            progress["failed"].append(i)

        with open(progress_path, "w") as f:
            json.dump(progress, f, indent=2)

        if errors > 20:
            print("Too many errors, saving progress and stopping.")
            break

    print(f"\nDone. Generated {generated} speech clips.")
    print(f"  Estimated cost: ${progress['total_cost_estimate']:.2f}")


def _run_speech_v1(args, strategy: SpeechStrategy, prompts: list[dict],
                   speech_dir: str, output_dir: str):
    """v1: legacy per-shot dialog generation (backward compat)."""
    if args.start_index:
        prompts = [p for p in prompts if p["index"] >= args.start_index]
    if args.limit:
        prompts = prompts[:args.limit]

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
    parser.add_argument("--strategy", choices=["replicate-wan", "fal-seedance", "fal-seedance-pro", "runpod-wan", "runpod-wan22", "runpod-wan-enriched", "runpod-vace"],
                        default="runpod-wan",
                        help="Video generation backend (default: runpod-wan)")
    parser.add_argument("--audio", action="store_true",
                        help="Generate audio clips (instead of video)")
    parser.add_argument("--audio-strategy", choices=["elevenlabs", "mmaudio", "runpod-mmaudio"],
                        default=None,
                        help="Audio generation backend (default: auto-detect all for stitch, elevenlabs for generate)")
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
        audio_name = args.audio_strategy or "elevenlabs"
        audio_strategy = audio_strategies[audio_name]()
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
            "runpod-wan22": lambda: RunPodWan22Strategy(
                output_dir=args.output_dir,
                keep_pod=getattr(args, "keep_pod", False),
                concurrent_audio=getattr(args, "concurrent_audio", False),
            ),
            "runpod-wan-enriched": lambda: _create_wan_enriched_strategy(args),
            "runpod-vace": lambda: _create_vace_strategy(args),
        }
        strategy = strategies[args.strategy]()
        run_decode(args, strategy)


def _create_wan_enriched_strategy(args):
    output_dir = args.output_dir
    characters_path = os.path.join(output_dir, "characters.json")

    character_shot_map = {}
    characters_data = {}

    if os.path.exists(characters_path):
        with open(characters_path) as f:
            characters_data = json.load(f)
        character_shot_map = build_character_shot_map(characters_data)
    else:
        print("Warning: characters.json not found. Running without prompt enrichment.")

    return RunPodWanEnrichedStrategy(
        output_dir=output_dir,
        keep_pod=getattr(args, "keep_pod", False),
        concurrent_audio=getattr(args, "concurrent_audio", False),
        character_shot_map=character_shot_map,
        characters_data=characters_data,
    )


def _create_vace_strategy(args):
    output_dir = args.output_dir
    characters_path = os.path.join(output_dir, "characters.json")

    portraits = {}
    character_shot_map = {}
    characters_data = {}

    if os.path.exists(characters_path):
        with open(characters_path) as f:
            characters_data = json.load(f)
        portraits = generate_portraits(characters_path, output_dir)
        character_shot_map = build_character_shot_map(characters_data)
    else:
        print("Warning: characters.json not found. Running VACE without reference images (T2V fallback).")

    return RunPodVaceStrategy(
        output_dir=output_dir,
        keep_pod=getattr(args, "keep_pod", False),
        concurrent_audio=getattr(args, "concurrent_audio", False),
        portraits=portraits,
        character_shot_map=character_shot_map,
        characters_data=characters_data,
    )


if __name__ == "__main__":
    main()
