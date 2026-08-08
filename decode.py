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

from config import (  # noqa: E402 -- re-export for backwards compat
    DECODE_AUDIO_STRATEGIES,
    MUSIC_STRATEGIES,
    VIDEO_STRATEGY_NAMES,
    load_env,
)
from clip_types import (  # noqa: E402,F401 -- re-export for backwards compat
    ClipResult,
    AudioClipResult,
    SpeechClipResult,
)
import manifest  # noqa: E402


from prompt_format import (  # noqa: E402,F401 -- re-export for backwards compat
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
    RunPodLtx2Strategy,
)


from strategies_audio import (  # noqa: E402 -- re-export for backwards compat
    AudioStrategy,
    ElevenLabsStrategy,
    KokoroSpeechStrategy,
    MMAudioStrategy,
    RunPodMMAudioStrategy,
    ReplicateMusicGenStrategy,
    RunPodMusicGenStrategy,
    SpeechStrategy,
    filter_speech_from_sound,  # noqa: F401 -- re-export for backwards compat
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


def _record_success(progress: dict, completed_set: set, idx: int, results: list) -> None:
    """Record a successful generation (first attempt or retry) in the progress dict."""
    progress["completed"].append(idx)
    completed_set.add(idx)
    progress["total_cost_estimate"] += sum(r.cost for r in results)
    progress["clips"][str(idx)] = [
        {"path": os.path.basename(r.path), "duration_s": r.actual_duration_s}
        for r in results
    ]


# ---------------------------------------------------------------------------
# Decode loop
# ---------------------------------------------------------------------------


def run_decode(args, strategy: GenerationStrategy):
    """Main decode loop: read prompts, generate clips, track progress."""
    output_dir = args.output_dir

    try:
        prompts, _ = manifest.load_shots(output_dir)
    except FileNotFoundError:
        print(f"Error: {manifest.shots_path(output_dir)} not found. Run encoder first.")
        sys.exit(1)

    # Apply start index and limit
    if args.start_index:
        prompts = [p for p in prompts if p["index"] >= args.start_index]
        print(f"Starting from shot index {args.start_index}")

    if args.limit:
        prompts = prompts[:args.limit]
        print(f"Processing {len(prompts)} shots")

    clips_dir = manifest.clips_dir(output_dir, strategy.name)
    os.makedirs(clips_dir, exist_ok=True)

    # Track progress for resume (per-strategy)
    progress_path = manifest.decode_progress_path(output_dir, strategy.name)
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            progress = json.load(f)
        progress["_mtime"] = os.path.getmtime(progress_path)
        if not manifest.check_encode_fingerprint(progress, output_dir, f"decode progress for {strategy.name}"):
            sys.exit(1)
        progress.pop("_mtime", None)
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

        # Adopt clips already on disk, but only when EVERY part of the shot is
        # there. Interrupting a run mid-shot leaves a split shot with its first
        # part(s) written and the rest missing; treating that as done both
        # truncates the shot and -- since this path used to leave clips_meta
        # empty -- hid the parts from the stitch's single-file fallback, which
        # dropped the shot from the film without a word. Partial shots
        # regenerate instead.
        expected_parts = strategy.expected_part_count(entry["duration_s"])
        existing = manifest.existing_clip_parts(clips_dir, idx, expected_parts)
        if existing:
            completed_set.add(idx)
            if idx not in progress["completed"]:
                progress["completed"].append(idx)
            progress["clips"][str(idx)] = [
                {"path": os.path.basename(p), "duration_s": manifest.probe_duration(p)}
                for p in existing
            ]
            continue

        prompt_text = strategy.format_prompt(entry)
        print(f"  Shot {idx} ({generated + 1}/{total - len(completed_set)} remaining)...")

        results = strategy.generate(prompt_text, clips_dir, idx, entry["duration_s"], seed=idx, entry=entry)

        if results:
            generated += 1
            _record_success(progress, completed_set, idx, results)
        else:
            errors += 1
            progress["failed"].append(idx)
            # Retry once after a short wait
            print(f"  Retrying shot {idx} in 10s...")
            time.sleep(10)
            results = strategy.generate(prompt_text, clips_dir, idx, entry["duration_s"], seed=idx, entry=entry)
            if results:
                generated += 1
                _record_success(progress, completed_set, idx, results)
                progress["failed"] = [f for f in progress["failed"] if f != idx]

        # Save progress every 5 clips
        if generated % 5 == 0:
            with open(progress_path, "w") as f:
                json.dump(progress, f, indent=2)

        if errors > 20:
            print("Too many errors, saving progress and stopping.")
            break

    # Finish pipelined audio (last shot + wait for thread)
    strategy.finish_audio()

    # Final save
    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2)

    # Save audio progress if pipelined audio was used
    audio_results, audio_failures = strategy.get_audio_results()
    if audio_results or audio_failures:
        audio_progress_path = manifest.audio_progress_path(output_dir, "runpod-mmaudio-pipelined")
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

    try:
        prompts, _ = manifest.load_shots(output_dir)
    except FileNotFoundError:
        print(f"Error: {manifest.shots_path(output_dir)} not found. Run encoder first.")
        sys.exit(1)

    # Apply start index and limit
    if args.start_index:
        prompts = [p for p in prompts if p["index"] >= args.start_index]
        print(f"Starting from shot index {args.start_index}")

    if args.limit:
        prompts = prompts[:args.limit]
        print(f"Processing {len(prompts)} shots")

    audio_dir = manifest.audio_dir(output_dir, strategy.name)
    os.makedirs(audio_dir, exist_ok=True)

    # Progress tracking (per audio strategy)
    progress_path = manifest.audio_progress_path(output_dir, strategy.name)
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            progress = json.load(f)
        progress["_mtime"] = os.path.getmtime(progress_path)
        if not manifest.check_encode_fingerprint(progress, output_dir, f"audio progress for {strategy.name}"):
            sys.exit(1)
        progress.pop("_mtime", None)
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

        # Extract description: music strategies read the 'music' field; SFX strategies read 'sound'
        desc_field = "music" if getattr(strategy, "uses_music_field", False) else "sound"
        desc = entry.get("description")
        sound = desc.get(desc_field) if isinstance(desc, dict) else None
        if not sound:
            skipped += 1
            progress["skipped"].append(idx)
            skipped_set.add(idx)
            continue

        # Skip if already generated
        ext = ".mp3" if isinstance(strategy, ElevenLabsStrategy) else ".flac"
        if manifest.clip_exists_for_shot(audio_dir, idx, ext=ext):
            completed_set.add(idx)
            if idx not in progress["completed"]:
                progress["completed"].append(idx)
            continue

        print(f"  Shot {idx} ({generated + 1}/{to_generate} remaining)...")

        results = strategy.generate(sound, audio_dir, idx, entry["duration_s"], seed=idx)

        if results:
            generated += 1
            _record_success(progress, completed_set, idx, results)
        else:
            errors += 1
            progress["failed"].append(idx)
            print(f"  Retrying shot {idx} in 10s...")
            time.sleep(10)
            results = strategy.generate(sound, audio_dir, idx, entry["duration_s"], seed=idx)
            if results:
                generated += 1
                _record_success(progress, completed_set, idx, results)
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


def run_speech(args, strategy: "SpeechStrategy | KokoroSpeechStrategy"):
    """Speech generation loop: one TTS clip per global dialog line.

    Reads prompts.json v2 (``dialog`` is a flat subtitle timeline) and
    generates TTS for each line at its original SRT timestamp, so lines
    that span shot boundaries are spoken exactly once.

    If ``speakers.json``/``voice_map.json`` exist (see ``encode.py stage4``
    and ``voice_casting.py``), each line is voiced by its attributed
    character's mapped voice; otherwise every line falls back to
    ``strategy`` (the single ``--speech-voice``) unchanged from before.
    """
    output_dir = args.output_dir

    try:
        shots, dialog = manifest.load_shots(output_dir)
    except FileNotFoundError:
        print(f"Error: {manifest.shots_path(output_dir)} not found. Run encoder first.")
        sys.exit(1)

    line_indices = range(len(dialog))
    if getattr(args, "start_index", None) or getattr(args, "limit", None):
        selected_shots = shots
        if args.start_index:
            selected_shots = [s for s in selected_shots if s["index"] >= args.start_index]
        if args.limit:
            selected_shots = selected_shots[:args.limit]
        if not selected_shots:
            print("No shots selected by --start-index/--limit; nothing to do.")
            return
        window_start = selected_shots[0]["start_s"]
        window_end = selected_shots[-1]["start_s"] + selected_shots[-1]["duration_s"]
        line_indices = [i for i, line in enumerate(dialog)
                        if line["start_s"] < window_end and line["end_s"] > window_start]
        print(f"Restricting speech to shots [{selected_shots[0]['index']}, "
              f"{selected_shots[-1]['index']}] -> {len(line_indices)} dialog lines "
              f"(of {len(dialog)})")

    speakers = manifest.load_speakers(output_dir)
    voice_map = manifest.load_voice_map(output_dir)
    strategies_by_voice: dict[str, "SpeechStrategy | KokoroSpeechStrategy"] = {strategy.voice: strategy}

    def strategy_for_line(line_idx: int) -> "SpeechStrategy | KokoroSpeechStrategy":
        character = speakers.get(line_idx)
        voice = voice_map.get(character, strategy.voice) if character else strategy.voice
        if voice not in strategies_by_voice:
            strategies_by_voice[voice] = type(strategy)(voice=voice)
        return strategies_by_voice[voice]

    speech_dir = manifest.speech_dir(output_dir)
    os.makedirs(speech_dir, exist_ok=True)

    progress_path = manifest.speech_progress_path(output_dir)
    progress = {"format": "v2", "completed": [], "failed": [],
                "total_cost_estimate": 0.0, "clips": {}}
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            loaded = json.load(f)
        if loaded.get("format") == "v2":
            # Speech clips are keyed by dialog-line index, which a re-encode
            # renumbers just as it renumbers shots -- reusing them would put
            # the wrong words on the wrong moments.
            loaded["_mtime"] = os.path.getmtime(progress_path)
            if not manifest.check_encode_fingerprint(loaded, output_dir, "speech progress"):
                sys.exit(1)
            loaded.pop("_mtime", None)
            progress = loaded

    completed_set = set(progress["completed"])
    generated = 0
    errors = 0

    print(f"Generating speech for {len(line_indices)} dialog lines via ElevenLabs TTS "
          f"({len(completed_set)} done)...")

    for i in line_indices:
        line = dialog[i]
        if i in completed_set:
            continue

        expected_path = manifest.speech_clip_path(speech_dir, i)
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

        line_strategy = strategy_for_line(i)
        result = line_strategy.generate(line["text"], speech_dir, i, 0, line["start_s"])
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


from stitch import stitch_clips  # noqa: E402,F811 -- re-export for backwards compat


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="lossy decoder: prompts -> video")
    parser.add_argument("output_dir", help="Output directory containing shots.json")
    parser.add_argument("--start-index", type=int, default=None,
                        help="Skip shots before this index (e.g., 10 to skip credits)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N shots (after start-index)")
    parser.add_argument("--stitch", action="store_true",
                        help="Only run the stitching step (skip generation)")
    parser.add_argument("--strategy", choices=VIDEO_STRATEGY_NAMES,
                        default="runpod-wan",
                        help="Video generation backend (default: runpod-wan)")
    parser.add_argument("--audio", action="store_true",
                        help="Generate audio clips (instead of video)")
    parser.add_argument("--audio-strategy",
                        choices=DECODE_AUDIO_STRATEGIES,
                        default=None,
                        help="Audio generation backend (default: auto-detect all for stitch, elevenlabs for generate)")
    parser.add_argument("--keep-pod", nargs="?", type=int, const=True, default=False,
                        metavar="MINUTES",
                        help="Keep the RunPod pod alive after decode, for a subsequent audio "
                             "stage (default 30 minutes; pass a number to change it). The pod "
                             "terminates itself once that elapses -- it never stays up "
                             "indefinitely, since a kept pod bills until terminated.")
    parser.add_argument("--speech", action="store_true",
                        help="Generate speech/dialogue clips (instead of video)")
    parser.add_argument("--speech-voice", default="Roger",
                        help="Fallback voice name for lines with no "
                             "per-character voice mapping (default: Roger). If "
                             "speakers.json/voice_map.json exist in output_dir "
                             "(see encode.py stage4 / voice_casting.py), "
                             "attributed lines use their mapped voice instead.")
    parser.add_argument("--speech-strategy", choices=["elevenlabs", "kokoro"],
                        default="elevenlabs",
                        help="TTS backend: elevenlabs (paid, fal.ai) or "
                             "kokoro (free, local, open-weight; requires the "
                             "optional kokoro extra). Voice names in "
                             "--speech-voice/voice_map.json are backend-specific.")
    parser.add_argument("--music-strategy",
                        choices=MUSIC_STRATEGIES,
                        default=None,
                        help="Music generation strategy to overlay on stitch (music-bucket shots only)")
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
            "musicgen": lambda: ReplicateMusicGenStrategy(),
            "runpod-musicgen": lambda: RunPodMusicGenStrategy(output_dir=args.output_dir),
        }
        audio_name = args.audio_strategy or "elevenlabs"
        audio_strategy = audio_strategies[audio_name]()
        run_audio(args, audio_strategy)
    elif args.speech:
        if args.speech_strategy == "kokoro":
            # "Roger" is an ElevenLabs name; only honor an explicit --speech-voice
            # override, otherwise fall back to Kokoro's own default voice.
            voice = args.speech_voice if args.speech_voice != "Roger" else "af_heart"
            speech_strategy = KokoroSpeechStrategy(voice=voice)
        else:
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
            "runpod-wan22": lambda: _create_wan22_strategy(args),
            "runpod-wan-enriched": lambda: _create_wan_enriched_strategy(args),
            "runpod-vace": lambda: _create_vace_strategy(args),
            "runpod-ltx2": lambda: _create_ltx2_strategy(args),
        }
        strategy = strategies[args.strategy]()
        run_decode(args, strategy)


def _create_wan_enriched_strategy(args):
    output_dir = args.output_dir
    characters_path = manifest.characters_path(output_dir)

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


def _create_ltx2_strategy(args):
    output_dir = args.output_dir
    characters_path = manifest.characters_path(output_dir)

    character_shot_map = {}
    characters_data = {}

    if os.path.exists(characters_path):
        with open(characters_path) as f:
            characters_data = json.load(f)
        character_shot_map = build_character_shot_map(characters_data)
    else:
        print("Warning: characters.json not found. Running without prompt enrichment.")

    return RunPodLtx2Strategy(
        output_dir=output_dir,
        keep_pod=getattr(args, "keep_pod", False),
        character_shot_map=character_shot_map,
        characters_data=characters_data,
    )


def _create_wan22_strategy(args):
    output_dir = args.output_dir
    characters_path = manifest.characters_path(output_dir)

    character_shot_map = {}
    characters_data = {}

    # Character prompt enrichment is on by default when characters.json exists.
    # Set LOSSY_NO_ENRICH=1 to decode from the raw shot descriptions instead.
    enrich = os.environ.get("LOSSY_NO_ENRICH", "").lower() not in ("1", "true", "yes", "on")

    if enrich and os.path.exists(characters_path):
        with open(characters_path) as f:
            characters_data = json.load(f)
        character_shot_map = build_character_shot_map(characters_data)
    elif not enrich:
        print("Character enrichment disabled (LOSSY_NO_ENRICH). Using raw descriptions.")
    else:
        print("Warning: characters.json not found. Running without prompt enrichment.")

    return RunPodWan22Strategy(
        output_dir=output_dir,
        keep_pod=getattr(args, "keep_pod", False),
        concurrent_audio=getattr(args, "concurrent_audio", False),
        character_shot_map=character_shot_map,
        characters_data=characters_data,
    )


def _create_vace_strategy(args):
    output_dir = args.output_dir
    characters_path = manifest.characters_path(output_dir)

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
