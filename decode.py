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

from keyframes import KeyframeSetupError  # noqa: E402
from runpod_pod import GatedModelError, check_gated_access  # noqa: E402

from config import (  # noqa: E402 -- re-export for backwards compat
    DECODE_AUDIO_STRATEGIES,
    MUSIC_STRATEGIES,
    VIDEO_STRATEGIES,
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
    RunPodLtx25Strategy,
    RunPodLtx25IngredientsStrategy,
    RunPodLtx25KeyframeStrategy,
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
    # A shot that failed on an *earlier* run and succeeds on this one must stop
    # being reported as failed. Only the in-run retry path used to clear this,
    # so a resumed run accumulated shots sitting in both lists at once -- the
    # full run's first checkpoint showed shots 10-14 completed and failed
    # simultaneously, which makes the failed list useless as a QC signal over
    # 2069 shots. Clearing it here covers every success path.
    progress["failed"] = [f for f in progress["failed"] if f != idx]
    progress["total_cost_estimate"] += sum(r.cost for r in results)
    progress["clips"][str(idx)] = [
        {"path": os.path.basename(r.path), "duration_s": r.actual_duration_s}
        for r in results
    ]


# ---------------------------------------------------------------------------
# Decode loop
# ---------------------------------------------------------------------------


def run_decode(args, strategy: GenerationStrategy) -> list[int]:
    """Main decode loop: read prompts, generate clips, track progress.

    Returns the indices (within the --start-index/--limit scope) of shots that
    are still not completed: failed, or never reached because the loop stopped
    on too many errors. ``main`` exits nonzero when this is non-empty, so a
    supervisor (``tools/capped_run.py``) can tell an incomplete decode from a
    finished one.

    Retry budget: ``strategy.generate`` retries each *clip* internally (RunPod
    strategies tolerate two wedges per clip), and this loop then retries the
    whole *shot* once more after 10s. A shot that keeps wedging can therefore
    cost up to 2 x the per-clip budget before it is recorded failed. Every
    nonzero exit re-runs failed shots again under ``capped_run`` (up to
    ``--retries`` times), each paying that budget again.

    Resume: only shots with every part on disk are adopted; a partial split
    shot is regenerated, overwriting its stray parts. Adopting a shot clears it
    from ``failed``, and a shot appears in ``failed`` at most once.
    """
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

    # Inputs a strategy makes ahead of the video pass (keyframes). Runs before any
    # pod exists, so a missing prerequisite stops the run before anything bills.
    strategy.prepare([p for p in prompts if p["index"] not in completed_set])

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
            progress["failed"] = [f for f in progress["failed"] if f != idx]
            continue

        prompt_text = strategy.format_prompt(entry)
        print(f"  Shot {idx} ({generated + 1}/{total - len(completed_set)} remaining)...")

        results = strategy.generate(prompt_text, clips_dir, idx, entry["duration_s"], seed=idx, entry=entry)

        if results:
            generated += 1
            _record_success(progress, completed_set, idx, results)
        else:
            errors += 1
            if idx not in progress["failed"]:  # a resume re-fails a shot already listed
                progress["failed"].append(idx)
            # Retry once after a short wait
            print(f"  Retrying shot {idx} in 10s...")
            time.sleep(10)
            results = strategy.generate(prompt_text, clips_dir, idx, entry["duration_s"], seed=idx, entry=entry)
            if results:
                generated += 1
                _record_success(progress, completed_set, idx, results)

        # Save progress every 5 clips
        if generated % 5 == 0:
            manifest.write_json_atomic(progress_path, progress)

        if errors > 20:
            print("Too many errors, saving progress and stopping.")
            break

    # Finish pipelined audio (last shot + wait for thread)
    strategy.finish_audio()

    # Final save
    manifest.write_json_atomic(progress_path, progress)

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
        manifest.write_json_atomic(audio_progress_path, audio_progress)
        print(f"  Pipelined audio: {len(audio_results)} completed, {len(audio_failures)} failed")

    # Signal clean exit for --keep-pod support
    strategy.mark_clean_exit()

    print(f"\nDone. Generated {generated} clips.")
    print(f"  Total: {len(progress['completed'])} completed, {len(progress['failed'])} failed")
    print(f"  Estimated cost: ${progress['total_cost_estimate']:.2f}")

    incomplete = [p["index"] for p in prompts if p["index"] not in completed_set]
    if incomplete:
        print(f"  INCOMPLETE: {len(incomplete)} shots not decoded: {incomplete[:20]}"
              f"{' ...' if len(incomplete) > 20 else ''}")
    return incomplete


# ---------------------------------------------------------------------------
# A/B manifest runner (research 0028)
# ---------------------------------------------------------------------------

AB_JOB_KEYS = ("shot", "arm", "seed", "frames", "prompt")


def load_ab_jobs(path: str) -> list[dict]:
    """Read an A/B manifest (``{"jobs": [...]}``) and check every job is runnable."""
    with open(path) as f:
        raw = json.load(f)
    jobs = raw.get("jobs") if isinstance(raw, dict) else None
    if not jobs:
        raise ValueError(f"{path}: expected {{'jobs': [...]}} with at least one job")
    seen = set()
    for i, job in enumerate(jobs):
        missing = [k for k in AB_JOB_KEYS if k not in job]
        if missing:
            raise ValueError(f"{path}: job {i} is missing {missing}")
        key = ab_job_key(job)
        if key in seen:
            raise ValueError(f"{path}: job {i} repeats {key}")
        seen.add(key)
        if job.get("start_image"):
            # Optional I2V first frame (research 0029's keyframe arm), relative
            # to the manifest. Checked here so a missing keyframe fails before
            # a pod is paid for, not at that job's turn.
            img = os.path.join(os.path.dirname(os.path.abspath(path)), job["start_image"])
            if not os.path.isfile(img):
                raise ValueError(f"{path}: job {i} start_image not found: {img}")
            job["start_image"] = img
    return jobs


def ab_job_key(job: dict) -> str:
    return f"{job['arm']}/{job['shot']}"


def dry_run_ab(strategy_cls, jobs: list[dict]) -> None:
    """Print the jobs and build each one's ComfyUI graph. No pod, no network, no spend.

    Building the graph is what rejects a bad frame count or a malformed prompt,
    so a clean dry run means the real run will not fail at submit.
    """
    probe = object.__new__(strategy_cls)  # skip __init__: it creates a RunPod session
    for job in jobs:
        start = os.path.basename(job["start_image"]) if job.get("start_image") else None
        probe._build_workflow(job["prompt"], job["seed"], length=job["frames"], start_image=start)
        print(f"  {ab_job_key(job):<28} seed {job['seed']:<5} {job['frames']:>3} frames  "
              f"{'img ' if start else ''}{job['prompt'][:60]}...")
    print(f"Dry run OK: {len(jobs)} jobs build valid workflows. Nothing was submitted.")
    # Informational: a real run stops on this before any pod is started. One free HEAD request.
    gated = [m for m in getattr(strategy_cls, "PINNED_MODELS", []) if m.gate_url]
    if gated:
        try:
            check_gated_access(gated, os.environ.get("HF_TOKEN"))
            print(f"Gate check OK: HF_TOKEN may download {', '.join(os.path.basename(m.dest) for m in gated)}.")
        except GatedModelError as e:
            print(f"Gate check FAILED (a real run stops here, before any pod): {e}")


def run_ab_manifest(args, strategy: GenerationStrategy, jobs: list[dict]):
    """Render an A/B manifest's jobs: each prompt verbatim at its own seed and frame count.

    Bypasses ``strategy.generate`` and ``format_prompt`` on purpose: the manifest
    carries the exact prompt per arm, and the arms are only comparable if nothing
    rewrites them. Clips land in one directory per arm (``manifest.ab_clips_dir``)
    because arms share shot indices. Progress is saved after every clip, so
    ``tools/capped_run.py`` can restart this after a pod death and pick up where
    it stopped.
    """
    output_dir = args.output_dir
    progress_path = manifest.ab_progress_path(output_dir, strategy.name)
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            progress = json.load(f)
    else:
        progress = {"completed": [], "failed": [], "total_cost_estimate": 0.0}
    done = set(progress["completed"])

    def save():
        manifest.write_json_atomic(progress_path, progress)

    print(f"A/B manifest: {len(jobs)} jobs via {strategy.name} ({len(done)} already done)...")
    for job in jobs:
        key = ab_job_key(job)
        clips_dir = manifest.ab_clips_dir(output_dir, strategy.name, job["arm"])
        clip_name = manifest.clip_filename(job["shot"], None, ".mp4")
        if key in done:
            continue
        if os.path.exists(os.path.join(clips_dir, clip_name)):
            done.add(key)
            progress["completed"].append(key)
            continue
        os.makedirs(clips_dir, exist_ok=True)
        print(f"  {key} (seed {job['seed']}, {job['frames']} frames)...")
        strategy._ensure_pod()
        start_image = strategy._upload_image(job["start_image"]) if job.get("start_image") else None
        if job.get("start_image") and start_image is None:
            # An arm defined by its first frame must not quietly render text-only
            # when the upload fails: that would score as the arm but be the control.
            print(f"  {key}: start image upload failed; marking failed")
            result = None
        else:
            result = strategy._generate_one_clip(
                job["prompt"], clips_dir, clip_name, job["frames"], job["seed"],
                start_image=start_image)
        if result is None:
            if key not in progress["failed"]:
                progress["failed"].append(key)
            save()
            continue
        done.add(key)
        progress["completed"].append(key)
        progress["failed"] = [k for k in progress["failed"] if k != key]
        progress["total_cost_estimate"] += result.cost
        save()

    save()
    strategy.mark_clean_exit()
    print(f"\nDone. {len(progress['completed'])}/{len(jobs)} completed, "
          f"{len(progress['failed'])} failed. Clips under "
          f"{manifest.ab_clips_dir(output_dir, strategy.name, '<arm>')}")
    if progress["failed"]:
        sys.exit(1)


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

        # An all-speech description has nothing for an ambience model to make;
        # the strategy would return [] and the loop would count it as a failure.
        if strategy.filters_speech and filter_speech_from_sound(sound) is None:
            skipped += 1
            progress["skipped"].append(idx)
            skipped_set.add(idx)
            if idx in progress["failed"]:
                progress["failed"].remove(idx)
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
            if idx not in progress["failed"]:  # a resume re-fails a shot already listed
                progress["failed"].append(idx)
            print(f"  Retrying shot {idx} in 10s...")
            time.sleep(10)
            results = strategy.generate(sound, audio_dir, idx, entry["duration_s"], seed=idx)
            if results:
                generated += 1
                _record_success(progress, completed_set, idx, results)

        # Save progress every 5 clips
        if generated % 5 == 0:
            manifest.write_json_atomic(progress_path, progress)

        if errors > 20:
            print("Too many errors, saving progress and stopping.")
            break

    # Final save
    manifest.write_json_atomic(progress_path, progress)

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
            manifest.write_json_atomic(progress_path, progress)
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

        manifest.write_json_atomic(progress_path, progress)

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
    parser.add_argument("--style-prefix", default=None,
                        help="One sentence placed at the front of every video prompt, ahead of the "
                             "identity blocks, to anchor the film's look (e.g. 'Live-action 1990s "
                             "sitcom footage of real actors, photographed on 35mm film.'). Default: "
                             "the optional top-level style_prefix in shots.json. LTX strategies only.")
    parser.add_argument("--ab-manifest", default=None, metavar="JSON",
                        help="Render the jobs of an A/B manifest (e.g. "
                             "docs/research/0028-style-drift/ab_manifest.json) instead of the shot "
                             "list: each job's prompt verbatim at its own seed and frame count, "
                             "clips under clips/<strategy>-ab/<arm>/. LTX strategies only.")
    parser.add_argument("--dry-run", action="store_true",
                        help="With --ab-manifest: validate the jobs and build their workflows, "
                             "then exit. With --strategy runpod-ltx25-keyframe: print which shots "
                             "get a keyframe and which fall back to text. No pod is created, no "
                             "network call is made and nothing is spent.")
    parser.add_argument("--keyframe-fal-cap", type=float, default=RunPodLtx25KeyframeStrategy.FAL_CAP_USD,
                        metavar="USD",
                        help="runpod-ltx25-keyframe: hard cap on the estimated fal spend (sheets and "
                             "keyframes) in one run; shots past it decode from text (default: %(default)s). "
                             "tools/capped_run.py caps RunPod only, so this is the fal cap. Spend "
                             "already in <output_dir>/keyframes/cost.jsonl counts against it.")
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
            "runpod-ltx25": lambda: _create_ltx2_strategy(args, RunPodLtx25Strategy),
            "runpod-ltx25-keyframe": lambda: _create_keyframe_strategy(args),
            "runpod-ltx25-ingredients": lambda: _create_ltx2_strategy(args, RunPodLtx25IngredientsStrategy),
        }
        if args.ab_manifest:
            _run_ab(args, strategies)
            return
        if VIDEO_STRATEGIES[args.strategy].ab_manifest_only:
            sys.exit(f"{args.strategy} only runs with --ab-manifest (its inputs are per clip).")
        if args.dry_run:
            if args.strategy == RunPodLtx25KeyframeStrategy.name:
                dry_run_keyframes(args)
            elif issubclass(VIDEO_STRATEGIES[args.strategy].cls, RunPodLtx2Strategy):
                dry_run_shot_plan(args)
            else:
                sys.exit("--dry-run needs --ab-manifest or an LTX strategy "
                         "(it would otherwise start a real decode).")
            return
        strategy = strategies[args.strategy]()
        try:
            incomplete = run_decode(args, strategy)
        except KeyframeSetupError as e:   # raised by prepare(), before any pod exists
            sys.exit(f"Error: {e}")
        if incomplete:
            sys.exit(1)  # shots remain failed/undecoded: not a finished film


def dry_run_shot_plan(args) -> None:
    """Build every remaining shot's prompts and ComfyUI graphs, as a real decode
    would, and print the totals. No pod, no network, nothing written."""
    try:
        prompts, _ = manifest.load_shots(args.output_dir)
    except FileNotFoundError:
        sys.exit(f"Error: {manifest.shots_path(args.output_dir)} not found. Run encoder first.")
    if args.start_index:
        prompts = [p for p in prompts if p["index"] >= args.start_index]
    if args.limit:
        prompts = prompts[:args.limit]
    strategy_cls = VIDEO_STRATEGIES[args.strategy].cls
    done = set()
    progress_path = manifest.decode_progress_path(args.output_dir, strategy_cls.name)
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            done = set(json.load(f).get("completed", []))
    characters_data = {}
    if os.path.exists(manifest.characters_path(args.output_dir)):
        with open(manifest.characters_path(args.output_dir)) as f:
            characters_data = json.load(f)
    style = getattr(args, "style_prefix", None) or manifest.load_style_prefix(args.output_dir)
    probe = object.__new__(strategy_cls)   # skip __init__: it creates a RunPod session
    probe._init_character_identity(build_character_shot_map(characters_data), characters_data, style)
    todo = [p for p in prompts if p["index"] not in done]
    parts = frames = 0
    for entry in todo:
        counts = probe._target_durations(entry["duration_s"])
        text = probe.format_prompt(entry)
        for i, n in enumerate(counts):   # building the graph is what rejects a bad frame count
            probe._build_workflow(probe._part_prompt(text, entry, i, len(counts)),
                                  entry["index"] + i, length=n, start_image="chain.png" if i else None)
        parts += len(counts)
        frames += sum(counts)
    print(f"Shot plan for {args.output_dir} via {strategy_cls.name} (dry run):")
    print(f"  {len(prompts)} shots in scope, {len(prompts) - len(todo)} already done, "
          f"{len(todo)} to generate as {parts} clips ({frames / probe.FPS / 60:.1f} min of video).")
    print(f"  Style prefix: {style or '(none)'}; characters: "
          f"{len(characters_data.get('characters', [])) or 'none (no identity blocks)'}")
    if todo:
        print(f"  First prompt (shot {todo[0]['index']}): {probe.format_prompt(todo[0])[:400]}")
    print("Dry run OK. No network call was made and nothing was written.")


def dry_run_keyframes(args) -> None:
    """Print which shots would get a keyframe and which fall back to text. No network, no files written."""
    try:
        prompts, _ = manifest.load_shots(args.output_dir)
    except FileNotFoundError:
        sys.exit(f"Error: {manifest.shots_path(args.output_dir)} not found. Run encoder first.")
    if args.start_index:
        prompts = [p for p in prompts if p["index"] >= args.start_index]
    if args.limit:
        prompts = prompts[:args.limit]
    characters_data = {}
    if os.path.exists(manifest.characters_path(args.output_dir)):
        with open(manifest.characters_path(args.output_dir)) as f:
            characters_data = json.load(f)
    probe = object.__new__(RunPodLtx25KeyframeStrategy)   # skip __init__: it creates a RunPod session
    probe._init_character_identity(
        build_character_shot_map(characters_data), characters_data,
        getattr(args, "style_prefix", None) or manifest.load_style_prefix(args.output_dir))
    probe._fal_cap_usd = args.keyframe_fal_cap
    print(f"Keyframe plan for {len(prompts)} shots of {args.output_dir} (dry run):")
    print("\n".join(probe.describe_plan(prompts)))
    print("Dry run OK. No network call was made and nothing was written.")


def _run_ab(args, strategies: dict):
    """Dispatch --ab-manifest: validate, then dry-run or render."""
    strategy_cls = VIDEO_STRATEGIES[args.strategy].cls
    if not issubclass(strategy_cls, RunPodLtx2Strategy):
        sys.exit("--ab-manifest supports the LTX strategies only "
                 "(runpod-ltx2, runpod-ltx25, runpod-ltx25-keyframe, runpod-ltx25-ingredients)")
    jobs = load_ab_jobs(args.ab_manifest)
    if args.dry_run:
        dry_run_ab(strategy_cls, jobs)
        return
    try:   # a gate failure is raised before any pod exists (strategy init) or ends the pod it hit
        run_ab_manifest(args, strategies[args.strategy](), jobs)
    except GatedModelError as e:
        sys.exit(f"Error: {e}")


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


def _create_keyframe_strategy(args):
    return _create_ltx2_strategy(args, RunPodLtx25KeyframeStrategy,
                                 fal_cap_usd=args.keyframe_fal_cap)


def _create_ltx2_strategy(args, strategy_cls=RunPodLtx2Strategy, **extra):
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

    style_prefix = getattr(args, "style_prefix", None) or manifest.load_style_prefix(output_dir)
    if style_prefix:
        print(f"Style prefix: {style_prefix}")

    return strategy_cls(
        output_dir=output_dir,
        keep_pod=getattr(args, "keep_pod", False),
        character_shot_map=character_shot_map,
        characters_data=characters_data,
        style_prefix=style_prefix,
        **extra,
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
