"""FFmpeg stitching for the lossy decoder.

Handles concatenating generated video clips into a final reconstructed film,
with optional audio (SFX) and speech track muxing. All operations use FFmpeg
subprocess calls.
"""

import json
import os
import subprocess
import sys


def _stitch_audio(output_dir: str, audio_strategy: str, prompts: list[dict],
                  start_index: int | None) -> str | None:
    """Concatenate per-shot audio clips into a single audio track.

    Returns path to the combined audio file, or None if no audio clips exist.
    Audio clips are duration-adjusted to match original shot durations using
    FFmpeg's atrim/apad filters.
    """
    audio_dir = os.path.join(output_dir, "audio", audio_strategy)
    if not os.path.exists(audio_dir):
        return None

    # Load audio progress for clip metadata
    progress_path = os.path.join(output_dir, f"audio_progress_{audio_strategy}.json")
    audio_meta = {}
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            audio_meta = json.load(f).get("clips", {})

    if start_index is not None:
        prompts = [p for p in prompts if p["index"] >= start_index]

    # Build per-shot audio files, adjusting duration to match original
    adjusted_dir = os.path.join(output_dir, "audio_adjusted", audio_strategy)
    os.makedirs(adjusted_dir, exist_ok=True)

    audio_entries = []
    for entry in prompts:
        idx = entry["index"]
        idx_str = str(idx)
        target_duration = entry["duration_s"]

        if idx_str in audio_meta:
            clips = audio_meta[idx_str]
        else:
            # Try to find clips on disk by convention
            clips = []
            for ext in (".mp3", ".flac"):
                path = os.path.join(audio_dir, f"{idx:04d}{ext}")
                if os.path.exists(path):
                    clips = [{"path": f"{idx:04d}{ext}", "duration_s": target_duration}]
                    break

        if not clips:
            # Generate silence for this shot
            silence_path = os.path.join(adjusted_dir, f"{idx:04d}.wav")
            if not os.path.exists(silence_path):
                subprocess.run(
                    ["ffmpeg", "-y", "-f", "lavfi", "-i",
                     f"anullsrc=r=44100:cl=stereo",
                     "-t", str(target_duration),
                     silence_path],
                    capture_output=True,
                )
            audio_entries.append(silence_path)
            continue

        if len(clips) == 1:
            # Single clip -- trim or pad to match target duration
            clip_path = os.path.join(audio_dir, clips[0]["path"])
            adjusted_path = os.path.join(adjusted_dir, f"{idx:04d}.wav")
            # Regenerate if source clip is newer than adjusted (stale cache)
            needs_regen = not os.path.exists(adjusted_path) or (
                os.path.getmtime(clip_path) > os.path.getmtime(adjusted_path))
            if needs_regen:
                subprocess.run(
                    ["ffmpeg", "-y", "-i", clip_path,
                     "-af", f"apad=whole_dur={target_duration},atrim=0:{target_duration}",
                     "-ar", "44100", "-ac", "2",
                     adjusted_path],
                    capture_output=True,
                )
            audio_entries.append(adjusted_path)
        else:
            # Multiple clips (split shot) -- concatenate parts, then adjust
            parts_file = os.path.join(adjusted_dir, f"{idx:04d}_parts.txt")
            with open(parts_file, "w") as f:
                for clip_info in clips:
                    clip_path = os.path.join(audio_dir, clip_info["path"])
                    f.write(f"file '{os.path.abspath(clip_path)}'\n")

            concat_path = os.path.join(adjusted_dir, f"{idx:04d}_concat.wav")
            adjusted_path = os.path.join(adjusted_dir, f"{idx:04d}.wav")
            newest_source = max(
                os.path.getmtime(os.path.join(audio_dir, c["path"])) for c in clips
            )
            needs_regen = not os.path.exists(adjusted_path) or (
                newest_source > os.path.getmtime(adjusted_path))
            if needs_regen:
                subprocess.run(
                    ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                     "-i", parts_file, "-ar", "44100", "-ac", "2",
                     concat_path],
                    capture_output=True,
                )
                subprocess.run(
                    ["ffmpeg", "-y", "-i", concat_path,
                     "-af", f"apad=whole_dur={target_duration},atrim=0:{target_duration}",
                     adjusted_path],
                    capture_output=True,
                )
                if os.path.exists(concat_path):
                    os.remove(concat_path)

            audio_entries.append(adjusted_path)

    if not audio_entries:
        return None

    # Concatenate all adjusted audio into one track
    concat_file = os.path.join(output_dir, f"audio_concat_{audio_strategy}.txt")
    with open(concat_file, "w") as f:
        for path in audio_entries:
            f.write(f"file '{os.path.abspath(path)}'\n")

    audio_track_path = os.path.join(output_dir, f"audio_track_{audio_strategy}.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
         "-i", concat_file, "-c", "copy",
         audio_track_path],
        capture_output=True,
    )

    print(f"  Audio track: {audio_track_path}")
    return audio_track_path


def _stitch_speech(output_dir: str, prompts: list[dict],
                   start_index: int | None) -> str | None:
    """Build time-aligned speech track from per-line TTS clips.

    Each speech clip is placed at its SRT-derived offset within the shot
    using FFmpeg adelay, then all shots are concatenated.
    """
    speech_dir = os.path.join(output_dir, "speech")
    if not os.path.exists(speech_dir):
        return None

    progress_path = os.path.join(output_dir, "speech_progress.json")
    speech_meta = {}
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            speech_meta = json.load(f).get("clips", {})

    if start_index is not None:
        prompts = [p for p in prompts if p["index"] >= start_index]

    adjusted_dir = os.path.join(output_dir, "speech_adjusted")
    os.makedirs(adjusted_dir, exist_ok=True)

    speech_entries = []
    for entry in prompts:
        idx = entry["index"]
        idx_str = str(idx)
        target_duration = entry["duration_s"]
        adjusted_path = os.path.join(adjusted_dir, f"{idx:04d}.wav")

        clips = speech_meta.get(idx_str, [])
        if not clips:
            # Silence for this shot
            if not os.path.exists(adjusted_path):
                subprocess.run(
                    ["ffmpeg", "-y", "-f", "lavfi", "-i",
                     f"anullsrc=r=44100:cl=stereo",
                     "-t", str(target_duration), adjusted_path],
                    capture_output=True,
                )
            speech_entries.append(adjusted_path)
            continue

        if not os.path.exists(adjusted_path):
            if len(clips) == 1:
                # Single line: adelay + pad/trim
                clip = clips[0]
                clip_path = os.path.join(speech_dir, clip["path"])
                delay_ms = int(clip["offset_s"] * 1000)
                subprocess.run(
                    ["ffmpeg", "-y", "-i", clip_path,
                     "-af", (f"adelay={delay_ms}|{delay_ms},"
                             f"apad=whole_dur={target_duration},"
                             f"atrim=0:{target_duration}"),
                     "-ar", "44100", "-ac", "2", adjusted_path],
                    capture_output=True,
                )
            else:
                # Multiple lines: adelay each, amix together, pad/trim
                inputs = []
                filters = []
                for i, clip in enumerate(clips):
                    clip_path = os.path.join(speech_dir, clip["path"])
                    inputs.extend(["-i", clip_path])
                    delay_ms = int(clip["offset_s"] * 1000)
                    filters.append(f"[{i}]adelay={delay_ms}|{delay_ms}[d{i}]")

                mix_inputs = "".join(f"[d{i}]" for i in range(len(clips)))
                filters.append(
                    f"{mix_inputs}amix=inputs={len(clips)}:duration=longest,"
                    f"apad=whole_dur={target_duration},"
                    f"atrim=0:{target_duration}[out]"
                )
                filter_complex = ";".join(filters)

                subprocess.run(
                    ["ffmpeg", "-y"] + inputs +
                    ["-filter_complex", filter_complex,
                     "-map", "[out]",
                     "-ar", "44100", "-ac", "2", adjusted_path],
                    capture_output=True,
                )

        speech_entries.append(adjusted_path)

    if not speech_entries:
        return None

    concat_file = os.path.join(output_dir, "speech_concat.txt")
    with open(concat_file, "w") as f:
        for path in speech_entries:
            f.write(f"file '{os.path.abspath(path)}'\n")

    speech_track_path = os.path.join(output_dir, "speech_track.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
         "-i", concat_file, "-c", "copy", speech_track_path],
        capture_output=True,
    )

    print(f"  Speech track: {speech_track_path}")
    return speech_track_path


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
    strategy_name = args.strategy
    clips_dir = os.path.join(output_dir, "clips", strategy_name)
    prompts_path = os.path.join(output_dir, "prompts.json")

    if not os.path.exists(clips_dir):
        # Fall back to flat clips/ for backwards compat with old runs
        clips_dir_flat = os.path.join(output_dir, "clips")
        if os.path.exists(clips_dir_flat):
            clips_dir = clips_dir_flat
        else:
            print(f"Error: {clips_dir} not found. Run decode first.")
            sys.exit(1)

    with open(prompts_path) as f:
        prompts_full = json.load(f)

    prompts = prompts_full
    if args.start_index:
        prompts = [p for p in prompts if p["index"] >= args.start_index]

    # Load clip metadata from progress (try per-strategy, fall back to legacy)
    progress_path = os.path.join(output_dir, f"decode_progress_{strategy_name}.json")
    if not os.path.exists(progress_path):
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
    adjusted_dir = os.path.join(output_dir, "adjusted", strategy_name)
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

    # Write concat list file (per-strategy to avoid overwriting)
    concat_file = os.path.join(output_dir, f"concat_{strategy_name}.txt")
    with open(concat_file, "w") as f:
        for path in concat_list:
            f.write(f"file '{os.path.abspath(path)}'\n")

    # Concatenate — include audio strategy in filename to avoid overwriting
    audio_strategy = getattr(args, "audio_strategy", None)
    if audio_strategy:
        output_name = f"reconstructed_{strategy_name}+{audio_strategy}.mp4"
    else:
        output_name = f"reconstructed_{strategy_name}.mp4"
    output_path = os.path.join(output_dir, output_name)
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

    # Mux audio if available (SFX and/or speech)
    audio_strategy = getattr(args, "audio_strategy", None)
    speech_voice = getattr(args, "speech_voice", None)

    if audio_strategy or speech_voice:
        # Compute actual adjusted video durations per shot (may differ from
        # original target due to setpts rounding). Audio must match these
        # exactly to stay in sync.
        shot_durations = {}  # index -> actual adjusted video duration
        for adjusted_path in concat_list:
            basename = os.path.splitext(os.path.basename(adjusted_path))[0]
            idx_str = basename.split("-")[0]
            idx = int(idx_str)
            dur = _probe_duration(adjusted_path)
            shot_durations[idx] = shot_durations.get(idx, 0) + dur

        stitched_prompts = []
        for p in prompts_full:
            if p["index"] in shot_durations:
                # Override duration with actual video duration for sync
                patched = dict(p)
                patched["duration_s"] = shot_durations[p["index"]]
                stitched_prompts.append(patched)

        # Build SFX track
        audio_track = None
        if audio_strategy:
            audio_track = _stitch_audio(output_dir, audio_strategy, stitched_prompts, None)

        # Build speech track
        speech_track = None
        if speech_voice:
            speech_track = _stitch_speech(output_dir, stitched_prompts, None)

        # Determine final audio to mux
        audio_to_mux = None
        if audio_track and os.path.exists(audio_track) and speech_track and os.path.exists(speech_track):
            # Mix SFX + speech into combined track
            combined_path = os.path.join(output_dir, f"combined_audio_{audio_strategy}.wav")
            subprocess.run(
                ["ffmpeg", "-y",
                 "-i", audio_track, "-i", speech_track,
                 "-filter_complex", "amix=inputs=2:duration=longest",
                 combined_path],
                capture_output=True,
            )
            audio_to_mux = combined_path
        elif audio_track and os.path.exists(audio_track):
            audio_to_mux = audio_track
        elif speech_track and os.path.exists(speech_track):
            audio_to_mux = speech_track

        if audio_to_mux:
            muxed_path = output_path.replace(".mp4", "_with_audio.mp4")
            result = subprocess.run(
                ["ffmpeg", "-y",
                 "-i", output_path,
                 "-i", audio_to_mux,
                 "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                 "-shortest",
                 muxed_path],
                capture_output=True,
            )
            if result.returncode == 0:
                os.replace(muxed_path, output_path)
                label = "SFX+speech" if audio_track and speech_track else ("speech" if speech_track else "SFX")
                print(f"  {label} muxed into {output_path}")
            else:
                print(f"  Warning: audio mux failed, silent video preserved")
                if os.path.exists(muxed_path):
                    os.remove(muxed_path)
