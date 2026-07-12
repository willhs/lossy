"""FFmpeg stitching for the lossy decoder.

Handles concatenating generated video clips into a final reconstructed film,
with optional audio (SFX) and speech track muxing. All operations use FFmpeg
subprocess calls.
"""

import json
import os
import subprocess
import sys

import manifest


def _stale_for_target(adjusted_path: str, target_duration: float) -> bool:
    """True if adjusted_path is missing or was built for a different target.

    Audio adjusted clips are cut to a target_duration derived from the
    (now shots.json-accurate) video's per-shot duration. Regen is normally
    keyed off source-clip mtime, but that misses the case where the video
    fix changes the *target* without the audio source clip itself changing
    -- without this check, stale audio_adjusted/ files silently keep chasing
    a pre-fix duration.
    """
    if not os.path.exists(adjusted_path):
        return True
    return abs(manifest.probe_duration(adjusted_path) - target_duration) > 1 / 24


def _stitch_audio(output_dir: str, audio_strategy: str, prompts: list[dict],
                  start_index: int | None) -> str | None:
    """Concatenate per-shot audio clips into a single audio track.

    Returns path to the combined audio file, or None if no audio clips exist.
    Audio clips are duration-adjusted to match original shot durations using
    FFmpeg's atrim/apad filters.
    """
    audio_dir = manifest.audio_dir(output_dir, audio_strategy)
    if not os.path.exists(audio_dir):
        return None

    # Load audio progress for clip metadata
    progress_path = manifest.audio_progress_path(output_dir, audio_strategy)
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
                filename = manifest.audio_clip_filename(idx, None, ext)
                if os.path.exists(os.path.join(audio_dir, filename)):
                    clips = [{"path": filename, "duration_s": target_duration}]
                    break

        if not clips:
            # Generate silence for this shot
            silence_path = os.path.join(adjusted_dir, f"{idx:04d}.wav")
            if _stale_for_target(silence_path, target_duration):
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
            # Regenerate if source clip is newer than adjusted (stale cache),
            # or the target duration itself changed since adjusted was built.
            needs_regen = _stale_for_target(adjusted_path, target_duration) or (
                os.path.exists(adjusted_path)
                and os.path.getmtime(clip_path) > os.path.getmtime(adjusted_path))
            if needs_regen:
                subprocess.run(
                    ["ffmpeg", "-y", "-i", clip_path,
                     "-af", f"apad=whole_dur={target_duration},atrim=0:{target_duration}",
                     "-ar", "44100", "-ac", "2",
                     "-t", str(target_duration),
                     adjusted_path],
                    capture_output=True,
                )
            audio_entries.append(adjusted_path)
        else:
            # Multiple clips (split shot) -- crossfade parts, then adjust
            adjusted_path = os.path.join(adjusted_dir, f"{idx:04d}.wav")
            newest_source = max(
                os.path.getmtime(os.path.join(audio_dir, c["path"])) for c in clips
            )
            needs_regen = _stale_for_target(adjusted_path, target_duration) or (
                os.path.exists(adjusted_path)
                and newest_source > os.path.getmtime(adjusted_path))
            if needs_regen:
                clip_paths = [
                    os.path.join(audio_dir, c["path"]) for c in clips
                ]
                crossfade_s = 0.5
                # Build acrossfade filter chain:
                # For N clips, chain N-1 acrossfade filters.
                inputs = []
                for i, cp in enumerate(clip_paths):
                    inputs += ["-i", cp]
                if len(clip_paths) == 2:
                    af = (f"[0:a][1:a]acrossfade=d={crossfade_s}:c1=tri:c2=tri,"
                          f"apad=whole_dur={target_duration},atrim=0:{target_duration}")
                else:
                    # Chain: first pair -> [tmp1], then [tmp1][2:a] -> [tmp2], etc.
                    parts = []
                    for i in range(len(clip_paths) - 1):
                        left = f"[{i}:a]" if i == 0 else f"[tmp{i}]"
                        right = f"[{i + 1}:a]"
                        out = f"[tmp{i + 1}]" if i < len(clip_paths) - 2 else ""
                        parts.append(
                            f"{left}{right}acrossfade=d={crossfade_s}:c1=tri:c2=tri{out}"
                        )
                    af = (";".join(parts)
                          + f",apad=whole_dur={target_duration},atrim=0:{target_duration}")
                subprocess.run(
                    ["ffmpeg", "-y"] + inputs +
                    ["-filter_complex", af,
                     "-ar", "44100", "-ac", "2",
                     "-t", str(target_duration),
                     adjusted_path],
                    capture_output=True,
                )

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
    """Build a time-aligned speech track from per-line TTS clips.

    Each clip is placed at its global SRT timestamp using FFmpeg adelay and
    mixed into a single track matching the duration of the stitched shots.
    Clips outside the stitched window are excluded.
    """
    speech_dir = manifest.speech_dir(output_dir)
    if not os.path.exists(speech_dir):
        return None

    progress_path = manifest.speech_progress_path(output_dir)
    if not os.path.exists(progress_path):
        return None
    with open(progress_path) as f:
        speech_progress = json.load(f)

    try:
        _, global_dialog = manifest.load_shots(output_dir)
    except (FileNotFoundError, ValueError):
        return None

    return _stitch_speech_global(prompts, speech_dir, output_dir,
                                 speech_progress, global_dialog)


def _stitch_speech_global(shots: list[dict], speech_dir: str, output_dir: str,
                          speech_progress: dict, global_dialog: list[dict]) -> str | None:
    """v2: build speech track by placing clips at global SRT timestamps.

    All clips are mixed into a single continuous track using FFmpeg adelay.
    Clips outside the stitched window are excluded. Delays are relative to
    the start of the first stitched shot so the track aligns with the video.
    """
    if not shots:
        return None

    clips = speech_progress.get("clips", {})

    window_start = shots[0]["start_s"]
    total_duration = sum(p["duration_s"] for p in shots)
    window_end = window_start + total_duration

    # Iterate only generated clips (not all dialog lines) for efficiency
    active_clips = []
    for idx_str, clip_meta in sorted(clips.items(), key=lambda x: int(x[0])):
        i = int(idx_str)
        if i >= len(global_dialog):
            continue
        entry = global_dialog[i]
        if entry["start_s"] >= window_end or entry["end_s"] <= window_start:
            continue
        clip_path = os.path.join(speech_dir, clip_meta["path"])
        if not os.path.exists(clip_path):
            continue
        delay_ms = max(0, int((entry["start_s"] - window_start) * 1000))
        active_clips.append((clip_path, delay_ms))

    speech_track_path = os.path.join(output_dir, "speech_track.wav")

    if not active_clips:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
             "-t", str(total_duration), speech_track_path],
            capture_output=True,
        )
        print(f"  Speech track: {speech_track_path} (silence — no clips in window)")
        return speech_track_path

    if len(active_clips) == 1:
        clip_path, delay_ms = active_clips[0]
        subprocess.run(
            ["ffmpeg", "-y", "-i", clip_path,
             "-af", (f"adelay={delay_ms}|{delay_ms},"
                     f"apad=whole_dur={total_duration},"
                     f"atrim=0:{total_duration}"),
             "-ar", "44100", "-ac", "2", "-t", str(total_duration),
             speech_track_path],
            capture_output=True,
        )
    else:
        inputs = []
        filters = []
        for i, (clip_path, delay_ms) in enumerate(active_clips):
            inputs.extend(["-i", clip_path])
            filters.append(f"[{i}]adelay={delay_ms}|{delay_ms}[d{i}]")
        mix_inputs = "".join(f"[d{i}]" for i in range(len(active_clips)))
        filters.append(
            f"{mix_inputs}amix=inputs={len(active_clips)}:duration=longest:normalize=0,"
            f"apad=whole_dur={total_duration},"
            f"atrim=0:{total_duration}[out]"
        )
        subprocess.run(
            ["ffmpeg", "-y"] + inputs +
            ["-filter_complex", ";".join(filters),
             "-map", "[out]",
             "-ar", "44100", "-ac", "2", "-t", str(total_duration),
             speech_track_path],
            capture_output=True,
        )

    print(f"  Speech track: {speech_track_path}")
    return speech_track_path


def _probe_duration(clip_path: str) -> float:
    """Get video duration via ffprobe. Raises if ffprobe cannot parse the file."""
    return manifest.probe_duration(clip_path)


def _retime_clip(clip_path: str, adjusted_path: str, target_duration: float,
                 actual_duration: float) -> None:
    """Trim or slow a single clip to hit target_duration, writing to adjusted_path."""
    # Boundary-locked targets can dip to near-zero (or, in pathological
    # cases, negative) if prior shots ran long -- floor to one frame so
    # ffmpeg always gets a valid trim; any residual gets caught up by
    # later shots rather than producing a degenerate clip here.
    target_duration = max(target_duration, 1 / 24)
    speed_factor = target_duration / actual_duration if actual_duration > 0 else 1.0

    # A relative "close enough" threshold (e.g. 5%) leaves multiple frames
    # of residual on longer clips, which then compounds across shots under
    # boundary-locked retiming. Compare absolute difference against one
    # frame instead so every shot lands as close to its target as ffmpeg's
    # own frame-boundary trim precision allows.
    if abs(target_duration - actual_duration) < 1 / 24:
        subprocess.run(["cp", clip_path, adjusted_path], capture_output=True)
    elif speed_factor < 1.0:
        # Clip is longer than the target. Trim (cut off the tail) instead of
        # speeding it up -- avoids the fast-motion "squeeze" artifact on short
        # shots. Same output duration as the speed-up would give.
        subprocess.run(
            ["ffmpeg", "-i", clip_path, "-t", f"{target_duration}",
             "-an", "-y", adjusted_path],
            capture_output=True,
        )
    else:
        # Clip is shorter than the target: slow it down to fill the shot
        # (trimming can't lengthen a clip).
        subprocess.run(
            ["ffmpeg", "-i", clip_path, "-filter:v", f"setpts={speed_factor}*PTS",
             "-an", "-y", adjusted_path],
            capture_output=True,
        )


def _retime_shot_clips(clip_paths: list[str], original_duration: float,
                       adjusted_dir: str) -> list[str]:
    """Produce this shot's adjusted clip(s), matching original_duration.

    Single-part shots are trimmed/slowed as a whole (existing behavior).
    Split shots (multiple chained parts) are left untouched except for the
    last part: earlier parts keep their generated timing so I2V chaining
    continuity isn't disturbed, and only the last part is trimmed/slowed so
    the group's total lands within ~1 frame of shots.json's duration --
    correcting the frame-quantization/remainder-flooring overshoot that
    otherwise passes straight through split shots uncorrected.

    original_duration is boundary-locked (the caller passes how much time
    this shot needs to occupy to catch the running timeline back up to its
    true absolute end_s), so it can differ run-to-run even when the source
    clips haven't changed. Cache invalidation therefore checks the existing
    adjusted output's actual duration against the target, not just mtimes.
    """
    adjusted_paths = [
        os.path.join(adjusted_dir, f"{os.path.splitext(os.path.basename(p))[0]}.mp4")
        for p in clip_paths
    ]

    mtime_stale = any(
        not os.path.exists(ap) or os.path.getmtime(cp) > os.path.getmtime(ap)
        for cp, ap in zip(clip_paths, adjusted_paths)
    )
    target_stale = mtime_stale or (
        abs(sum(_probe_duration(p) for p in adjusted_paths) - original_duration) > 1 / 24
    )
    if not target_stale:
        return adjusted_paths

    if len(clip_paths) == 1:
        clip_path, adjusted_path = clip_paths[0], adjusted_paths[0]
        actual_duration = _probe_duration(clip_path)
        _retime_clip(clip_path, adjusted_path, original_duration, actual_duration)
        return adjusted_paths

    # Split shot: copy earlier parts through unchanged.
    for clip_path, adjusted_path in zip(clip_paths[:-1], adjusted_paths[:-1]):
        subprocess.run(["cp", clip_path, adjusted_path], capture_output=True)

    earlier_total = sum(_probe_duration(p) for p in adjusted_paths[:-1])
    last_clip_path, last_adjusted_path = clip_paths[-1], adjusted_paths[-1]
    remaining_budget = original_duration - earlier_total

    if remaining_budget <= 0:
        # Earlier parts alone already meet/exceed the shot's budget -- no
        # room left to give the last part. Copy it through unchanged rather
        # than produce a zero/negative-length clip.
        subprocess.run(["cp", last_clip_path, last_adjusted_path], capture_output=True)
    else:
        last_actual = _probe_duration(last_clip_path)
        _retime_clip(last_clip_path, last_adjusted_path, remaining_budget, last_actual)

    return adjusted_paths


def _stitch_range(output_dir: str, strategy_name: str, prompts_full: list[dict],
                  clips_meta: dict, clips_dir: str, adjusted_dir: str,
                  start_index: int | None, end_index: int | None,
                  output_path: str, audio_strategy: str | None,
                  speech_voice: str | None, music_strategy: str | None = None,
                  label: str = "film") -> str | None:
    """Stitch a range of clips into a single video with optional audio.

    Filters prompts to start_index <= index <= end_index, collects and
    speed-adjusts clips, concatenates via FFmpeg, and optionally muxes
    audio/speech tracks. Returns the output path, or None if no clips found.
    """
    # Filter prompts to range
    prompts = prompts_full
    if start_index is not None:
        prompts = [p for p in prompts if p["index"] >= start_index]
    if end_index is not None:
        prompts = [p for p in prompts if p["index"] <= end_index]

    # Collect existing clips in order, grouped by shot so a split shot's
    # parts can be corrected as a unit (see _retime_shot_clips).
    shot_groups = []  # list of (entry, [clip_path, ...])
    for entry in prompts:
        idx = entry["index"]
        idx_str = str(idx)

        if idx_str in clips_meta:
            paths = [
                os.path.join(clips_dir, clip_info["path"])
                for clip_info in clips_meta[idx_str]
            ]
            paths = [p for p in paths if os.path.exists(p)]
        else:
            single_path = manifest.clip_path(clips_dir, idx)
            paths = [single_path] if os.path.exists(single_path) else []

        if paths:
            shot_groups.append((entry, paths))

    if not shot_groups:
        print(f"  No clips found for {label}.")
        return None

    total_clips = sum(len(paths) for _, paths in shot_groups)
    print(f"Stitching {total_clips} clips ({len(shot_groups)} shots) for {label}...")

    # Speed-adjust each clip (reuses cached adjusted clips). Targets are
    # boundary-locked, not just duration-locked: each shot is retimed to
    # close the gap between the running actual position and this shot's
    # true absolute end_s, so small per-shot rounding residuals (ffmpeg's
    # -t trims to the nearest frame) get corrected out shot-by-shot instead
    # of silently accumulating across the whole scene.
    os.makedirs(adjusted_dir, exist_ok=True)

    concat_list = []
    running_pos = shot_groups[0][0]["start_s"]
    for entry, clip_paths in shot_groups:
        target_duration = entry["end_s"] - running_pos
        adjusted_paths = _retime_shot_clips(clip_paths, target_duration, adjusted_dir)
        concat_list.extend(adjusted_paths)
        running_pos += sum(_probe_duration(p) for p in adjusted_paths)

    # Write range-specific concat file
    range_suffix = ""
    if start_index is not None or end_index is not None:
        range_suffix = f"_{start_index or 'start'}-{end_index or 'end'}"
    concat_file = os.path.join(output_dir, f"concat_{strategy_name}{range_suffix}.txt")
    with open(concat_file, "w") as f:
        for path in concat_list:
            f.write(f"file '{os.path.abspath(path)}'\n")

    # Concatenate video
    subprocess.run(
        [
            "ffmpeg", "-f", "concat", "-safe", "0",
            "-i", concat_file,
            "-c", "copy", "-y", output_path,
        ],
        capture_output=True,
    )

    print(f"  Saved to {output_path}")

    # Report stats
    total_original = sum(entry["duration_s"] for entry, _ in shot_groups)
    print(f"  Original duration: {total_original:.1f}s ({total_original / 60:.1f}min)")
    print(f"  Clips used: {total_clips}")

    # Mux audio (auto-discovers all available audio strategies if none specified)
    _mux_audio(output_dir, output_path, concat_list, prompts_full,
               audio_strategy, speech_voice, music_strategy)

    return output_path


def _mux_audio(output_dir: str, video_path: str, concat_list: list[str],
               prompts_full: list[dict], audio_strategy: str | None,
               speech_voice: str | None, music_strategy: str | None = None):
    """Build and mux audio/speech tracks into a video file.

    If audio_strategy is given, only that SFX track is used. Otherwise,
    auto-discovers all audio strategy directories and mixes them together.
    If music_strategy is given, its track is overlaid at full volume on top
    of the SFX track (music dominates for music-bucket shots).
    """
    # Compute actual adjusted video durations per shot
    shot_durations = {}
    for adjusted_path in concat_list:
        basename = os.path.splitext(os.path.basename(adjusted_path))[0]
        idx_str = basename.split("-")[0]
        idx = int(idx_str)
        dur = _probe_duration(adjusted_path)
        shot_durations[idx] = shot_durations.get(idx, 0) + dur

    # Diagnostic tripwire: flag any shot whose corrected video duration still
    # deviates from shots.json by more than ~1 frame (24fps) -- should be rare
    # now that _retime_shot_clips corrects split-shot overshoot, but a
    # regression here would otherwise silently re-desync audio placement.
    FRAME_TOLERANCE_S = 1 / 24
    stitched_prompts = []
    for p in prompts_full:
        if p["index"] in shot_durations:
            actual = shot_durations[p["index"]]
            deviation = actual - p["duration_s"]
            if abs(deviation) > FRAME_TOLERANCE_S:
                print(
                    f"  Warning: shot {p['index']} duration deviates from "
                    f"shots.json by {deviation:+.3f}s ({actual:.3f}s vs "
                    f"{p['duration_s']:.3f}s expected)"
                )
            patched = dict(p)
            patched["duration_s"] = actual
            stitched_prompts.append(patched)

    # Build SFX track(s)
    audio_tracks = []
    if audio_strategy:
        strategies = [audio_strategy]
    else:
        # Auto-discover all audio strategy directories
        audio_root = os.path.join(output_dir, "audio")
        if os.path.isdir(audio_root):
            strategies = sorted(
                d for d in os.listdir(audio_root)
                if os.path.isdir(os.path.join(audio_root, d))
            )
        else:
            strategies = []

    for strat in strategies:
        track = _stitch_audio(output_dir, strat, stitched_prompts, None)
        if track and os.path.exists(track):
            audio_tracks.append((strat, track))

    # Build music track (music-bucket shots only; silence elsewhere)
    music_track = None
    if music_strategy:
        music_track = _stitch_audio(output_dir, music_strategy, stitched_prompts, None)
        if music_track and not os.path.exists(music_track):
            music_track = None

    # Build speech track
    speech_track = None
    if speech_voice:
        speech_track = _stitch_speech(output_dir, stitched_prompts, None)

    # Collect all tracks to mix, with volume levels:
    #   Music strategies (-4 dB): prominent but not overpowering
    #   SFX strategies (-8 dB): ducked under music and dialogue
    #   Explicit music_strategy track (0 dB): full volume (legacy path)
    #   Speech is boosted to be clearly audible over everything
    SPEECH_VOLUME_DB = 6  # boost speech

    def _strategy_vol(strat: str) -> int:
        return -4 if "music" in strat else -8

    sfx_tracks = [(t, _strategy_vol(strat)) for strat, t in audio_tracks]
    if music_track:
        sfx_tracks.append((music_track, 0))
    if speech_track and os.path.exists(speech_track):
        sfx_tracks.append((speech_track, SPEECH_VOLUME_DB))

    # Determine final audio to mux
    audio_to_mux = None
    if len(sfx_tracks) > 1:
        combined_label = audio_strategy or "all"
        combined_path = os.path.join(output_dir, f"combined_audio_{combined_label}.wav")
        inputs = []
        filters = []
        for i, (t, vol) in enumerate(sfx_tracks):
            inputs.extend(["-i", t])
            filters.append(f"[{i}]volume={vol}dB[v{i}]")
        mix_inputs = "".join(f"[v{i}]" for i in range(len(sfx_tracks)))
        filters.append(
            f"{mix_inputs}amix=inputs={len(sfx_tracks)}:duration=longest:normalize=0,"
            f"alimiter=limit=0.95[out]"
        )
        subprocess.run(
            ["ffmpeg", "-y"] + inputs +
            ["-filter_complex", ";".join(filters),
             "-map", "[out]",
             combined_path],
            capture_output=True,
        )
        audio_to_mux = combined_path
    elif len(sfx_tracks) == 1:
        track, vol = sfx_tracks[0]
        filtered_path = os.path.join(output_dir, "single_audio_filtered.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-i", track, "-af", f"volume={vol}dB", filtered_path],
            capture_output=True,
        )
        audio_to_mux = filtered_path

    if audio_to_mux:
        muxed_path = video_path.replace(".mp4", "_with_audio.mp4")
        # Explicit -t (video's real length) instead of -shortest: audio and
        # video are now built to the same target duration by construction,
        # so any mismatch here indicates a bug upstream rather than expected
        # drift -- -shortest would silently absorb it instead of surfacing it.
        video_duration = _probe_duration(video_path)
        audio_duration = _probe_duration(audio_to_mux)
        if abs(audio_duration - video_duration) > 1 / 24:
            print(
                f"  Warning: muxed audio length ({audio_duration:.3f}s) "
                f"differs from video length ({video_duration:.3f}s) by "
                f"more than 1 frame"
            )
        result = subprocess.run(
            ["ffmpeg", "-y",
             "-i", video_path,
             "-i", audio_to_mux,
             "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
             "-t", f"{video_duration}",
             muxed_path],
            capture_output=True,
        )
        if result.returncode == 0:
            os.replace(muxed_path, video_path)
            labels = [s for s in strategies if any(s in t for t in audio_tracks)]
            if speech_track and os.path.exists(speech_track):
                labels.append("speech")
            print(f"  {'+'.join(labels)} muxed into {video_path}")
        else:
            print(f"  Warning: audio mux failed, silent video preserved")
            if os.path.exists(muxed_path):
                os.remove(muxed_path)


def stitch_clips(args):
    """Concatenate all clips into a single reconstructed video.

    Speed-adjusts each clip to match original shot duration using FFmpeg's
    setpts filter. Uses clip metadata from decode_progress.json when available,
    falls back to ffprobe for clips without metadata.

    If a scenes.json file exists in the output directory, also produces
    separate videos for each named scene range.
    """
    output_dir = args.output_dir
    strategy_name = args.strategy
    clips_dir = manifest.clips_dir(output_dir, strategy_name)

    if not os.path.exists(clips_dir):
        clips_dir_flat = os.path.join(output_dir, "clips")
        if os.path.exists(clips_dir_flat):
            clips_dir = clips_dir_flat
        else:
            print(f"Error: {clips_dir} not found. Run decode first.")
            sys.exit(1)

    try:
        prompts_full, _ = manifest.load_shots(output_dir)
    except FileNotFoundError:
        print(f"Error: {manifest.shots_path(output_dir)} not found. Run encoder first.")
        sys.exit(1)

    # Load clip metadata from progress (try per-strategy, fall back to legacy)
    progress_path = manifest.decode_progress_path(output_dir, strategy_name)
    if not os.path.exists(progress_path):
        progress_path = os.path.join(output_dir, "decode_progress.json")
    clips_meta = {}
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            progress = json.load(f)
        clips_meta = progress.get("clips", {})

    adjusted_dir = os.path.join(output_dir, "adjusted", strategy_name)
    audio_strategy = getattr(args, "audio_strategy", None)
    speech_voice = getattr(args, "speech_voice", None)
    start_index = getattr(args, "start_index", None)
    music_strategy = getattr(args, "music_strategy", None)

    # Auto-detect speech if speech_progress.json exists
    if not speech_voice:
        speech_progress = os.path.join(output_dir, "speech_progress.json")
        if os.path.exists(speech_progress):
            speech_voice = "auto"

    # Build output filename
    output_path = manifest.reconstructed_path(output_dir, strategy_name, audio_strategy)

    # Stitch the full (or --start-index filtered) video
    result = _stitch_range(
        output_dir, strategy_name, prompts_full, clips_meta, clips_dir,
        adjusted_dir, start_index, None, output_path,
        audio_strategy, speech_voice, music_strategy, label="full reconstruction",
    )
    if result is None:
        print("No clips found to stitch.")
        sys.exit(1)

    # Stitch named scenes if scenes.json exists
    scenes_path = os.path.join(output_dir, "scenes.json")
    if os.path.exists(scenes_path):
        with open(scenes_path) as f:
            scenes = json.load(f)

        if scenes:
            print(f"\n{'='*40}")
            print(f"Stitching {len(scenes)} scene(s)...")
            print(f"{'='*40}")

        scene_outputs = []
        for scene in scenes:
            name = scene["name"]
            scene_start = scene["start"]
            scene_end = scene["end"]
            scene_output = os.path.join(
                output_dir, f"scene_{name}_{strategy_name}.mp4"
            )

            scene_result = _stitch_range(
                output_dir, strategy_name, prompts_full, clips_meta, clips_dir,
                adjusted_dir, scene_start, scene_end, scene_output,
                audio_strategy, speech_voice, music_strategy,
                label=f"scene '{name}' (#{scene_start}-#{scene_end})",
            )
            if scene_result:
                scene_outputs.append((name, scene_result))

        if scene_outputs:
            print(f"\nScene videos:")
            for name, path in scene_outputs:
                print(f"  {name}: {path}")
