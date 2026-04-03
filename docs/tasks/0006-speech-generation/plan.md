---
id: plan-0006
type: spec
purpose: "Implementation plan for speech/dialogue track generation from subtitles and mixing with SFX audio."
tags: ["plan", "speech", "dialogue", "decode", "fal", "elevenlabs"]
related: ["./task.md", "docs/tasks/0005-audio-generation/plan.md"]
---

# Speech Generation Implementation Plan

## Overview

Generate time-aligned voice/dialogue audio from subtitle data already extracted by the encoder, and mix it with the existing SFX audio track. TTS is performed via fal.ai's ElevenLabs TTS Turbo v2.5 endpoint. Speech clips are placed at their original SRT timestamp offsets within each shot using FFmpeg `adelay`, then mixed with the SFX track during the final mux.

**Primary Goal**: `decode.py --stitch` produces a video with both SFX and spoken dialogue.

**Approach**: Enrich the dialogue field with timestamps, add a `SpeechStrategy` class (parallel to `AudioStrategy`), a `run_speech()` generation loop, a `_stitch_speech()` function for time-alignment, and mix the two audio tracks during mux.

## Current State Analysis

- `align_subtitles_to_shots()` (encode.py:191-204) maps subtitles to shots but **discards timestamps** -- only returns `{shot_index: [text strings]}`
- `parse_srt()` (encode.py:161-188) correctly parses `{start_s, end_s, text}` from SRT files
- `dialogue` field in prompts.json is currently `string[] | null` (encode.py:639)
- Audio pipeline: `AudioStrategy` base class (decode.py:778) with `ElevenLabsStrategy` and `MMAudioStrategy`
- `run_audio()` (decode.py:1246) is the generation loop with progress tracking
- `_stitch_audio()` (decode.py:1370) concatenates per-shot audio clips, adjusting duration with `apad/atrim`
- `stitch_clips()` (decode.py:1625-1654) muxes a single audio track onto the final video
- Comparator audio toggle (tools/compare.html:411-447) cycles between `generated`/`original`/`off` -- audio comes from the muxed video element, not separate audio files
- Test file: `test_encode.py` (tests `align_subtitles_to_shots`, `parse_srt`, etc.)
- No `test_decode.py` or `test_speech.py` yet in the project root (audio tests from task 0005 are referenced but let me verify)

### Key Discoveries

- ElevenLabs TTS Turbo v2.5 via fal.ai (`fal-ai/elevenlabs/tts/turbo-v2.5`): text + voice_id params, MP3 output, ~$0.05/1k chars
- Speech clips are short (individual subtitle lines, typically 1-10 seconds) -- no duration splitting needed unlike SFX
- The SRT timestamps give absolute positions within the film; we need to convert to shot-relative offsets for placement
- FFmpeg `adelay` filter can position a clip at a specific offset within a shot's duration
- Multiple dialogue lines per shot need to be individually generated and placed, then mixed together for that shot
- The comparator needs a 4th audio state, but since audio comes from the muxed video, a separate speech-only audio element is needed to toggle speech independently

## Desired End State

```bash
# Generate speech clips
python decode.py output/film --speech --speech-voice Roger

# Stitch video + SFX + speech into final film
python decode.py output/film --strategy fal-seedance --stitch --audio-strategy elevenlabs --speech-voice Roger
# -> output/film/reconstructed_fal-seedance.mp4 (with SFX + speech mixed)
```

Pipeline integration:
```bash
python pipeline.py media/film.mp4 -o output/film --strategy fal-seedance --audio-strategy elevenlabs --speech-voice Roger
```

## What We're NOT Doing

- Multi-voice support (different speakers mapped to different voices)
- Speaker diarization in the encoder
- Speech rate/pacing adjustment to match original timing
- Lip-sync in generated video
- Emotion/tone detection

---

## Phase 1: Enrich Dialogue Field with SRT Timestamps

### Overview
Change `align_subtitles_to_shots()` to preserve SRT timestamps as shot-relative offsets, so each dialogue entry becomes `{text, start_s, end_s}` instead of a plain string. This is the data foundation for time-aligned speech placement.

### Changes Required:

#### 1. Update `align_subtitles_to_shots()` return type
**File**: `encode.py:191-204`

Change the return type from `dict[int, list[str]]` to `dict[int, list[dict]]`. Each dict has `{text, start_s, end_s}` where `start_s` and `end_s` are **relative to shot start**.

```python
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
```

#### 2. Update dialogue usage in stage2 Gemini prompt
**File**: `encode.py:604-606`

The Gemini prompt currently joins dialogue strings with `" / "`. Update to extract `.text` from the enriched objects:

```python
dialogue = dialogue_map.get(idx, [])
if dialogue:
    texts = [d["text"] if isinstance(d, dict) else d for d in dialogue]
    context_lines.append(f"Dialogue during this shot: \"{' / '.join(texts)}\"")
```

The `isinstance` check provides backward compatibility with old-format `string[]` dialogue.

#### 3. Update dialogue storage in prompt_entry
**File**: `encode.py:639`

The enriched objects are stored directly -- no change needed to the storage line itself since `dialogue` is already stored as-is. The type just changes from `list[str]` to `list[dict]`.

#### 4. Update existing tests
**File**: `test_encode.py`

The `TestAlignSubtitlesToShots` tests need to verify the new return shape `{text, start_s, end_s}` instead of plain strings.

### Success Criteria:

#### Automated Verification:
- [x] `python -m pytest test_encode.py -v -k align` -- updated tests pass with new return format

#### Manual Verification:
- [ ] Run `python encode.py stage2 output/test --limit 3` on existing output -- prompts.json dialogue field now contains `{text, start_s, end_s}` objects

---

## Phase 2: SpeechStrategy Class + run_speech()

### Overview
Add `SpeechStrategy` class and `run_speech()` generation loop to `decode.py`. Follows the same pattern as `AudioStrategy`/`run_audio()` but generates TTS per dialogue line rather than per shot.

### Changes Required:

#### 1. Add `SpeechClipResult` dataclass
**File**: `decode.py` (after `AudioClipResult` at line 46)

```python
@dataclass
class SpeechClipResult:
    """Result of generating a single speech clip."""
    path: str
    duration_s: float
    offset_s: float  # shot-relative offset for placement
    cost: float
```

#### 2. Add `SpeechStrategy` class
**File**: `decode.py` (after `MMAudioStrategy`, around line 967)

```python
class SpeechStrategy:
    """TTS via fal.ai ElevenLabs Turbo v2.5."""

    MODEL_ID = "fal-ai/elevenlabs/tts/turbo-v2.5"
    COST_PER_1K_CHARS = 0.05

    def __init__(self, voice: str = "Roger"):
        self.voice = voice

    def generate(
        self,
        text: str,
        speech_dir: str,
        shot_index: int,
        line_index: int,
        offset_s: float,
    ) -> SpeechClipResult | None:
        """Generate a single TTS clip for one dialogue line."""
        import fal_client
        import httpx

        clip_name = f"{shot_index:04d}-{line_index:02d}.mp3"
        clip_path = os.path.join(speech_dir, clip_name)

        try:
            result = fal_client.subscribe(
                self.MODEL_ID,
                arguments={
                    "text": text,
                    "voice": self.voice,
                },
                with_logs=False,
            )

            audio_url = result["audio"]["url"]
            resp = httpx.get(audio_url, follow_redirects=True)
            resp.raise_for_status()
            with open(clip_path, "wb") as f:
                f.write(resp.content)

            # Get actual duration via ffprobe
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries",
                 "format=duration", "-of", "csv=p=0", clip_path],
                capture_output=True, text=True,
            )
            duration = float(probe.stdout.strip()) if probe.stdout.strip() else 0

            cost = len(text) / 1000 * self.COST_PER_1K_CHARS
            return SpeechClipResult(
                path=clip_path,
                duration_s=duration,
                offset_s=offset_s,
                cost=cost,
            )

        except Exception as e:
            print(f"  Error generating speech {clip_name}: {e}")
            return None
```

#### 3. Add `run_speech()` function
**File**: `decode.py` (after `run_audio()`)

Iterates prompts, extracts enriched dialogue entries, calls `SpeechStrategy.generate()` per line. Tracks progress in `speech_progress.json`. Structure mirrors `run_audio()`.

```python
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
```

#### 4. Add CLI flags
**File**: `decode.py` CLI section (line 1662+)

Add `--speech` flag and `--speech-voice` argument:

```python
parser.add_argument("--speech", action="store_true",
                    help="Generate speech/dialogue clips (instead of video)")
parser.add_argument("--speech-voice", default="Roger",
                    help="ElevenLabs voice name for speech (default: Roger)")
```

Update dispatch in `main()`:

```python
if args.stitch:
    stitch_clips(args)
elif args.audio:
    ...
elif args.speech:
    strategy = SpeechStrategy(voice=args.speech_voice)
    run_speech(args, strategy)
else:
    ...
```

### Success Criteria:

#### Automated Verification:
- [x] `python -m pytest -v -k speech` -- new tests pass for SpeechStrategy and SpeechClipResult

#### Manual Verification:
- [ ] `python decode.py output/test --speech --speech-voice Roger --limit 3` generates MP3 clips in `output/test/speech/`

---

## Phase 3: Speech Stitching + Mixing

### Overview
Add `_stitch_speech()` to time-align speech clips within each shot using FFmpeg `adelay`, then mix the speech track with the SFX track during the final mux in `stitch_clips()`.

### Changes Required:

#### 1. Add `_stitch_speech()` function
**File**: `decode.py` (after `_stitch_audio()`)

For each shot that has speech clips:
1. Place each clip at its `offset_s` using `adelay` (milliseconds)
2. Mix multiple clips within the same shot using `amix`
3. Pad/trim result to shot duration
4. For shots without speech, generate silence
5. Concatenate all per-shot speech segments into `speech_track.wav`

```python
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
```

#### 2. Update `stitch_clips()` to mix speech + SFX
**File**: `decode.py:1625-1654`

After the existing SFX mux, add speech track mixing. When both tracks exist, use FFmpeg `amix` to combine them before muxing onto video:

```python
# After existing audio mux block:
speech_voice = getattr(args, "speech_voice", None)
if speech_voice:
    speech_track = _stitch_speech(output_dir, stitched_prompts, None)
    if speech_track and os.path.exists(speech_track):
        if audio_track and os.path.exists(audio_track):
            # Mix SFX + speech into combined track
            combined_path = os.path.join(output_dir, "combined_audio.wav")
            subprocess.run(
                ["ffmpeg", "-y",
                 "-i", audio_track, "-i", speech_track,
                 "-filter_complex", "amix=inputs=2:duration=longest",
                 combined_path],
                capture_output=True,
            )
            audio_to_mux = combined_path
        else:
            audio_to_mux = speech_track

        # Mux combined/speech-only audio onto video
        muxed_path = output_path.replace(".mp4", "_with_audio.mp4")
        result = subprocess.run(
            ["ffmpeg", "-y",
             "-i", output_path,
             "-i", audio_to_mux,
             "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
             "-shortest", muxed_path],
            capture_output=True,
        )
        if result.returncode == 0:
            os.replace(muxed_path, output_path)
            print(f"  Speech+SFX muxed into {output_path}")
        else:
            print(f"  Warning: speech mux failed")
            if os.path.exists(muxed_path):
                os.remove(muxed_path)
```

**Important**: The existing audio mux block (lines 1625-1654) should be refactored so that speech and SFX are combined in a single mux step rather than sequential re-muxes. The approach above handles this by building a combined audio track first, then muxing once.

### Success Criteria:

#### Automated Verification:
- [x] `python -m pytest -v` -- all tests pass

#### Manual Verification:
- [ ] `python decode.py output/test --strategy fal-seedance --stitch --audio-strategy elevenlabs --speech-voice Roger` produces video with both SFX and speech
- [ ] `ffprobe` confirms audio stream is present
- [ ] Speech is audible at approximately the right timestamps

---

## Phase 4: Pipeline Integration

### Overview
Add `speech` stage to `pipeline.py` between `audio` and `stitch`. Controlled by `--speech-voice` CLI arg.

### Changes Required:

#### 1. Update `pipeline.py`
**File**: `pipeline.py`

- Add `"speech"` to `STAGES` list: `["encode1", "encode2", "decode", "audio", "speech", "stitch"]`
- Add `--speech-voice` CLI argument (default: None, meaning skip speech)
- Add `speech` command to `build_commands()`:

```python
commands["speech"] = [
    sys.executable, "decode.py", args.output,
    "--speech", "--speech-voice", args.speech_voice or "Roger",
]
if args.start_index is not None:
    commands["speech"] += ["--start-index", str(args.start_index)]
if args.limit:
    commands["speech"] += ["--limit", str(args.limit)]
```

- Update `stitch` command to pass `--speech-voice` when set:

```python
if args.speech_voice:
    commands["stitch"] += ["--speech-voice", args.speech_voice]
```

- Auto-skip speech stage when `--speech-voice` not provided:

```python
if not args.speech_voice:
    skip.add("speech")
```

### Success Criteria:

#### Automated Verification:
- [ ] `python pipeline.py media/test.mp4 -o /tmp/test --strategy fal-seedance --audio-strategy elevenlabs --speech-voice Roger --dry-run` -- shows speech stage in output

---

## Phase 5: Comparator Speech Toggle

### Overview
Add a 4th audio state to the comparator: `generated` -> `generated+speech` -> `original` -> `off`. Since the speech track is a separate file (`speech_track.wav`), serve it as a separate `<audio>` element that can be toggled independently.

### Changes Required:

#### 1. Update comparator HTML
**File**: `tools/compare.html`

- Add a hidden `<audio>` element for the speech track:

```html
<audio id="speech-audio" preload="auto"></audio>
```

- Update `setAudioSource()` to handle the `generated+speech` state:
  - `generated`: play reconstructed video audio (SFX), mute speech
  - `generated+speech`: play both reconstructed video audio and speech track
  - `original`: play original video audio, mute speech
  - `off`: mute everything

- Update `cycleAudioSource()` for 4-state cycle

- When loading a project, set the speech audio `src` to the speech track path (e.g., `output/{name}/speech_track.wav`)

- Sync speech audio playback position with video during seek/play/pause events

#### 2. Update serve.py scan
**File**: `tools/serve.py`

Add `has_speech` flag to the scan API response when `speech_track.wav` exists in the output directory. The comparator uses this to show/hide the speech toggle state.

### Success Criteria:

#### Manual Verification:
- [ ] Comparator cycles through 4 audio states when speech track exists
- [ ] Speech plays in sync with video at correct timestamps
- [ ] When no speech track exists, comparator falls back to 3-state cycle (no change)

---

## Phase 6: Tests

### Overview
Add mock-based tests for the new speech functionality. Tests cover the enriched timestamp alignment, speech strategy data structures, and pipeline integration.

### Changes Required:

#### 1. Update `test_encode.py`
**File**: `test_encode.py`

Update `TestAlignSubtitlesToShots` to verify enriched return format:

```python
class TestAlignSubtitlesToShots:
    def test_basic_alignment(self):
        subtitles = [
            {"start_s": 1.0, "end_s": 3.0, "text": "Hello"},
            {"start_s": 5.0, "end_s": 7.0, "text": "World"},
        ]
        scenes = [
            {"index": 0, "start_s": 0.0, "end_s": 4.0},
            {"index": 1, "start_s": 4.0, "end_s": 8.0},
        ]
        result = align_subtitles_to_shots(subtitles, scenes)
        assert 0 in result
        assert result[0][0] == {"text": "Hello", "start_s": 1.0, "end_s": 3.0}
        assert result[1][0] == {"text": "World", "start_s": 1.0, "end_s": 3.0}

    def test_subtitle_spanning_shots(self):
        subtitles = [{"start_s": 3.0, "end_s": 6.0, "text": "Spanning"}]
        scenes = [
            {"index": 0, "start_s": 0.0, "end_s": 5.0},
            {"index": 1, "start_s": 5.0, "end_s": 10.0},
        ]
        result = align_subtitles_to_shots(subtitles, scenes)
        # Clamped to shot boundaries
        assert result[0][0]["start_s"] == 3.0
        assert result[0][0]["end_s"] == 5.0  # clamped to shot end - shot start
        assert result[1][0]["start_s"] == 0.0  # starts at shot boundary
        assert result[1][0]["end_s"] == 1.0
```

#### 2. Add `test_decode.py` (or extend existing)
**File**: `test_decode.py`

```python
class TestSpeechClipResult:
    def test_fields(self):
        result = SpeechClipResult(path="/tmp/0001-00.mp3", duration_s=2.5,
                                   offset_s=1.0, cost=0.005)
        assert result.offset_s == 1.0
```

#### 3. Add pipeline test for speech stage

```python
def test_build_commands_with_speech(self, base_args):
    base_args.speech_voice = "Roger"
    commands = build_commands(base_args)
    assert "speech" in commands
    assert "--speech-voice" in commands["speech"]
    assert "Roger" in commands["speech"]
```

### Success Criteria:

#### Automated Verification:
- [ ] `python -m pytest -v` -- all tests pass (existing + new)

---

## Final Checklist

- [x] Phase 1: Enriched dialogue with timestamps
- [x] Phase 2: SpeechStrategy + run_speech()
- [x] Phase 3: _stitch_speech() + mixed mux
- [x] Phase 4: Pipeline integration
- [x] Phase 5: Comparator speech toggle
- [x] Phase 6: Tests
- [x] All tests passing: `python -m pytest -v`
- [x] Dry-run pipeline works with --speech-voice
- [ ] Manual: end-to-end test produces video with audible speech at correct timestamps

## Key Files to Create/Modify

```
encode.py (modify) -- align_subtitles_to_shots() enrichment, stage2 dialogue handling
decode.py (modify) -- SpeechClipResult, SpeechStrategy, run_speech(), _stitch_speech(), CLI flags, stitch_clips() mixing
pipeline.py (modify) -- speech stage, --speech-voice arg
tools/compare.html (modify) -- 4-state audio toggle, speech audio element
tools/serve.py (modify) -- has_speech flag in scan API
test_encode.py (modify) -- updated alignment tests
test_decode.py (create or modify) -- speech tests
```

## References

- Task: `docs/tasks/0006-speech-generation/task.md`
- Audio Generation Plan: `docs/tasks/0005-audio-generation/plan.md`
- ADR-002: `docs/design/adr/002-stateless-cli-pipeline.md`
