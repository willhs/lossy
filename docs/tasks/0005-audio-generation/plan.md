---
id: plan-0005
type: spec
purpose: "Implementation plan for per-shot audio generation and muxing into the reconstructed film."
tags: ["plan", "audio", "decode", "fal"]
related: ["./task.md"]
---

# Audio Generation v1 Implementation Plan

## Overview

Add audio generation to `decode.py` so the reconstructed film has a soundtrack. Per-shot audio is generated from the `sound` descriptions already in `prompts.json`, using text-to-audio APIs via fal.ai. The stitch step muxes audio onto the final video.

**Primary Goal**: `decode.py --stitch` produces a video with audio instead of silent output.

**Approach**: Add audio generation strategies to `decode.py` (mirroring the existing video strategy pattern), a new `--audio` flag for generation, and automatic audio muxing during stitch.

## Current State Analysis

- `decode.py` has a `GenerationStrategy` base class for video backends (line 60), with `generate()` and `format_prompt()` methods
- `stitch_clips()` (line 1059) speed-adjusts video clips with `setpts` filter, strips audio with `-an`, then concats with `-c copy`
- `prompts.json` entries have `description.sound` (free text from Gemini) and `duration_s` (shot duration in seconds)
- `load_env()` (line 20) loads `.env` for API keys; `FAL_KEY` is already configured for Seedance
- Tests are pure-function only (no API mocking), testing formatters, data structures, and strategy logic

### Key Discoveries
- ElevenLabs v2 via fal.ai (`fal-ai/elevenlabs/sound-effects/v2`): max 22s, `text` + `duration_seconds` params, MP3 output, $0.002/sec
- MMAudio V2 (`fal-ai/mmaudio-v2/text-to-audio`): max 30s, `prompt` + `duration` params, FLAC output, seed support, $0.001/sec
- Shots can exceed 22s (e.g., 28.8s in pipeline_test2), so splitting is needed for ElevenLabs
- The stitch `-an` flag must be removed and replaced with audio muxing
- Older prompts.json files (pre-audio-encoding) lack the `sound` field entirely — must handle gracefully

## Desired End State

```bash
# Generate video clips (existing, unchanged)
python decode.py output/film --strategy fal-seedance

# Generate audio clips
python decode.py output/film --audio --audio-strategy elevenlabs

# Stitch video + audio into final film
python decode.py output/film --strategy fal-seedance --stitch
# → output/film/reconstructed_fal-seedance.mp4 (with audio track)
```

Pipeline integration:
```bash
python pipeline.py media/film.mp4 -o output/film --strategy fal-seedance --audio-strategy elevenlabs
```

## What We're NOT Doing

- Speech/dialogue synthesis
- Per-shot routing to different audio models (music vs SFX)
- Audio effects processing (reverb, EQ, spatial audio)
- Modifying encode.py or prompts.json format

---

## Phase 1: Audio Strategy Classes

### Overview
Add audio generation strategy base class and ElevenLabs implementation to `decode.py`. These are separate from video strategies — different base class, different registry.

### Tasks

#### 1. Add AudioStrategy base class and ElevenLabs strategy

- [x] Add `AudioClipResult` dataclass to `decode.py` (after `ClipResult`)

```python
@dataclass
class AudioClipResult:
    """Result of generating a single audio clip."""
    path: str
    actual_duration_s: float
    cost: float
```

- [x] Add `AudioStrategy` base class to `decode.py`

```python
class AudioStrategy:
    """Base class for audio generation backends."""

    name: str = "base"

    def generate(
        self,
        sound_description: str,
        audio_dir: str,
        shot_index: int,
        target_duration_s: float,
        seed: int | None = None,
    ) -> list[AudioClipResult]:
        """Generate audio clip(s) for a shot.

        Returns a list because long shots may be split into multiple clips.
        """
        raise NotImplementedError
```

- [x] Add `ElevenLabsStrategy` to `decode.py`

```python
class ElevenLabsStrategy(AudioStrategy):
    """ElevenLabs Sound Effects v2 via fal.ai — $0.002/sec, max 22s."""

    name = "elevenlabs"
    MODEL_ID = "fal-ai/elevenlabs/sound-effects/v2"
    MAX_DURATION = 22
    MIN_DURATION = 0.5
    COST_PER_SECOND = 0.002

    def _target_durations(self, target_s: float) -> list[float]:
        """Split target duration into chunks within 0.5-22s range.

        Examples:
            3.7s  -> [3.7]
            20.0s -> [20.0]
            28.0s -> [22.0, 6.0]
            50.0s -> [22.0, 22.0, 6.0]
        """
        clamped = max(self.MIN_DURATION, target_s)
        if clamped <= self.MAX_DURATION:
            return [clamped]

        parts = []
        remaining = clamped
        while remaining > self.MAX_DURATION:
            parts.append(float(self.MAX_DURATION))
            remaining -= self.MAX_DURATION
        remainder = max(self.MIN_DURATION, remaining)
        parts.append(remainder)
        return parts

    def generate(
        self,
        sound_description: str,
        audio_dir: str,
        shot_index: int,
        target_duration_s: float,
        seed: int | None = None,
    ) -> list[AudioClipResult]:
        import fal_client
        import httpx

        durations = self._target_durations(target_duration_s)
        results = []

        for part_idx, duration in enumerate(durations):
            if len(durations) == 1:
                clip_name = f"{shot_index:04d}.mp3"
            else:
                clip_name = f"{shot_index:04d}-{part_idx + 1:02d}.mp3"

            clip_path = os.path.join(audio_dir, clip_name)

            try:
                arguments = {
                    "text": sound_description,
                    "duration_seconds": duration,
                    "prompt_influence": 0.3,
                }

                result = fal_client.subscribe(
                    self.MODEL_ID,
                    arguments=arguments,
                    with_logs=False,
                )

                audio_url = result["audio"]["url"]
                resp = httpx.get(audio_url, follow_redirects=True)
                resp.raise_for_status()
                with open(clip_path, "wb") as f:
                    f.write(resp.content)

                cost = duration * self.COST_PER_SECOND
                results.append(AudioClipResult(
                    path=clip_path,
                    actual_duration_s=duration,
                    cost=cost,
                ))

            except Exception as e:
                print(f"  Error generating audio {clip_name}: {e}")
                return []

        return results
```

- [x] Add `MMAudioStrategy` to `decode.py`

```python
class MMAudioStrategy(AudioStrategy):
    """MMAudio V2 text-to-audio via fal.ai — $0.001/sec, max 30s."""

    name = "mmaudio"
    MODEL_ID = "fal-ai/mmaudio-v2/text-to-audio"
    MAX_DURATION = 30
    MIN_DURATION = 1
    COST_PER_SECOND = 0.001

    def _target_durations(self, target_s: float) -> list[float]:
        """Split target duration into chunks within 1-30s range."""
        clamped = max(self.MIN_DURATION, target_s)
        if clamped <= self.MAX_DURATION:
            return [clamped]

        parts = []
        remaining = clamped
        while remaining > self.MAX_DURATION:
            parts.append(float(self.MAX_DURATION))
            remaining -= self.MAX_DURATION
        remainder = max(self.MIN_DURATION, remaining)
        parts.append(remainder)
        return parts

    def generate(
        self,
        sound_description: str,
        audio_dir: str,
        shot_index: int,
        target_duration_s: float,
        seed: int | None = None,
    ) -> list[AudioClipResult]:
        import fal_client
        import httpx

        durations = self._target_durations(target_duration_s)
        results = []

        for part_idx, duration in enumerate(durations):
            if len(durations) == 1:
                clip_name = f"{shot_index:04d}.flac"
            else:
                clip_name = f"{shot_index:04d}-{part_idx + 1:02d}.flac"

            clip_path = os.path.join(audio_dir, clip_name)

            try:
                arguments = {
                    "prompt": sound_description,
                    "duration": duration,
                    "num_steps": 25,
                    "cfg_strength": 4.5,
                }
                if seed is not None:
                    arguments["seed"] = (seed + part_idx) % 65536

                result = fal_client.subscribe(
                    self.MODEL_ID,
                    arguments=arguments,
                    with_logs=False,
                )

                audio_url = result["audio"]["url"]
                resp = httpx.get(audio_url, follow_redirects=True)
                resp.raise_for_status()
                with open(clip_path, "wb") as f:
                    f.write(resp.content)

                cost = duration * self.COST_PER_SECOND
                results.append(AudioClipResult(
                    path=clip_path,
                    actual_duration_s=duration,
                    cost=cost,
                ))

            except Exception as e:
                print(f"  Error generating audio {clip_name}: {e}")
                return []

        return results
```

### Success Criteria

#### Automated Verification:
- [x] Run: `python -m pytest test_decode.py -v -k audio` — all new tests pass

---

## Phase 2: Audio Generation Loop

### Overview
Add `run_audio()` function and `--audio` / `--audio-strategy` CLI flags to `decode.py`. Mirrors `run_decode()` with progress tracking and resume.

### Tasks

#### 1. Add `run_audio()` function

- [x] Add `run_audio()` to `decode.py` (after `run_decode()`)

```python
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

        print(f"  Shot {idx} ({generated + 1}/{total - len(completed_set) - len(skipped_set)} remaining)...")

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
```

#### 2. Update CLI to support audio generation

- [x] Add `--audio` flag and `--audio-strategy` argument to the argparse in `main()`

```python
# In the existing argument parser, add:
parser.add_argument("--audio", action="store_true",
                    help="Generate audio clips (instead of video)")
parser.add_argument("--audio-strategy", choices=["elevenlabs", "mmaudio"],
                    default="elevenlabs",
                    help="Audio generation backend (default: elevenlabs)")
```

- [x] Update `main()` dispatch to handle `--audio`

```python
# In main(), after load_env():
if args.stitch:
    stitch_clips(args)
elif args.audio:
    audio_strategies = {
        "elevenlabs": ElevenLabsStrategy,
        "mmaudio": MMAudioStrategy,
    }
    audio_strategy = audio_strategies[args.audio_strategy]()
    run_audio(args, audio_strategy)
else:
    strategies = {
        "replicate-wan": ReplicateWanStrategy,
        "fal-seedance": FalSeedanceStrategy,
        "fal-seedance-pro": FalSeedanceProStrategy,
        "runpod-wan": RunPodWanStrategy,
    }
    strategy = strategies[args.strategy]()
    run_decode(args, strategy)
```

### Success Criteria

#### Manual Verification:
- [x] Manual: Run `python decode.py output/pipeline_test2 --audio --audio-strategy elevenlabs --limit 2` and verify audio clips appear in `output/pipeline_test2/audio/elevenlabs/`
- [x] Manual: Verify audio clips are roughly the right duration (check with `ffprobe`)

---

## Phase 3: Audio Stitch + Mux

### Overview
Modify `stitch_clips()` to detect audio clips and mux them onto the reconstructed video. Audio clips are concatenated into a single track, duration-matched per shot, then combined with the video via FFmpeg.

### Tasks

#### 1. Add `_stitch_audio()` helper

- [x] Add `_stitch_audio()` function to `decode.py` (before `stitch_clips()`)

```python
def _stitch_audio(output_dir: str, audio_strategy: str, prompts: list[dict],
                  start_index: int | None) -> str | None:
    """Concatenate per-shot audio clips into a single audio track.

    Returns path to the combined audio file, or None if no audio clips exist.
    Audio clips are duration-adjusted to match original shot durations using
    FFmpeg's atempo filter (for speed changes) or apad/atrim (for padding/trimming).
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
            # Single clip — trim or pad to match target duration
            clip_path = os.path.join(audio_dir, clips[0]["path"])
            adjusted_path = os.path.join(adjusted_dir, f"{idx:04d}.wav")
            if not os.path.exists(adjusted_path):
                # Use atrim to cut to length, apad to extend if needed
                subprocess.run(
                    ["ffmpeg", "-y", "-i", clip_path,
                     "-af", f"apad=whole_dur={target_duration},atrim=0:{target_duration}",
                     "-ar", "44100", "-ac", "2",
                     adjusted_path],
                    capture_output=True,
                )
            audio_entries.append(adjusted_path)
        else:
            # Multiple clips (split shot) — concatenate parts, then adjust
            parts_file = os.path.join(adjusted_dir, f"{idx:04d}_parts.txt")
            with open(parts_file, "w") as f:
                for clip_info in clips:
                    clip_path = os.path.join(audio_dir, clip_info["path"])
                    f.write(f"file '{os.path.abspath(clip_path)}'\n")

            concat_path = os.path.join(adjusted_dir, f"{idx:04d}_concat.wav")
            adjusted_path = os.path.join(adjusted_dir, f"{idx:04d}.wav")
            if not os.path.exists(adjusted_path):
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
    concat_file = os.path.join(output_dir, "audio_concat.txt")
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
```

#### 2. Modify `stitch_clips()` to mux audio

- [x] Add `--audio-strategy` awareness to `stitch_clips()`

After the existing video concatenation (the `ffmpeg -f concat` call that produces `reconstructed_<strategy>.mp4`), add audio muxing:

```python
# At the end of stitch_clips(), after the video concat and stats:

# Mux audio if available
audio_strategy = getattr(args, "audio_strategy", None)
if audio_strategy:
    audio_track = _stitch_audio(output_dir, audio_strategy, prompts_full, args.start_index)
    if audio_track and os.path.exists(audio_track):
        muxed_path = output_path.replace(".mp4", "_with_audio.mp4")
        result = subprocess.run(
            ["ffmpeg", "-y",
             "-i", output_path,
             "-i", audio_track,
             "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
             "-shortest",
             muxed_path],
            capture_output=True,
        )
        if result.returncode == 0:
            # Replace silent video with muxed version
            os.replace(muxed_path, output_path)
            print(f"  Audio muxed into {output_path}")
        else:
            print(f"  Warning: audio mux failed, silent video preserved")
            if os.path.exists(muxed_path):
                os.remove(muxed_path)
```

Note: `prompts_full` refers to the full prompts list before start_index filtering. The stitch function already loads prompts — store a reference before filtering.

### Success Criteria

#### Manual Verification:
- [x] Manual: Run full flow on pipeline_test2 (limit 3): generate video, generate audio, stitch — verify output has audio track
- [x] Manual: Run `ffprobe reconstructed_fal-seedance.mp4` and confirm both video and audio streams present

---

## Phase 4: Pipeline Integration

### Overview
Add `audio` and `audio-stitch` stages to `pipeline.py` so the full pipeline chains video+audio generation and produces a film with sound.

### Tasks

#### 1. Update pipeline.py

- [x] Add `"audio"` to `STAGES` list (between `"decode"` and `"stitch"`)

```python
STAGES = ["encode1", "encode2", "decode", "audio", "stitch"]
```

- [x] Add `AUDIO_STRATEGIES` list

```python
AUDIO_STRATEGIES = ["elevenlabs", "mmaudio"]
```

- [x] Add `--audio-strategy` CLI argument

```python
parser.add_argument("--audio-strategy", choices=AUDIO_STRATEGIES,
                    default=None,
                    help="Audio generation backend (default: none, skip audio)")
```

- [x] Add `audio` command to `build_commands()`

```python
commands["audio"] = [
    sys.executable, "decode.py", args.output,
    "--audio", "--audio-strategy", args.audio_strategy or "elevenlabs",
]
if args.start_index is not None:
    commands["audio"] += ["--start-index", str(args.start_index)]
if args.limit:
    commands["audio"] += ["--limit", str(args.limit)]
```

- [x] Update `stitch` command to pass `--audio-strategy` when set

```python
commands["stitch"] = [
    sys.executable, "decode.py", args.output,
    "--strategy", args.strategy,
    "--stitch",
]
if args.audio_strategy:
    commands["stitch"] += ["--audio-strategy", args.audio_strategy]
```

- [x] Skip audio stage automatically when `--audio-strategy` is not provided

In `run_pipeline()`, add audio to the auto-skip set when no audio strategy is specified:

```python
skip = set(args.skip) if args.skip else set()
if not args.audio_strategy:
    skip.add("audio")
```

### Success Criteria

#### Automated Verification:
- [x] Run: `python -m pytest test_pipeline.py -v` — all existing tests pass
- [x] Run: `python pipeline.py media/test.mp4 -o output/test --strategy fal-seedance --audio-strategy elevenlabs --dry-run` — verify audio stage appears in output

---

## Phase 5: Tests

### Overview
Add pure-function tests for the new audio code, matching the existing test style (no API mocking, test data structures and logic).

### Tasks

#### 1. Add audio tests to `test_decode.py`

- [x] Add `TestElevenLabsTargetDurations` class

```python
class TestElevenLabsTargetDurations:
    def setup_method(self):
        self.strategy = ElevenLabsStrategy()

    def test_short_clip(self):
        assert self.strategy._target_durations(3.7) == [3.7]

    def test_at_max(self):
        assert self.strategy._target_durations(22.0) == [22.0]

    def test_over_max(self):
        assert self.strategy._target_durations(28.0) == [22.0, 6.0]

    def test_much_over_max(self):
        assert self.strategy._target_durations(50.0) == [22.0, 22.0, 6.0]

    def test_very_short(self):
        result = self.strategy._target_durations(0.1)
        assert result == [0.5]  # Clamped to MIN_DURATION

    def test_zero(self):
        result = self.strategy._target_durations(0.0)
        assert result == [0.5]  # Clamped to MIN_DURATION
```

- [x] Add `TestMMAudioTargetDurations` class

```python
class TestMMAudioTargetDurations:
    def setup_method(self):
        self.strategy = MMAudioStrategy()

    def test_short_clip(self):
        assert self.strategy._target_durations(5.0) == [5.0]

    def test_at_max(self):
        assert self.strategy._target_durations(30.0) == [30.0]

    def test_over_max(self):
        assert self.strategy._target_durations(35.0) == [30.0, 5.0]

    def test_very_short(self):
        result = self.strategy._target_durations(0.3)
        assert result == [1.0]  # Clamped to MIN_DURATION
```

- [x] Add `TestAudioClipResult` class

```python
class TestAudioClipResult:
    def test_fields(self):
        result = AudioClipResult(path="/tmp/0001.mp3", actual_duration_s=5.0, cost=0.01)
        assert result.path == "/tmp/0001.mp3"
        assert result.actual_duration_s == 5.0
        assert result.cost == 0.01
```

#### 2. Add pipeline test for audio stage

- [x] Add test to `test_pipeline.py` for audio stage in `build_commands()`

```python
def test_build_commands_with_audio(self, base_args):
    base_args.audio_strategy = "elevenlabs"
    commands = build_commands(base_args)
    assert "audio" in commands
    audio_cmd = commands["audio"]
    assert "--audio" in audio_cmd
    assert "--audio-strategy" in audio_cmd
    assert "elevenlabs" in audio_cmd
```

- [x] Add test for stitch command including audio-strategy

```python
def test_stitch_includes_audio_strategy(self, base_args):
    base_args.audio_strategy = "elevenlabs"
    commands = build_commands(base_args)
    stitch_cmd = commands["stitch"]
    assert "--audio-strategy" in stitch_cmd
    assert "elevenlabs" in stitch_cmd
```

### Success Criteria

#### Automated Verification:
- [x] Run: `python -m pytest -v` — all tests pass (existing + new)

---

## Final Checklist

- [x] All phases complete
- [x] All tests passing: `python -m pytest -v`
- [x] Dry-run pipeline works: `python pipeline.py media/test.mp4 -o /tmp/test --strategy fal-seedance --audio-strategy elevenlabs --dry-run`
- [x] Manual: end-to-end test on a few shots produces video with audible soundtrack

## Documentation Updates

- [x] Update `docs/design/architecture.md` to describe the audio generation stage
- [x] Update `README.md` with audio CLI usage and `FAL_KEY` requirement for audio
- [x] Update `docs/work/roadmap.md` to move audio generation to Done

## References

- Task: `docs/tasks/0005-audio-generation/task.md`
- ADR-002: `docs/design/adr/002-stateless-cli-pipeline.md`
- ADR-005: `docs/design/adr/005-yamnet-audio-classification.md`
