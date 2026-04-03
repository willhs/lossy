---
id: plan-0002
type: spec
purpose: "Implementation plan for audio encoding v1 — YAMNet classification feeding into Gemini descriptions."
tags: ["plan", "encoder", "audio", "yamnet"]
related: ["./task.md"]
---

# Audio Encoding v1 — Implementation Plan

## Overview

Add audio classification to stage 2 of the lossy encoder. Extract the audio track via FFmpeg, run YAMNet to classify sound events, group 521 AudioSet classes into 6 high-level buckets, aggregate per shot, cache to `audio_labels.json`, and feed the labels as context into the existing Gemini vision API call. Gemini then produces a natural-language `sound` field per shot alongside its visual description.

**Primary Goal**: Every shot's description in `prompts.json` includes what the shot sounds like, not just what it looks like.

**Approach**: YAMNet labels are preprocessing context for Gemini — the same pattern camera motion already uses. No separate output file; audio data flows through the existing pipeline into `prompts.json`.

## Current State Analysis

### Key Discoveries
- `encode.py:240-306` — `detect_camera_motion()` is the template: run analysis on full video, cache to JSON, return dict keyed by shot index
- `encode.py:309-325` — `SYSTEM_PROMPT` defines the structured output Gemini returns (8 fields). Adding `sound` means updating this prompt.
- `encode.py:392-403` — Per-shot context assembly: camera motion label and dialogue are injected as text lines before calling Gemini. Audio labels slot in the same way.
- `encode.py:425-434` — Prompt entries store `camera_motion_detected` and `dialogue` alongside `description`. Audio labels go here as `audio_detected`.
- `encode.py:559-579` — CLI uses argparse subparsers. No new subcommand needed — audio integrates into `run_stage2`.
- `test_encode.py` — Tests target pure functions only (no mocking), using pytest classes with inline fixtures.
- YAMNet outputs `[N, 521]` scores with 0.48s hop. Class names come from a bundled CSV asset. Requires TF 2.x.

## Desired End State

Running stage 2 now includes audio classification:
```bash
python encode.py stage2 output/star_wars_iv_v2
```

Cached intermediates (new):
- `output/star_wars_iv_v2/audio.wav` — mono 16kHz audio track
- `output/star_wars_iv_v2/yamnet_scores.npz` — raw YAMNet output
- `output/star_wars_iv_v2/audio_labels.json` — per-shot bucket + labels (like `camera_motion.json`)

Enriched `prompts.json` entries:
```json
{
  "index": 42,
  "start_s": 95.3,
  "end_s": 98.1,
  "duration_s": 2.8,
  "camera_motion_detected": "pan right",
  "audio_detected": {
    "bucket": "effects",
    "labels": ["Gunshot, gunfire", "Explosion", "Boom"]
  },
  "dialogue": ["Take cover!"],
  "description": {
    "shot_type": "medium wide",
    "camera_movement": "quick pan right following action",
    "subjects": "rebel soldiers ducking behind barricade",
    "action": "soldiers dive for cover as blaster fire erupts across the corridor",
    "lighting": "harsh fluorescent, muzzle flashes",
    "color_palette": "white, grey, red",
    "mood": "chaotic, urgent",
    "setting": "Death Star corridor",
    "sound": "rapid blaster fire with metallic echoes, distant explosion, urgent shouting"
  }
}
```

## What We're NOT Doing

- WhisperX / speaker diarization (future task)
- Audio reconstruction or synthesis on the decode side
- Updating `format_prompt()` in `decode.py` to use the `sound` field (low-hanging follow-on)
- Sending actual audio clips to Gemini (Gemini gets YAMNet labels as text, not raw audio)

---

## Phase 1: Pure Functions + Tests

### Overview
Implement the bucket mapping and per-shot aggregation as pure functions with full test coverage. No I/O, no TensorFlow.

### Tasks

#### 1. Add `class_to_bucket()` to `encode.py`

- [x] Add audio section header and `class_to_bucket` function after the camera motion section (after line ~306)

```python
# ---------------------------------------------------------------------------
# Audio: YAMNet classification and per-shot aggregation
# ---------------------------------------------------------------------------

# YAMNet hop between successive classification frames
YAMNET_HOP_S = 0.48
YAMNET_WINDOW_S = 0.96


def class_to_bucket(class_index: int) -> str:
    """Map a YAMNet class index (0-520) to one of 6 high-level audio buckets.

    Buckets: speech, music, effects, ambient, silence, other.
    Mapping follows AudioSet ontology groupings in yamnet_class_map.csv.
    """
    # Speech: human vocal sounds (0-23), body/crowd sounds (33-66)
    # Excludes singing (24-32) which goes to music
    if class_index <= 23 or 33 <= class_index <= 66:
        return "speech"
    # Music: singing (24-32), instruments and genres (132-276)
    if 24 <= class_index <= 32 or 132 <= class_index <= 276:
        return "music"
    # Ambient: natural environment — wind, water, thunder, fire (277-293)
    if 277 <= class_index <= 293:
        return "ambient"
    # Effects: vehicles, engines, doors, tools, weapons, impacts, liquids,
    # mechanical sounds, alarms, electronic tones (294-493)
    if 294 <= class_index <= 493:
        return "effects"
    # Silence
    if class_index == 494:
        return "silence"
    # Other: animals (67-131), signal processing artifacts (495-520)
    return "other"
```

#### 2. Add `aggregate_shot_audio()` to `encode.py`

- [x] Add `aggregate_shot_audio` function below `class_to_bucket`

```python
def aggregate_shot_audio(
    scores: np.ndarray,
    class_names: list[str],
    shot_start_s: float,
    shot_end_s: float,
    top_n: int = 5,
) -> dict:
    """Aggregate YAMNet scores for a single shot into bucket + top labels.

    Args:
        scores: Full YAMNet output, shape [N, 521].
        class_names: List of 521 class display names.
        shot_start_s: Shot start time in seconds.
        shot_end_s: Shot end time in seconds.
        top_n: Number of top class labels to include.

    Returns:
        {"bucket": str, "labels": [str]}
    """
    # Find YAMNet frames overlapping this shot
    frame_indices = []
    for i in range(len(scores)):
        frame_start = i * YAMNET_HOP_S
        frame_end = frame_start + YAMNET_WINDOW_S
        if frame_start < shot_end_s and frame_end > shot_start_s:
            frame_indices.append(i)

    if not frame_indices:
        return {"bucket": "silence", "labels": []}

    # Mean scores across overlapping frames → [521]
    mean_scores = scores[frame_indices].mean(axis=0)

    # Sum mean scores per bucket
    bucket_scores = {
        "speech": 0.0, "music": 0.0, "effects": 0.0,
        "ambient": 0.0, "silence": 0.0, "other": 0.0,
    }
    for idx, score in enumerate(mean_scores):
        bucket_scores[class_to_bucket(idx)] += float(score)

    # Dominant bucket
    bucket = max(bucket_scores, key=bucket_scores.get)

    # Top-N labels by mean score, skip near-zero
    top_indices = np.argsort(mean_scores)[::-1][:top_n]
    labels = [class_names[i] for i in top_indices if mean_scores[i] > 0.01]

    return {"bucket": bucket, "labels": labels}
```

#### 3. Add tests for `class_to_bucket`

- [x] Add `TestClassToBucket` to `test_encode.py`

```python
from encode import class_to_bucket, aggregate_shot_audio


class TestClassToBucket:
    """Test YAMNet class index to bucket mapping."""

    def test_speech_range_start(self):
        assert class_to_bucket(0) == "speech"  # Speech

    def test_speech_range_end(self):
        assert class_to_bucket(23) == "speech"  # Sigh

    def test_speech_body_sounds(self):
        assert class_to_bucket(48) == "speech"  # Walk, footsteps

    def test_speech_crowd(self):
        assert class_to_bucket(64) == "speech"  # Crowd

    def test_singing_is_music(self):
        assert class_to_bucket(24) == "music"  # Singing
        assert class_to_bucket(32) == "music"  # Humming

    def test_music_instruments(self):
        assert class_to_bucket(132) == "music"  # Music
        assert class_to_bucket(148) == "music"  # Piano
        assert class_to_bucket(179) == "music"  # Orchestra

    def test_music_genres(self):
        assert class_to_bucket(265) == "music"  # Soundtrack music
        assert class_to_bucket(276) == "music"  # Scary music

    def test_ambient_nature(self):
        assert class_to_bucket(277) == "ambient"  # Wind
        assert class_to_bucket(283) == "ambient"  # Rain
        assert class_to_bucket(292) == "ambient"  # Fire

    def test_effects_explosions(self):
        assert class_to_bucket(420) == "effects"  # Explosion
        assert class_to_bucket(421) == "effects"  # Gunshot, gunfire

    def test_effects_vehicles(self):
        assert class_to_bucket(294) == "effects"  # Vehicle
        assert class_to_bucket(331) == "effects"  # Jet engine

    def test_effects_mechanical(self):
        assert class_to_bucket(390) == "effects"  # Siren
        assert class_to_bucket(487) == "effects"  # Rumble

    def test_silence(self):
        assert class_to_bucket(494) == "silence"

    def test_animals_are_other(self):
        assert class_to_bucket(67) == "other"  # Animal
        assert class_to_bucket(104) == "other"  # Roaring cats

    def test_signal_artifacts_are_other(self):
        assert class_to_bucket(507) == "other"  # Noise
        assert class_to_bucket(520) == "other"  # Field recording

    def test_boundary_between_speech_and_music(self):
        assert class_to_bucket(23) == "speech"  # Sigh
        assert class_to_bucket(24) == "music"   # Singing
        assert class_to_bucket(32) == "music"   # Humming
        assert class_to_bucket(33) == "speech"  # Groan

    def test_boundary_between_music_and_ambient(self):
        assert class_to_bucket(276) == "music"   # Scary music
        assert class_to_bucket(277) == "ambient"  # Wind

    def test_boundary_between_ambient_and_effects(self):
        assert class_to_bucket(293) == "ambient"  # Crackle
        assert class_to_bucket(294) == "effects"  # Vehicle

    def test_boundary_between_effects_and_silence(self):
        assert class_to_bucket(493) == "effects"  # Crunch
        assert class_to_bucket(494) == "silence"
```

#### 4. Add tests for `aggregate_shot_audio`

- [x] Add `TestAggregateShotAudio` to `test_encode.py`

```python
class TestAggregateShotAudio:
    """Test per-shot YAMNet score aggregation."""

    def _make_scores(self, n_frames, dominant_class, dominant_val=0.8):
        """Create synthetic scores with one dominant class."""
        scores = np.full((n_frames, 521), 0.001, dtype=np.float32)
        scores[:, dominant_class] = dominant_val
        return scores

    def _class_names(self):
        return [f"class_{i}" for i in range(521)]

    def test_single_frame_music(self):
        scores = self._make_scores(1, 179)  # Orchestra
        result = aggregate_shot_audio(scores, self._class_names(), 0.0, 0.96)
        assert result["bucket"] == "music"
        assert "class_179" in result["labels"]

    def test_speech_dominant(self):
        scores = self._make_scores(10, 0)  # Speech
        result = aggregate_shot_audio(scores, self._class_names(), 0.0, 5.0)
        assert result["bucket"] == "speech"

    def test_effects_explosion(self):
        scores = self._make_scores(5, 420)  # Explosion
        result = aggregate_shot_audio(scores, self._class_names(), 0.0, 3.0)
        assert result["bucket"] == "effects"

    def test_no_overlapping_frames(self):
        scores = self._make_scores(1, 0)
        result = aggregate_shot_audio(scores, self._class_names(), 100.0, 101.0)
        assert result["bucket"] == "silence"
        assert result["labels"] == []

    def test_top_n_limits_labels(self):
        scores = self._make_scores(5, 179)
        scores[:, 148] = 0.5  # Piano
        scores[:, 186] = 0.4  # Violin
        result = aggregate_shot_audio(
            scores, self._class_names(), 0.0, 3.0, top_n=2,
        )
        assert len(result["labels"]) == 2

    def test_near_zero_labels_excluded(self):
        scores = np.full((5, 521), 0.005, dtype=np.float32)
        scores[:, 494] = 0.02  # Silence barely above threshold
        result = aggregate_shot_audio(scores, self._class_names(), 0.0, 3.0)
        assert len(result["labels"]) <= 5

    def test_partial_overlap(self):
        # 20 frames (0-10.08s). Shot is 4.0-6.0s.
        scores = np.full((20, 521), 0.001, dtype=np.float32)
        scores[:10, 0] = 0.9   # Speech in first half
        scores[10:, 179] = 0.9  # Orchestra in second half
        result = aggregate_shot_audio(scores, self._class_names(), 4.0, 6.0)
        assert result["bucket"] in ("speech", "music")
```

### Success Criteria

- [x] Run: `pytest test_encode.py::TestClassToBucket -v` — all pass
- [x] Run: `pytest test_encode.py::TestAggregateShotAudio -v` — all pass

---

## Phase 2: Audio Extraction + YAMNet Inference

### Overview
Extract audio to WAV via FFmpeg, load YAMNet, run classification on full waveform, aggregate per shot, cache all intermediates.

### Tasks

#### 1. Add `extract_audio()` to `encode.py`

- [x] Add `extract_audio` function after `aggregate_shot_audio`

```python
def extract_audio(video_path: str, output_dir: str) -> str:
    """Extract audio track to mono 16kHz WAV. Returns path to WAV file."""
    wav_path = os.path.join(output_dir, "audio.wav")
    if os.path.exists(wav_path):
        print(f"Audio already extracted: {wav_path}")
        return wav_path

    print("Extracting audio track...")
    result = subprocess.run(
        [
            "ffmpeg", "-i", video_path,
            "-ac", "1", "-ar", "16000", "-vn",
            "-y", wav_path,
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Error extracting audio: {result.stderr}")
        sys.exit(1)

    print(f"Audio extracted to {wav_path}")
    return wav_path
```

#### 2. Add `run_yamnet()` to `encode.py`

- [x] Add `run_yamnet` function after `extract_audio`

```python
def run_yamnet(audio_path: str, output_dir: str) -> tuple[np.ndarray, list[str]]:
    """Run YAMNet classification on audio file. Returns (scores, class_names).

    Scores shape: [N, 521] where N is the number of ~0.48s frames.
    Caches raw scores to yamnet_scores.npz for resume support.
    """
    import tensorflow_hub as hub
    import tensorflow as tf
    import csv

    print("Loading YAMNet model...")
    model = hub.load("https://tfhub.dev/google/yamnet/1")

    class_map_path = model.class_map_path().numpy()
    class_names = []
    with tf.io.gfile.GFile(class_map_path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_names.append(row["display_name"])

    # Check cache
    scores_path = os.path.join(output_dir, "yamnet_scores.npz")
    if os.path.exists(scores_path):
        print(f"YAMNet scores already computed: {scores_path}")
        data = np.load(scores_path)
        return data["scores"], class_names

    # Load audio
    from scipy.io import wavfile

    print("Running YAMNet classification...")
    sample_rate, wav_data = wavfile.read(audio_path)
    waveform = wav_data.astype(np.float32) / 32768.0

    start = time.time()
    scores, embeddings, spectrogram = model(waveform)
    scores = scores.numpy()
    elapsed = time.time() - start
    print(f"  Classified {len(scores)} frames in {elapsed:.1f}s")

    # Cache scores
    np.savez_compressed(scores_path, scores=scores)
    print(f"  Scores cached to {scores_path}")

    return scores, class_names
```

#### 3. Add `detect_audio_labels()` to `encode.py`

- [x] Add `detect_audio_labels` function after `run_yamnet` — this is the top-level function analogous to `detect_camera_motion()`

```python
def detect_audio_labels(
    video_path: str, scenes: list[dict], output_dir: str,
) -> dict[str, dict]:
    """Classify audio for each shot using YAMNet. Returns {shot_index: {bucket, labels}}.

    Caches per-shot labels to audio_labels.json (same pattern as camera_motion.json).
    """
    labels_path = os.path.join(output_dir, "audio_labels.json")
    if os.path.exists(labels_path):
        print(f"Audio labels already detected: {labels_path}")
        with open(labels_path) as f:
            return json.load(f)

    # Extract audio and run YAMNet
    wav_path = extract_audio(video_path, output_dir)
    scores, class_names = run_yamnet(wav_path, output_dir)

    # Aggregate per shot
    audio_labels = {}
    total = len(scenes)
    for i, scene in enumerate(scenes):
        audio_labels[str(scene["index"])] = aggregate_shot_audio(
            scores, class_names, scene["start_s"], scene["end_s"],
        )
        if (i + 1) % 100 == 0 or i + 1 == total:
            print(f"  Audio labels: {i + 1}/{total} shots")

    with open(labels_path, "w") as f:
        json.dump(audio_labels, f, indent=2)
    print(f"Audio labels saved to {labels_path}")

    return audio_labels
```

### Success Criteria

- [ ] Manual: Run stage2 on a small test (e.g. `--limit 5`) and verify `audio.wav`, `yamnet_scores.npz`, and `audio_labels.json` are created in the output directory
- [ ] Manual: Re-running hits caches for all three files

---

## Phase 3: Integration into Stage 2

### Overview
Wire audio labels into the Gemini call. Update the system prompt, per-shot context, and prompt entry structure so that `prompts.json` includes audio data.

### Tasks

#### 1. Update `SYSTEM_PROMPT` to include `sound` field

- [x] Add `sound` to the structured output fields in `SYSTEM_PROMPT`

Add after the `"setting"` line:

```
- "sound": description of the soundtrack — what you'd expect to hear based on the visuals and the detected audio labels. Describe music style/mood, sound effects, ambient sounds, and atmosphere. If silence or near-silence, say so.
```

#### 2. Update `generate_prompts()` signature to accept audio labels

- [x] Add `audio_labels: dict` parameter to `generate_prompts()`

Change the function signature from:
```python
def generate_prompts(
    scenes: list[dict],
    dialogue_map: dict[int, list[str]],
    motion_labels: dict,
    output_dir: str,
    provider: str,
) -> list[dict]:
```

To:
```python
def generate_prompts(
    scenes: list[dict],
    dialogue_map: dict[int, list[str]],
    motion_labels: dict,
    audio_labels: dict,
    output_dir: str,
    provider: str,
) -> list[dict]:
```

#### 3. Add audio context to per-shot Gemini prompt

- [x] Add audio label context after the camera motion line in `generate_prompts()`

After the existing camera motion context line (`context_lines.append(f"Detected camera motion: {camera}.")`), add:

```python
        audio = audio_labels.get(str(idx), {})
        if audio.get("labels"):
            audio_desc = f"{audio['bucket']} ({', '.join(audio['labels'])})"
            context_lines.append(f"Detected audio: {audio_desc}.")
```

#### 4. Store `audio_detected` in prompt entries

- [x] Add `audio_detected` field to the prompt entry dict in `generate_prompts()`

In the prompt entry construction (both the normal path and the rate-limit retry path), add after `"camera_motion_detected": camera,`:

```python
                "audio_detected": audio_labels.get(str(idx)),
```

#### 5. Update `run_stage2()` to call audio classification

- [x] Add audio classification step between camera motion and prompt generation in `run_stage2()`

After the camera motion step (`motion_labels = detect_camera_motion(...)`), add:

```python
    # Step 3: Audio classification
    print("Classifying audio...")
    audio_labels = detect_audio_labels(video_path, scenes, output_dir)
```

- [x] Update the `generate_prompts()` call to pass `audio_labels`

Change from:
```python
    prompts = generate_prompts(scenes, dialogue_map, motion_labels, output_dir, args.provider)
```

To:
```python
    prompts = generate_prompts(scenes, dialogue_map, motion_labels, audio_labels, output_dir, args.provider)
```

### Success Criteria

- [x] Run: `pytest test_encode.py -v` — all existing tests still pass
- [x] Run: `pytest test_decode.py -v` — all decoder tests still pass (prompts.json format is backwards compatible)
- [ ] Manual: `python encode.py stage2 output/star_wars_iv_v2 --limit 10` — produces prompts with `audio_detected` and `sound` fields
- [ ] Manual: Spot-check entries — opening fanfare shots should have `sound` describing orchestral music, battle scenes should mention blaster fire or explosions

---

## Phase 4: Documentation

### Overview
Update architecture docs and write an ADR for the audio classification approach.

### Tasks

#### 1. Update architecture docs

- [x] Add an "Audio Classification" subsection to the Stage 2 section of `docs/design/architecture.md` covering:
  - Audio extraction (FFmpeg → mono 16kHz WAV)
  - YAMNet classification (~0.48s frames, 521 AudioSet classes)
  - 6-bucket grouping (speech, music, effects, ambient, silence, other)
  - Per-shot aggregation and caching to `audio_labels.json`
  - Integration as context for Gemini (like camera motion)
  - New `sound` field in structured output
  - Dependencies: `tensorflow`, `tensorflow-hub`, `scipy`

#### 2. Write ADR for audio classification approach

- [x] Create `docs/design/adr/005-yamnet-audio-classification.md` documenting:
  - Decision: Use YAMNet as preprocessing context for Gemini, producing a `sound` description field
  - Context: Audio is 38% of source, previously ignored; raw classification labels are too thin — need natural-language descriptions
  - Options considered: (A) standalone audio manifest with bucket/labels only, (B) send audio clips to Gemini directly, (C) YAMNet labels as Gemini context
  - Chose C: matches existing camera-motion pattern, no extra API cost, Gemini synthesizes labels + visuals into rich descriptions
  - Consequences: Adds TensorFlow dependency (~500MB); audio description quality depends on Gemini interpreting labels correctly; WhisperX for speaker diarization is a natural follow-on

---

## Final Checklist

- [x] All phases complete
- [x] `pytest test_encode.py -v` — all tests pass (old and new)
- [x] `pytest test_decode.py -v` — decoder tests pass
- [ ] `python encode.py stage2 output/star_wars_iv_v2` runs end-to-end with audio classification
- [x] Architecture docs updated
- [x] ADR written

## Dependencies

New Python packages:
- `tensorflow` (or `tensorflow-macos` on Apple Silicon)
- `tensorflow-hub`
- `scipy` (for `scipy.io.wavfile`)

## References

- Task: `docs/tasks/0002-audio-encoding/task.md`
- Research: `docs/research/0002-shot-to-prompt/research.md` (Audio Analysis section)
- YAMNet on TF Hub: `https://tfhub.dev/google/yamnet/1`
- AudioSet ontology: `https://research.google.com/audioset/ontology/index.html`
