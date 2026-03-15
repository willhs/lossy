---
id: plan-0003
type: spec
purpose: "Implementation plan for strategy pattern refactor of decode.py and Seedance 1.0 Pro Fast trial."
tags: ["plan", "decoder", "strategy-pattern", "seedance"]
related: ["./task.md"]
---

# Seedance Strategy Trial — Implementation Plan

## Overview

Refactor `decode.py` to support swappable video generation backends via a strategy pattern, then run a 15-shot trial with Seedance 1.0 Pro Fast on fal.ai. The refactor is minimal — all Replicate-specific code is already isolated in one function (`generate_clip`, lines 88-124), and the stitcher's only coupling is a hardcoded `clip_duration = 81 / 16` at line 269.

**Primary Goal**: Generate shots 10-24 with Seedance Fast at duration-matched lengths and compare against Wan 2.2 experiment 0001.

**Approach**: Extract a strategy interface, wrap existing Replicate code as one implementation, add Seedance as a second, persist clip metadata in `decode_progress.json` so the stitcher knows actual durations.

## Current State Analysis

All generation logic lives in `decode.py` (341 lines). Key coupling points:

- `generate_clip()` (line 88-124): Replicate SDK call, hardcoded `num_frames: 81`, `fps: 16`, model ID `wan-video/wan-2.2-t2v-fast`. Uses `httpx` to download output.
- `run_decode()` (line 192, 204): Hardcoded cost `+= 0.05` in two places.
- `stitch_clips()` (line 269): Hardcoded `clip_duration = 81 / 16`.
- `format_prompt()` (line 30-81): Strategy-independent, no changes needed.
- No tests exist for `decode.py`.

## Desired End State

- `python decode.py output/dir --strategy replicate-wan` behaves identically to current code.
- `python decode.py output/dir --strategy fal-seedance` generates clips with durations matching originals (within 1s for 2-12s range), splits shots >12s into multiple clips.
- Stitcher reads actual clip durations from progress metadata instead of hardcoding.
- Experiment 0002 documents the comparison.

## What We're NOT Doing

- Changing `format_prompt()` or the prompt format — same prompts go to both backends.
- Adding the RunPod/self-hosted strategy — that's a future task.
- Changing the encoder or any other file.

---

## Phase 1: Strategy Pattern Refactor

### Overview

Extract the strategy interface, wrap existing Replicate code, update progress tracking and stitcher to be strategy-aware. No new dependencies, no functional changes.

### Tasks

#### 1. Add strategy classes and ClipResult dataclass

- [x] Add the following to `decode.py` after the `load_env()` function (after line 28), before `format_prompt`:

```python
from dataclasses import dataclass


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
```

#### 2. Move existing Replicate code into ReplicateWanStrategy

- [x] Convert the existing `generate_clip()` function into a method on `ReplicateWanStrategy`. Place this class after `GenerationStrategy`:

```python
class ReplicateWanStrategy(GenerationStrategy):
    """Replicate Wan 2.2 Fast — fixed ~5.06s clips at $0.05 each."""

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
```

- [x] Remove the old standalone `generate_clip()` function (lines 88-124).

#### 3. Update `run_decode()` to use strategy

- [x] Update `run_decode()` to accept and use a strategy. Key changes:

Replace the current generate-and-track logic (lines 170-224) with strategy-aware code. The main changes are:
- `generate_clip(prompt_text, clip_path, seed=idx)` becomes `strategy.generate(prompt_text, clips_dir, idx, entry["duration_s"], seed=idx)`
- The return value is `list[ClipResult]` (empty = failure) instead of `bool`
- Cost comes from `ClipResult.cost` instead of hardcoded `0.05`
- Store clip metadata in progress for the stitcher: `progress["clips"][str(idx)]` = list of `{path, duration_s}`
- The file-existence skip check (lines 179-182) should check the clip naming pattern for the current strategy

The updated loop body:

```python
    # Initialize clips metadata in progress if not present
    if "clips" not in progress:
        progress["clips"] = {}

    for entry in prompts:
        idx = entry["index"]

        if idx in completed_set:
            continue

        # Skip if already generated (check for primary clip file)
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
```

#### 4. Update stitcher to use clip metadata

- [x] Replace the hardcoded `clip_duration = 81 / 16` in `stitch_clips()` with metadata-driven durations. The stitcher should:

1. Load `decode_progress.json` to get the `clips` metadata
2. For each shot, look up clip entries from metadata
3. If metadata exists, use `duration_s` from each clip entry (handles splits too)
4. If no metadata (backwards compat with old progress files), fall back to ffprobe

Replace the clip collection and speed-adjustment section (lines 251-290) with:

```python
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
            # Use metadata — handles both single clips and splits
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

    adjusted_dir = os.path.join(output_dir, "adjusted")
    os.makedirs(adjusted_dir, exist_ok=True)

    concat_list = []
    for i, (clip_path, original_duration, actual_duration) in enumerate(clip_entries):
        basename = os.path.splitext(os.path.basename(clip_path))[0]
        adjusted_path = os.path.join(adjusted_dir, f"{basename}.mp4")

        if not os.path.exists(adjusted_path):
            # For split clips, original_duration applies to the whole shot,
            # but each part's actual_duration is already close to its target.
            # Only adjust if there's meaningful difference.
            speed_factor = original_duration / actual_duration if actual_duration > 0 else 1.0

            # For split clips, don't speed-adjust — they're already duration-matched
            if basename.count("-") > 0:
                # This is a split part — just copy without speed adjustment
                speed_factor = 1.0

            if abs(speed_factor - 1.0) < 0.05:
                # Close enough — just copy
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
```

Wait — for split clips, the duration logic needs more thought. When shot 10 (19s) is split into 12s + 7s, each part is already the right duration. The stitcher should NOT try to speed-adjust split parts to 19s each. The split parts should be concatenated as-is.

The logic above handles this: split clips (detected by `-` in the basename) get `speed_factor = 1.0`. Single clips get the normal speed adjustment.

- [x] Add the `_probe_duration` helper for backwards compatibility (place before `stitch_clips`):

```python
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
```

- [x] Update the stats reporting at the end of `stitch_clips` to work with the new tuple format:

```python
    # Report stats
    total_original = sum(orig_dur for _, orig_dur, _ in clip_entries)
    print(f"  Original duration: {total_original:.1f}s ({total_original / 60:.1f}min)")
    print(f"  Clips used: {len(clip_entries)}")
```

#### 5. Add `--strategy` CLI arg

- [x] Update `main()` to add the strategy argument and pass it through:

```python
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
```

- [x] Update `run_decode` signature from `def run_decode(args):` to `def run_decode(args, strategy: GenerationStrategy):`.

### Success Criteria

#### Automated Verification:
- [x] Run: `python decode.py --help` — shows `--strategy` option with `replicate-wan` and `fal-seedance` choices
- [x] Run: `python -c "from decode import ReplicateWanStrategy, FalSeedanceStrategy, ClipResult"` — imports succeed

#### Manual Verification:
- [x] Manual: Confirm existing `decode_progress.json` in `output/star_wars_iv_v2/` still loads correctly (backwards compat)

---

## Phase 2: Seedance Strategy Implementation

### Overview

Implement `FalSeedanceStrategy` with duration rounding and shot splitting for clips >12s. Install `fal-client` dependency.

### Tasks

#### 1. Install fal-client

- [x] Run: `pip install fal-client` in the project venv
- [ ] Add `FAL_KEY` to `.env` file (user must provide their own key — manual step)

#### 2. Implement FalSeedanceStrategy

- [x] Add `FalSeedanceStrategy` class after `ReplicateWanStrategy`:

```python
class FalSeedanceStrategy(GenerationStrategy):
    """fal.ai Seedance 1.0 Pro Fast — 2-12s duration control at ~$0.10/clip (480p)."""

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
```

### Success Criteria

#### Automated Verification:
- [x] Run: `python -c "from decode import FalSeedanceStrategy; s = FalSeedanceStrategy(); print(s._target_durations(3.7))"` — prints `[4]`
- [x] Run: `python -c "from decode import FalSeedanceStrategy; s = FalSeedanceStrategy(); print(s._target_durations(19.0))"` — prints `[12, 7]`
- [x] Run: `python -c "from decode import FalSeedanceStrategy; s = FalSeedanceStrategy(); print(s._target_durations(0.8))"` — prints `[2]`
- [x] Run: `python -c "import fal_client; print('ok')"` — confirms fal-client installed

---

## Phase 3: Run Trial and Document

### Overview

Generate shots 10-24 with Seedance Fast, stitch, and compare against experiment 0001. Document findings.

### Tasks

#### 1. Generate test batch

- [ ] Run: `python decode.py output/star_wars_iv_v2 --strategy fal-seedance --start-index 10 --limit 15`

Note: this will write clips to the same `output/star_wars_iv_v2/clips/` directory. The Seedance clips will have different filenames only for split shots (e.g., `0010-01.mp4`). For non-split shots, they'll overwrite the Wan clips.

**To preserve the Wan clips for comparison**, first:
- [ ] Run: `mkdir -p output/star_wars_iv_v2/clips_wan && cp output/star_wars_iv_v2/clips/00{10..24}.mp4 output/star_wars_iv_v2/clips_wan/` to back up Wan clips before generating Seedance clips.

Then generate:
- [ ] Run the Seedance decode command and monitor for errors.

#### 2. Stitch Seedance output

- [ ] Run: `python decode.py output/star_wars_iv_v2 --stitch --start-index 10`
- [ ] Manual: Watch `output/star_wars_iv_v2/reconstructed.mp4` and note quality differences.

#### 3. Document experiment

- [ ] Create `docs/research/experiments/0002-seedance-duration-test.md` with findings:

```markdown
---
id: experiment-0002-seedance-duration
type: note
purpose: "Document findings from Seedance 1.0 Pro Fast trial: same 15 shots as experiment 0001, comparing duration accuracy and quality."
scope: ["research", "decoder"]
non_goals: []
tags: ["experiment", "decoder", "video-generation", "seedance", "fal-ai"]
related: ["research/experiments/0001-first-e2e-decode-test.md", "design/adr/004-replicate-wan22-for-video-generation.md"]
---

## Setup

- **Shots tested**: 10-24 (same as experiment 0001)
- **Model**: fal.ai Seedance 1.0 Pro Fast, 480p, 16:9
- **Duration control**: Integer seconds (2-12s), shots >12s split into multiple clips
- **Comparison baseline**: Experiment 0001 (Replicate Wan 2.2 Fast, fixed 5.06s clips)

## Cost

[Fill in actual cost after run]

## Duration Accuracy

[Table comparing: original duration, Seedance duration, delta, vs Wan 2.2 delta]

## Quality Observations

### vs Wan 2.2
[Side-by-side quality notes]

### Duration matching
[Did eliminating speed adjustment improve watchability?]

### Shot splitting
[How did shot 10 (19s -> 12s + 7s) look?]

## Key Takeaways

[Fill in after evaluation]
```

### Success Criteria

- [ ] All 15 shots generated successfully (0 failures)
- [ ] Total cost recorded and within expected range (~$1.50)
- [ ] Stitched output plays correctly in VLC
- [ ] Experiment 0002 document completed with quality comparison

---

## Phase 4: Tests

### Overview

Add unit tests for the new decoder logic. Follow the existing pattern in `test_encode.py` — test pure data transformation functions, don't call external APIs.

### Tasks

#### 1. Create test_decode.py

- [x] Create `test_decode.py` with tests for the strategy pattern and duration logic:

```python
"""Tests for decode.py strategy pattern and duration logic."""

import json
import os

import pytest

from decode import (
    ClipResult,
    FalSeedanceStrategy,
    ReplicateWanStrategy,
    format_prompt,
)


# ---------------------------------------------------------------------------
# format_prompt
# ---------------------------------------------------------------------------

class TestFormatPrompt:
    def test_full_entry(self):
        entry = {
            "description": {
                "shot_type": "wide",
                "camera_movement": "static",
                "action": "A spaceship approaches a planet.",
                "subjects": ["spaceship", "planet"],
                "lighting": "Harsh rim lighting from behind.",
                "color_palette": ["black", "blue", "white"],
                "mood": "ominous",
                "setting": "Deep space.",
            }
        }
        result = format_prompt(entry)
        assert "Cinematic wide shot, static." in result
        assert "A spaceship approaches a planet." in result
        assert "Mood: ominous." in result

    def test_minimal_entry(self):
        entry = {"description": {"action": "A door opens."}}
        result = format_prompt(entry)
        assert "A door opens." in result

    def test_subjects_list_joined(self):
        entry = {
            "description": {
                "action": "Run.",
                "subjects": ["trooper", "droid"],
            }
        }
        result = format_prompt(entry)
        assert "trooper, droid" in result

    def test_subjects_skipped_when_action_long(self):
        entry = {
            "description": {
                "action": "A very long action description that exceeds fifty characters easily.",
                "subjects": ["hidden subject"],
            }
        }
        result = format_prompt(entry)
        assert "hidden subject" not in result

    def test_color_palette_list(self):
        entry = {"description": {"color_palette": ["red", "gold"]}}
        result = format_prompt(entry)
        assert "red, gold" in result

    def test_color_palette_string(self):
        entry = {"description": {"color_palette": "warm tones"}}
        result = format_prompt(entry)
        assert "warm tones" in result


# ---------------------------------------------------------------------------
# FalSeedanceStrategy._target_durations
# ---------------------------------------------------------------------------

class TestTargetDurations:
    def setup_method(self):
        self.strategy = FalSeedanceStrategy()

    def test_short_clamps_to_min(self):
        assert self.strategy._target_durations(0.5) == [2]
        assert self.strategy._target_durations(0.8) == [2]
        assert self.strategy._target_durations(1.0) == [2]

    def test_rounds_to_nearest_int(self):
        assert self.strategy._target_durations(3.3) == [3]
        assert self.strategy._target_durations(3.7) == [4]
        assert self.strategy._target_durations(5.5) == [6]

    def test_exact_integers(self):
        assert self.strategy._target_durations(2.0) == [2]
        assert self.strategy._target_durations(7.0) == [7]
        assert self.strategy._target_durations(12.0) == [12]

    def test_at_max_boundary(self):
        assert self.strategy._target_durations(12.0) == [12]
        assert self.strategy._target_durations(12.4) == [12]

    def test_split_just_over_max(self):
        # 12.5 rounds to 12, but 13.0 should split
        result = self.strategy._target_durations(13.0)
        assert result == [12, 2]  # remainder 1.0 clamps to min 2

    def test_split_medium(self):
        assert self.strategy._target_durations(15.0) == [12, 3]
        assert self.strategy._target_durations(19.0) == [12, 7]

    def test_split_long(self):
        assert self.strategy._target_durations(25.0) == [12, 12, 2]

    def test_split_very_long(self):
        result = self.strategy._target_durations(38.0)
        assert result == [12, 12, 12, 2]

    def test_all_parts_within_range(self):
        """Every part should be between MIN_DURATION and MAX_DURATION."""
        for target in [0.5, 2.0, 7.5, 12.0, 15.0, 19.0, 25.0, 38.0]:
            parts = self.strategy._target_durations(target)
            for p in parts:
                assert 2 <= p <= 12, f"Part {p} out of range for target {target}"


# ---------------------------------------------------------------------------
# ClipResult
# ---------------------------------------------------------------------------

class TestClipResult:
    def test_dataclass_fields(self):
        r = ClipResult(path="/tmp/test.mp4", actual_duration_s=5.0, cost=0.10)
        assert r.path == "/tmp/test.mp4"
        assert r.actual_duration_s == 5.0
        assert r.cost == 0.10


# ---------------------------------------------------------------------------
# ReplicateWanStrategy properties
# ---------------------------------------------------------------------------

class TestReplicateWanStrategy:
    def test_name(self):
        assert ReplicateWanStrategy.name == "replicate-wan"

    def test_clip_duration_constant(self):
        assert ReplicateWanStrategy.CLIP_DURATION == pytest.approx(5.0625)
```

### Success Criteria

#### Automated Verification:
- [x] Run: `python -m pytest test_decode.py -v` — all tests pass (18/18)

---

## Phase 5: Update Documentation

### Overview

Update architecture docs and ADR-004 to reflect the strategy pattern and multi-backend support.

### Tasks

#### 1. Update architecture.md

- [x] Update the Decode section in `docs/design/architecture.md` (lines 57-64) to mention the strategy pattern:

Replace:
```markdown
### Decode

Input: scene manifest JSON. Output: directory of generated video clips + stitched output.

1. Read each scene description from the manifest.
2. Send each description as a prompt to a video generation API.
3. Download generated clips.
4. Stitch clips sequentially into a single reconstructed film (FFmpeg concat).
```

With:
```markdown
### Decode

Input: scene manifest JSON. Output: directory of generated video clips + stitched output.

1. Read each scene description from the manifest.
2. Send each description as a prompt to a video generation backend (selectable via `--strategy`).
3. Download generated clips. Long shots may be split into multiple clips by the strategy.
4. Speed-adjust clips to match original shot durations (skipped when strategy produces duration-matched clips).
5. Stitch clips sequentially into a single reconstructed film (FFmpeg concat).

Supported strategies:
- `replicate-wan` — Replicate Wan 2.2 Fast, fixed ~5s clips, cheapest ($0.05/clip)
- `fal-seedance` — fal.ai Seedance 1.0 Pro Fast, 2-12s duration control (~$0.10/clip at 480p)
```

- [x] Update the Tech Stack section (line 84) to list both backends:

Replace:
```markdown
- **Video generation**: Cloud API (Replicate, fal.ai, RunPod, etc.) — whichever is cheapest per clip
```

With:
```markdown
- **Video generation**: Swappable backends via strategy pattern — Replicate Wan 2.2 Fast (default, cheapest), fal.ai Seedance 1.0 Pro Fast (duration control)
```

#### 2. Update ADR-004

- [x] Add a "Status Update" section to the end of `docs/design/adr/004-replicate-wan22-for-video-generation.md` (before any trailing newline):

```markdown

# Status Update (2026-03-15)

The decoder now supports multiple video generation backends via a strategy pattern (`--strategy` CLI flag). Replicate Wan 2.2 Fast remains the default and cheapest option. Seedance 1.0 Pro Fast on fal.ai was added as an alternative that supports 2-12s integer duration control, addressing the speed-adjustment artifacts documented in the "Bad" consequences above.

See experiment 0002 (`docs/research/experiments/0002-seedance-duration-test.md`) for the comparison results.
```

#### 3. Update roadmap

- [x] Update `docs/work/roadmap.md` to move the speed-adjustment item to Done and reflect current state:

Move from Now:
```markdown
- **Fix speed-adjustment issues** — very short clips produce still frames in VLC after speed-adjust. Investigate minimum viable duration or alternative approach.
```

To Done (after implementation):
```markdown
- ~~**Fix speed-adjustment issues**~~ — added Seedance strategy with 2-12s duration control. Strategy pattern in decoder supports swappable backends.
```

### Success Criteria

- [x] Manual: Verify `docs/design/architecture.md` accurately describes the strategy pattern
- [x] Manual: Verify ADR-004 has the status update
- [x] Manual: Verify roadmap reflects current state

---

## Final Checklist

- [ ] All phases complete
- [ ] `--strategy replicate-wan` still works (no regressions)
- [ ] `--strategy fal-seedance` produces duration-matched clips
- [ ] `python -m pytest test_decode.py -v` — all tests pass
- [ ] Architecture docs and ADR-004 updated
- [ ] Roadmap reflects current state
- [ ] Experiment 0002 written with comparison findings
- [ ] Wan test clips backed up before overwrite

## References

- Task: `docs/work/tasks/0003-seedance-strategy-trial/task.md`
- Experiment 0001: `docs/research/experiments/0001-first-e2e-decode-test.md`
- Decoder research: `docs/research/0003-prompt-to-video/research.md`
- ADR-004: `docs/design/adr/004-replicate-wan22-for-video-generation.md`
