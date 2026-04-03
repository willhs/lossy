---
id: plan-0004
type: spec
purpose: "Implementation plan for model-aware prompt reformatting with strategy-owned formatters, negative prompts, and A/B testing."
tags: ["plan", "decoder", "prompts", "video-generation"]
related: ["./task.md"]
---

# Prompt Reformatting Implementation Plan

## Overview

Replace the single `format_prompt()` function with strategy-owned formatting via a method on `GenerationStrategy`. Each strategy produces model-optimized prompts from the same structured Gemini description. Add a Chinese negative prompt to the RunPod Wan strategy (Replicate's API doesn't expose a negative_prompt parameter). A/B test Wan formatting on ~15 shots.

**Primary Goal**: Measurably better video generation output through model-efficient prompt vocabulary and structure.

**Approach**: Add `format_prompt()` as a method on the strategy base class (defaulting to current behavior), override in Wan and Seedance strategies with model-optimized formatters. Wire the decode loop to call `strategy.format_prompt(entry)` instead of the module-level function.

## Current State Analysis

- `format_prompt()` at `decode.py:718-769` produces ~150-200 word prose with metadata labels ("Color palette:", "Mood:") and casual camera terms
- The decode loop at `decode.py:835` calls the module-level `format_prompt(entry)` — all strategies receive identical prompt text
- RunPod strategy has an empty-string negative prompt in ComfyUI node "5" (`decode.py:542-548`)
- Replicate Wan API does not support a `negative_prompt` parameter
- 6 tests for `format_prompt()` in `test_decode.py:21-73`

### Key Discoveries
- Replicate's `wan-video/wan-2.2-t2v-fast` only accepts: `prompt`, `num_frames`, `aspect_ratio`, `resolution`, `frames_per_second`, `go_fast`, `sample_shift`, `seed` — no negative prompt field
- RunPod/ComfyUI negative prompt is wired but empty (`decode.py:545`: `"text": ""`)
- `subjects` field in prompts.json is inconsistently typed — sometimes a string, sometimes a list (handled by existing code)

## Desired End State

- Each strategy has a `format_prompt(entry)` method that produces model-optimized text
- Wan formatter: front-loaded subject/action, cinematography vocabulary, no labels, ~150-200 words
- Seedance formatter: ~30-60 words, single action verb, intensity adverbs
- RunPod Wan strategy uses a Chinese negative prompt
- A/B experiment doc comparing old vs new Wan formatting on shots 10-24

## What We're NOT Doing

- Changing the encoder (Gemini system prompt stays as-is)
- Adding negative prompt to Replicate Wan (API doesn't support it)
- LLM-based prompt rewriting at decode time
- Prompt weighting or regional prompting
- ComfyUI Conditioning Combine experiments (separate "might do" item)

---

## Phase 1: Strategy-Owned Formatting

### Overview
Move `format_prompt()` onto the strategy base class and wire the decode loop to use it. No formatting changes yet — just the plumbing.

### Tasks

#### 1. Add `format_prompt()` method to `GenerationStrategy`
- [x] Add `format_prompt(self, entry: dict) -> str` method to `GenerationStrategy` base class (`decode.py:45-62`) that calls the existing module-level `format_prompt()` function

```python
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

    def format_prompt(self, entry: dict) -> str:
        """Format a structured prompt entry for this strategy's model.

        Subclasses override to produce model-optimized prompts.
        """
        return format_prompt(entry)
```

#### 2. Wire decode loop to use strategy method
- [x] Change `decode.py:835` from `format_prompt(entry)` to `strategy.format_prompt(entry)`

```python
# Before:
prompt_text = format_prompt(entry)

# After:
prompt_text = strategy.format_prompt(entry)
```

### Success Criteria
- [x] Run: `python -m pytest test_decode.py -v` — all existing tests pass
- [x] Run: `python pipeline.py media/film.mp4 -o /tmp/test --strategy replicate-wan --dry-run` — no errors (if dry-run is supported, otherwise just verify import works with `python -c "from decode import ReplicateWanStrategy; s = ReplicateWanStrategy(); print(s.format_prompt({'description': {'action': 'test'}}))"`)

---

## Phase 2: Wan-Optimized Formatter

### Overview
Create a Wan-specific formatter that produces prompts using professional cinematography vocabulary, front-loaded subject/action, no metadata labels, within the ~320 token sweet spot.

### Tasks

#### 1. Define cinematography vocabulary mapping
- [x] Add a module-level dict `CAMERA_TERMS` mapping casual terms to professional equivalents, placed near the top of `decode.py` (after imports, before strategy classes)

```python
CAMERA_TERMS = {
    "slow zoom out": "slow dolly out",
    "slow zoom in": "slow dolly in",
    "zoom out": "dolly out",
    "zoom in": "dolly in",
    "follows": "tracking shot follows",
    "moves left": "pan left",
    "moves right": "pan right",
    "moves up": "crane up",
    "moves down": "crane down",
    "shaky": "handheld",
    "smooth movement": "steadicam",
}
```

#### 2. Create `_format_prompt_wan()` helper function
- [x] Add a module-level function `_format_prompt_wan(entry: dict) -> str` below the existing `format_prompt()` function

The Wan formatter follows `Subject → Action → Camera → Style/Atmosphere` order, uses cinematography vocabulary, drops metadata labels, and aims for concise but complete descriptions.

```python
def _format_prompt_wan(entry: dict) -> str:
    """Wan-optimized prompt: Subject > Action > Camera > Style.

    Uses professional cinematography vocabulary, no metadata labels,
    front-loaded content. Targets ~150-200 words to stay within
    Wan's T5 encoder sweet spot (~320 tokens).
    """
    desc = entry["description"]
    parts = []

    # 1. Subject (front-loaded for T5 attention)
    subjects = desc.get("subjects", "")
    if subjects:
        if isinstance(subjects, list):
            subjects = ", ".join(subjects)
        parts.append(subjects.rstrip(".") + ".")

    # 2. Action (core content)
    action = desc.get("action", "")
    if action:
        parts.append(action)

    # 3. Camera (professional terms)
    shot_type = desc.get("shot_type", "")
    camera = desc.get("camera_movement", "")
    if camera:
        camera_lower = camera.lower()
        for casual, pro in CAMERA_TERMS.items():
            if casual in camera_lower:
                camera = camera_lower.replace(casual, pro)
                break
    if shot_type and camera:
        parts.append(f"{shot_type.title()} shot, {camera}.")
    elif shot_type:
        parts.append(f"{shot_type.title()} shot.")
    elif camera:
        parts.append(f"{camera}.")

    # 4. Style/Atmosphere (no labels, just descriptive text)
    setting = desc.get("setting", "")
    if setting:
        parts.append(setting)

    lighting = desc.get("lighting", "")
    if lighting:
        parts.append(lighting)

    palette = desc.get("color_palette", "")
    if palette:
        if isinstance(palette, list):
            palette = " and ".join(palette)
        parts.append(f"{palette} tones.")

    mood = desc.get("mood", "")
    if mood:
        parts.append(f"{mood} atmosphere.")

    return " ".join(parts)
```

#### 3. Override `format_prompt()` on `ReplicateWanStrategy`
- [x] Add `format_prompt` method to `ReplicateWanStrategy` (`decode.py:65`)

```python
def format_prompt(self, entry: dict) -> str:
    return _format_prompt_wan(entry)
```

#### 4. Override `format_prompt()` on `RunPodWanStrategy`
- [x] Add `format_prompt` method to `RunPodWanStrategy` (`decode.py:217`)

```python
def format_prompt(self, entry: dict) -> str:
    return _format_prompt_wan(entry)
```

#### 5. Write tests for Wan formatter
- [x] Add `TestFormatPromptWan` class to `test_decode.py`

```python
from decode import _format_prompt_wan, CAMERA_TERMS, ReplicateWanStrategy

class TestFormatPromptWan:
    def test_subject_action_camera_order(self):
        entry = {
            "description": {
                "shot_type": "wide",
                "camera_movement": "static",
                "action": "A spaceship approaches a planet.",
                "subjects": "Imperial Star Destroyer",
            }
        }
        result = _format_prompt_wan(entry)
        # Subject should appear before action
        subj_pos = result.index("Imperial Star Destroyer")
        action_pos = result.index("spaceship approaches")
        assert subj_pos < action_pos

    def test_no_metadata_labels(self):
        entry = {
            "description": {
                "action": "A door opens.",
                "color_palette": ["red", "gold"],
                "mood": "tense",
            }
        }
        result = _format_prompt_wan(entry)
        assert "Color palette:" not in result
        assert "Mood:" not in result
        assert "red and gold" in result
        assert "tense atmosphere" in result

    def test_camera_term_replacement(self):
        entry = {
            "description": {
                "shot_type": "medium wide",
                "camera_movement": "slow zoom out",
                "action": "Logo recedes.",
            }
        }
        result = _format_prompt_wan(entry)
        assert "dolly out" in result
        assert "zoom out" not in result

    def test_subjects_list_joined(self):
        entry = {
            "description": {
                "action": "Run.",
                "subjects": ["trooper", "droid"],
            }
        }
        result = _format_prompt_wan(entry)
        assert "trooper, droid" in result

    def test_strategy_uses_wan_formatter(self):
        strategy = ReplicateWanStrategy()
        entry = {
            "description": {
                "action": "A ship flies.",
                "mood": "epic",
            }
        }
        result = strategy.format_prompt(entry)
        assert "Mood:" not in result
        assert "epic atmosphere" in result
```

### Success Criteria
- [x] Run: `python -m pytest test_decode.py -v` — all tests pass (old and new)
- [ ] Manual: Run `python -c` snippet to compare old vs new formatting on a real prompts.json entry and verify the output reads well

```bash
python -c "
import json
from decode import format_prompt, _format_prompt_wan
entries = json.load(open('output/star_wars_iv_v2/prompts.json'))
for e in entries[10:12]:
    print('=== OLD ===')
    print(format_prompt(e))
    print()
    print('=== WAN ===')
    print(_format_prompt_wan(e))
    print()
"
```

---

## Phase 3: Seedance-Optimized Formatter

### Overview
Create a Seedance-specific formatter producing ~30-60 word prompts with single action verbs and intensity adverbs.

### Tasks

#### 1. Create `_format_prompt_seedance()` helper function
- [x] Add a module-level function `_format_prompt_seedance(entry: dict) -> str` below `_format_prompt_wan()`

```python
def _format_prompt_seedance(entry: dict) -> str:
    """Seedance-optimized prompt: ~30-60 words, single action, intensity adverbs."""
    desc = entry["description"]
    parts = []

    # Subject + single action verb (Seedance responds best to concise actions)
    subjects = desc.get("subjects", "")
    if subjects:
        if isinstance(subjects, list):
            subjects = subjects[0] if subjects else ""
        parts.append(subjects.rstrip("."))

    action = desc.get("action", "")
    if action:
        # Take just the first sentence for brevity
        first_sentence = action.split(".")[0].strip()
        if first_sentence:
            parts.append(first_sentence.rstrip(".") + ".")

    # Camera as a brief modifier
    camera = desc.get("camera_movement", "")
    if camera and camera.lower() != "static":
        parts.append(camera.rstrip(".") + ".")

    # One atmosphere phrase combining mood + setting
    mood = desc.get("mood", "")
    setting = desc.get("setting", "")
    if mood and setting:
        parts.append(f"{mood.split(',')[0].strip()} {setting.rstrip('.')}")
    elif setting:
        parts.append(setting)
    elif mood:
        parts.append(mood.split(",")[0].strip())

    return " ".join(parts)
```

#### 2. Override `format_prompt()` on `FalSeedanceStrategy`
- [x] Add `format_prompt` method to `FalSeedanceStrategy` (`decode.py:115`). `FalSeedanceProStrategy` inherits from it, so it gets the override automatically.

```python
def format_prompt(self, entry: dict) -> str:
    return _format_prompt_seedance(entry)
```

#### 3. Write tests for Seedance formatter
- [x] Add `TestFormatPromptSeedance` class to `test_decode.py`

```python
from decode import _format_prompt_seedance, FalSeedanceStrategy

class TestFormatPromptSeedance:
    def test_short_output(self):
        entry = {
            "description": {
                "shot_type": "wide",
                "camera_movement": "slow pan left",
                "action": "A spaceship approaches a planet. The planet grows larger in frame. Stars twinkle.",
                "subjects": "Imperial Star Destroyer",
                "lighting": "Harsh rim lighting.",
                "color_palette": ["black", "blue"],
                "mood": "ominous, foreboding",
                "setting": "Deep space.",
            }
        }
        result = _format_prompt_seedance(entry)
        word_count = len(result.split())
        assert word_count <= 60, f"Too long: {word_count} words"

    def test_single_action_sentence(self):
        entry = {
            "description": {
                "action": "A ship flies forward. It turns left. Then it explodes.",
            }
        }
        result = _format_prompt_seedance(entry)
        # Should only include first sentence's content
        assert "turns left" not in result
        assert "explodes" not in result

    def test_static_camera_omitted(self):
        entry = {
            "description": {
                "camera_movement": "static",
                "action": "A figure stands.",
            }
        }
        result = _format_prompt_seedance(entry)
        assert "static" not in result

    def test_subjects_uses_first_only(self):
        entry = {
            "description": {
                "action": "Running.",
                "subjects": ["trooper", "droid", "officer"],
            }
        }
        result = _format_prompt_seedance(entry)
        assert "trooper" in result
        assert "droid" not in result

    def test_strategy_uses_seedance_formatter(self):
        strategy = FalSeedanceStrategy()
        entry = {"description": {"action": "A door opens.", "mood": "tense"}}
        result = strategy.format_prompt(entry)
        word_count = len(result.split())
        assert word_count <= 60

    def test_pro_strategy_inherits_formatter(self):
        from decode import FalSeedanceProStrategy
        strategy = FalSeedanceProStrategy()
        entry = {"description": {"action": "A door opens.", "mood": "tense"}}
        result = strategy.format_prompt(entry)
        word_count = len(result.split())
        assert word_count <= 60
```

### Success Criteria
- [x] Run: `python -m pytest test_decode.py -v` — all tests pass

---

## Phase 4: Negative Prompt for RunPod Wan

### Overview
Add a Chinese negative prompt to the RunPod ComfyUI workflow. Replicate's API doesn't expose this parameter, so it only applies to RunPod.

### Tasks

#### 1. Add default negative prompt to RunPod ComfyUI workflow
- [x] Replace the empty string in the CLIPTextEncode negative prompt node (`decode.py:542-548`, node "5") with a Chinese negative prompt

```python
"5": {
    "class_type": "CLIPTextEncode",
    "inputs": {
        "text": "低质量, 模糊, 变形, 失真, 水印, 文字, 字幕, 低分辨率, 过曝, 欠曝",
        "clip": ["2", 0],
    },
},
```

Translation: "low quality, blurry, deformed, distorted, watermark, text, subtitles, low resolution, overexposed, underexposed"

#### 2. Add a test for the negative prompt
- [x] Add test to verify the workflow contains a non-empty negative prompt

```python
class TestRunPodNegativePrompt:
    def test_workflow_has_negative_prompt(self):
        from decode import RunPodWanStrategy
        strategy = RunPodWanStrategy.__new__(RunPodWanStrategy)
        workflow = strategy._build_workflow("test prompt")
        neg_text = workflow["5"]["inputs"]["text"]
        assert len(neg_text) > 0, "Negative prompt should not be empty"
        # Should be Chinese text
        assert any('\u4e00' <= c <= '\u9fff' for c in neg_text), "Negative prompt should contain Chinese characters"
```

### Success Criteria
- [x] Run: `python -m pytest test_decode.py::TestRunPodNegativePrompt -v` — passes

---

## Phase 5: A/B Test & Experiment Doc

### Overview
Decode shots 10-24 with both old and new Wan formatting using Replicate, then document results.

### Tasks

#### 1. Generate clips with current formatting (control)
- [ ] Manual: Ensure `output/star_wars_iv_v2/prompts.json` has shots 10-24. If not, run encode stages first.
- [ ] Manual: Temporarily revert `ReplicateWanStrategy.format_prompt` to use the base class (old) formatter, then decode shots 10-24

```bash
# Create output dir for control group
mkdir -p output/star_wars_iv_v2/clips/replicate-wan-control

python decode.py output/star_wars_iv_v2 --strategy replicate-wan
# (only shots 10-24 — skip others or use existing clips from experiment 0001)
```

Note: If experiment 0001 already has clips for shots 10-24 with the old formatter, reuse those instead of re-generating. Check `output/star_wars_iv_v2/clips/replicate-wan/` for existing clips.

#### 2. Generate clips with new Wan formatting (treatment)
- [ ] Manual: With the new `ReplicateWanStrategy.format_prompt` in place, decode shots 10-24 to a separate directory

```bash
# Clear progress to force re-generation
# Move existing clips aside, then run decode
mkdir -p output/star_wars_iv_v2/clips/replicate-wan-new
python decode.py output/star_wars_iv_v2 --strategy replicate-wan
```

#### 3. Create experiment document
- [ ] Create `docs/research/experiments/0002-prompt-reformatting-ab-test.md`

```markdown
---
id: experiment-0002
type: experiment
purpose: "A/B test comparing original prose prompts vs Wan-optimized cinematography prompts on shots 10-24."
tags: ["experiment", "prompts", "wan", "video-generation"]
related: ["../../tasks/0004-prompt-reformatting/task.md", "./0001-first-e2e-decode-test.md"]
---

# Experiment 0002: Prompt Reformatting A/B Test

## Hypothesis

Wan-optimized prompts (front-loaded subject/action, cinematography vocabulary, no metadata labels) will produce more accurate and visually coherent video clips compared to the current prose-style prompts.

## Setup

- **Source**: Star Wars EP IV, shots 10-24
- **Model**: Replicate wan-video/wan-2.2-t2v-fast, 832x480, 81 frames @ 16fps
- **Control**: Current `format_prompt()` — prose with "Color palette:", "Mood:" labels
- **Treatment**: `_format_prompt_wan()` — Subject > Action > Camera > Style, cinematography terms
- **Cost**: ~$0.75 per group ($1.50 total)

## Prompt Comparison

### Shot [N]

**Control prompt:**
> [paste old format]

**Treatment prompt:**
> [paste new format]

[Repeat for notable shots]

## Results

### Per-Shot Observations

| Shot | Control | Treatment | Notes |
|------|---------|-----------|-------|
| 10   |         |           |       |
| ...  |         |           |       |
| 24   |         |           |       |

### Summary

[Overall observations — which approach produced better results and why]

## Recommendation

[Keep new formatter / revert / iterate further]
```

#### 4. Run comparison and fill in results
- [ ] Manual: Use the compare tool (`python tools/serve.py`) to view control vs treatment clips side-by-side
- [ ] Manual: Fill in observations in the experiment doc
- [ ] Manual: Write summary and recommendation

### Success Criteria
- [ ] Manual: Experiment doc exists with filled-in observations for all 15 shots
- [ ] Manual: Clear recommendation on whether to keep the new formatter

---

## Final Checklist

- [ ] All phases complete
- [ ] All tests passing: `python -m pytest test_decode.py -v`
- [ ] Old module-level `format_prompt()` still exists and works (used by base class default)
- [ ] Experiment doc written with results and recommendation

## Documentation Updates

- [ ] Create experiment doc `docs/research/experiments/0002-prompt-reformatting-ab-test.md`
- [ ] Update `docs/research/0003-prompt-to-video/research.md` open question #3 to reference this work

## References

- Task: `docs/tasks/0004-prompt-reformatting/task.md`
- Research: `docs/research/0003-prompt-to-video/research.md`
- Experiment baseline: `docs/research/experiments/0001-first-e2e-decode-test.md`
