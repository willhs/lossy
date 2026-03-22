"""Tests for decode.py strategy pattern and duration logic."""

import json
import os

import pytest

from decode import (
    AudioClipResult,
    CAMERA_TERMS,
    ClipResult,
    ElevenLabsStrategy,
    FalSeedanceStrategy,
    FalSeedanceProStrategy,
    MMAudioStrategy,
    ReplicateWanStrategy,
    RunPodMMAudioStrategy,
    RunPodWanStrategy,
    SpeechClipResult,
    SpeechStrategy,
    _format_prompt_seedance,
    _format_prompt_wan,
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


# ---------------------------------------------------------------------------
# RunPodWanStrategy
# ---------------------------------------------------------------------------

class TestRunPodWanStrategy:
    def test_name(self):
        assert RunPodWanStrategy.name == "runpod-wan"

    def test_clip_duration_constant(self):
        assert RunPodWanStrategy.CLIP_DURATION == pytest.approx(5.0625)

    def test_build_workflow_structure(self):
        strategy = RunPodWanStrategy()
        workflow = strategy._build_workflow("a cat walking", seed=42)

        # Has all required nodes
        assert "1" in workflow  # UNETLoader
        assert "8" in workflow  # KSampler
        assert "10" in workflow  # SaveAnimatedWEBP

        # Prompt is injected
        assert workflow["4"]["inputs"]["text"] == "a cat walking"

        # Seed is injected
        assert workflow["8"]["inputs"]["seed"] == 42

        # Output dimensions are 480p 16:9
        assert workflow["7"]["inputs"]["width"] == 848
        assert workflow["7"]["inputs"]["height"] == 480
        assert workflow["7"]["inputs"]["length"] == 81

        # Model filenames match what we download
        assert "wan2.1" in workflow["1"]["inputs"]["unet_name"]
        assert "umt5_xxl" in workflow["2"]["inputs"]["clip_name"]

    def test_build_workflow_has_chinese_negative_prompt(self):
        strategy = RunPodWanStrategy()
        workflow = strategy._build_workflow("test prompt", seed=1)
        neg_text = workflow["5"]["inputs"]["text"]
        assert len(neg_text) > 0, "Negative prompt should not be empty"
        assert any('\u4e00' <= c <= '\u9fff' for c in neg_text), "Negative prompt should contain Chinese characters"


# ---------------------------------------------------------------------------
# Wan-optimized formatter
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Seedance-optimized formatter
# ---------------------------------------------------------------------------

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
        strategy = FalSeedanceProStrategy()
        entry = {"description": {"action": "A door opens.", "mood": "tense"}}
        result = strategy.format_prompt(entry)
        word_count = len(result.split())
        assert word_count <= 60


# ---------------------------------------------------------------------------
# ElevenLabsStrategy._target_durations
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# MMAudioStrategy._target_durations
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# AudioClipResult
# ---------------------------------------------------------------------------

class TestAudioClipResult:
    def test_fields(self):
        result = AudioClipResult(path="/tmp/0001.mp3", actual_duration_s=5.0, cost=0.01)
        assert result.path == "/tmp/0001.mp3"
        assert result.actual_duration_s == 5.0
        assert result.cost == 0.01


# ---------------------------------------------------------------------------
# SpeechClipResult
# ---------------------------------------------------------------------------

class TestSpeechClipResult:
    def test_fields(self):
        result = SpeechClipResult(
            path="/tmp/0001-00.mp3", duration_s=2.5, offset_s=1.0, cost=0.005
        )
        assert result.path == "/tmp/0001-00.mp3"
        assert result.duration_s == 2.5
        assert result.offset_s == 1.0
        assert result.cost == 0.005


# ---------------------------------------------------------------------------
# SpeechStrategy
# ---------------------------------------------------------------------------

class TestSpeechStrategy:
    def test_default_voice(self):
        s = SpeechStrategy()
        assert s.voice == "Roger"

    def test_custom_voice(self):
        s = SpeechStrategy(voice="Alice")
        assert s.voice == "Alice"

    def test_model_id(self):
        assert SpeechStrategy.MODEL_ID == "fal-ai/elevenlabs/tts/turbo-v2.5"

    def test_cost_constant(self):
        assert SpeechStrategy.COST_PER_1K_CHARS == 0.05


# ---------------------------------------------------------------------------
# RunPodMMAudioStrategy._target_durations
# ---------------------------------------------------------------------------

class TestRunPodMMAudioTargetDurations:
    def setup_method(self):
        # Bypass __init__'s RunPodSession import
        self.strategy = RunPodMMAudioStrategy.__new__(RunPodMMAudioStrategy)

    def test_short_clip(self):
        assert self.strategy._target_durations(5.0) == [5.0]

    def test_at_max(self):
        assert self.strategy._target_durations(30.0) == [30.0]

    def test_over_max(self):
        assert self.strategy._target_durations(35.0) == [30.0, 5.0]

    def test_much_over_max(self):
        assert self.strategy._target_durations(65.0) == [30.0, 30.0, 5.0]

    def test_very_short(self):
        assert self.strategy._target_durations(0.3) == [1.0]

    def test_zero(self):
        assert self.strategy._target_durations(0.0) == [1.0]
