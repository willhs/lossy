"""Tests for decode.py strategy pattern and duration logic."""

import json
import os
import threading
from unittest.mock import MagicMock, patch

import pytest

from decode import (
    AudioClipResult,
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
    filter_speech_from_sound,
    format_prompt,
    vary_prompt_for_part,
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
# CharacterIdentityMixin.format_prompt -- locked verbatim character strings
# ---------------------------------------------------------------------------

class TestCharacterIdentityMixin:
    """The mixin must prepend the exact same string for a character in every
    shot it appears in -- identical tokens hold continuity, not more detail.
    There must be no LLM rewording step in this path (see prompt_blend.py removal).
    """

    def _make_strategy(self, character_shot_map, characters_data):
        from strategies_video import CharacterIdentityMixin, GenerationStrategy

        class _FakeStrategy(CharacterIdentityMixin, GenerationStrategy):
            def format_prompt(self, entry):
                return super().format_prompt(entry)

        strategy = _FakeStrategy()
        strategy._init_character_identity(character_shot_map, characters_data)
        return strategy

    def test_identity_prefix_identical_across_shots(self):
        characters_data = {
            "characters": [
                {"name": "luke", "display_name": "Luke Skywalker",
                 "description": "A young man in a beige tunic with sandy blond hair."},
            ]
        }
        character_shot_map = {0: ["luke"], 5: ["luke"]}
        strategy = self._make_strategy(character_shot_map, characters_data)

        entry0 = {"index": 0, "description": {"action": "Luke looks at the twin suns."}}
        entry5 = {"index": 5, "description": {"action": "Luke ignites his lightsaber."}}

        result0 = strategy.format_prompt(entry0)
        result5 = strategy.format_prompt(entry5)

        # The description is the identity; the name is deliberately not in it.
        prefix = "A young man in a beige tunic with sandy blond hair."
        assert result0.startswith(prefix)
        assert result5.startswith(prefix)
        assert "Luke" not in result0 and "Luke" not in result5
        # Only the base shot text should differ -- the identity block is byte-identical.
        assert result0[: len(prefix)] == result5[: len(prefix)]

    def test_no_characters_in_shot_falls_back_to_base(self):
        strategy = self._make_strategy({}, {"characters": []})
        entry = {"index": 0, "description": {"action": "An empty corridor."}}
        result = strategy.format_prompt(entry)
        assert "An empty corridor." in result
        assert ":" not in result.split(".")[0]

    def test_no_blended_prompts_support(self):
        """The mixin no longer accepts or consults a blended_prompts arg."""
        import inspect
        from strategies_video import CharacterIdentityMixin

        sig = inspect.signature(CharacterIdentityMixin._init_character_identity)
        assert "blended_prompts" not in sig.parameters


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

    def test_frame_constants(self):
        assert RunPodWanStrategy.FPS == 16
        assert RunPodWanStrategy.MIN_FRAMES == 33
        assert RunPodWanStrategy.MAX_FRAMES == 97
        # All frame constants must satisfy 4n+1
        assert (RunPodWanStrategy.MIN_FRAMES - 1) % 4 == 0
        assert (RunPodWanStrategy.MAX_FRAMES - 1) % 4 == 0

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
        assert workflow["7"]["inputs"]["length"] == 81  # default

        # Model filenames match what we download
        assert "wan2.1" in workflow["1"]["inputs"]["unet_name"]
        assert "umt5_xxl" in workflow["2"]["inputs"]["clip_name"]

    def test_build_workflow_custom_length(self):
        strategy = RunPodWanStrategy()
        workflow = strategy._build_workflow("test", seed=1, length=49)
        assert workflow["7"]["inputs"]["length"] == 49

    def test_build_workflow_has_chinese_negative_prompt(self):
        strategy = RunPodWanStrategy()
        workflow = strategy._build_workflow("test prompt", seed=1)
        neg_text = workflow["5"]["inputs"]["text"]
        assert len(neg_text) > 0, "Negative prompt should not be empty"
        assert any('\u4e00' <= c <= '\u9fff' for c in neg_text), "Negative prompt should contain Chinese characters"


class TestRunPodWanTargetFrames:
    """Tests for RunPodWanStrategy._target_frames()."""

    def setup_method(self):
        self.strategy = RunPodWanStrategy()

    def test_short_shots_clamped_to_min(self):
        assert self.strategy._target_frames(0.5) == 33
        assert self.strategy._target_frames(1.0) == 33
        assert self.strategy._target_frames(1.5) == 33

    def test_normal_durations(self):
        assert self.strategy._target_frames(3.0) == 49   # 3.0 * 16 = 48 -> (48-1)/4 = 11.75 -> round(11.75) = 12 -> 12*4+1 = 49
        assert self.strategy._target_frames(5.0) == 81   # 5.0 * 16 = 80 -> (80-1)/4 = 19.75 -> round(19.75) = 20 -> 20*4+1 = 81
        assert self.strategy._target_frames(6.0) == 97   # 6.0 * 16 = 96 -> (96-1)/4 = 23.75 -> round(23.75) = 24 -> 24*4+1 = 97

    def test_long_shots_clamped_to_max(self):
        assert self.strategy._target_frames(9.0) == 97
        assert self.strategy._target_frames(15.0) == 97

    def test_boundary_values(self):
        # MIN_FRAMES = 33 -> 33/16 = 2.0625s
        assert self.strategy._target_frames(2.0) == 33
        # MAX_FRAMES = 97 -> 97/16 = 6.0625s
        assert self.strategy._target_frames(6.0) == 97

    def test_result_always_4n_plus_1(self):
        for duration in [0.5, 1.0, 2.0, 3.0, 3.7, 4.5, 5.0, 6.3, 7.0, 8.0, 10.0]:
            frames = self.strategy._target_frames(duration)
            assert (frames - 1) % 4 == 0, f"duration={duration} -> frames={frames} not 4n+1"

    def test_result_always_in_range(self):
        for duration in [0.1, 0.5, 1.0, 3.0, 5.0, 8.0, 15.0, 30.0]:
            frames = self.strategy._target_frames(duration)
            assert 33 <= frames <= 97, f"duration={duration} -> frames={frames} out of range"


class TestRunPodWanTargetDurations:
    """Tests for RunPodWanStrategy._target_durations()."""

    def setup_method(self):
        self.strategy = RunPodWanStrategy()

    def test_short_shot_single_clip(self):
        result = self.strategy._target_durations(3.0)
        assert len(result) == 1
        assert result[0] == 49

    def test_medium_shot_single_clip(self):
        result = self.strategy._target_durations(5.0)
        assert len(result) == 1
        assert result[0] == 81

    def test_max_duration_single_clip(self):
        result = self.strategy._target_durations(6.0)
        assert len(result) == 1
        assert result[0] == 97

    def test_long_shot_splits(self):
        # 15s -> 6.06s + 6.06s + 2.88s
        result = self.strategy._target_durations(15.0)
        assert len(result) >= 2
        assert result[0] == 97  # max chunk
        for frames in result:
            assert (frames - 1) % 4 == 0  # all valid 4n+1

    def test_very_long_shot_splits(self):
        # 25s -> multiple ~6s chunks
        result = self.strategy._target_durations(25.0)
        assert len(result) >= 4
        assert result[0] == 97
        assert result[1] == 97
        for frames in result:
            assert 33 <= frames <= 97

    def test_all_parts_valid_frame_counts(self):
        for duration in [3.0, 8.0, 12.0, 20.0, 35.0]:
            parts = self.strategy._target_durations(duration)
            for frames in parts:
                assert (frames - 1) % 4 == 0, f"duration={duration}: {frames} not 4n+1"
                assert 33 <= frames <= 97, f"duration={duration}: {frames} out of range"


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
# FalSeedanceStrategy.generate -- authored-manifest field tolerance
#
# Authored (non-source-film) manifests never populate encode-only fields
# (camera_motion_detected, audio_detected, temporal_segments). These are
# never read on the decode side except temporal_segments, which is
# consulted with .get() only for split (multi-part) shots -- absence must
# fall back to vary_prompt_for_part, not error. See docs/research/
# 0024-authored-manifest-verification/research.md.
# ---------------------------------------------------------------------------

class TestFalSeedanceGenerateMissingEncodeFields:
    def test_split_shot_without_temporal_segments_falls_back(self, tmp_path):
        """entry has no temporal_segments (authored manifest) and the target
        duration forces a multi-part split -- generate() must use
        vary_prompt_for_part instead of KeyError/crashing."""
        strategy = FalSeedanceStrategy()
        entry = {"description": {"action": "A gnome walks deeper into the cave."},
                  "duration_s": 15.0}
        fake_video = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 100
        mock_resp = MagicMock(content=fake_video, raise_for_status=MagicMock())

        with patch("fal_client.subscribe", return_value={"video": {"url": "https://example.com/c.mp4"}}):
            with patch("httpx.get", return_value=mock_resp):
                with patch("manifest.probe_duration", return_value=12.0):
                    results = strategy.generate(
                        "A gnome walks.", str(tmp_path), 0, 15.0, seed=0, entry=entry)

        assert len(results) == 2  # [12, 3] split per _target_durations
        assert (tmp_path / "0000-01.mp4").exists()
        assert (tmp_path / "0000-02.mp4").exists()

    def test_single_shot_missing_all_encode_fields(self, tmp_path):
        """A minimal authored entry -- no camera_motion_detected, audio_detected,
        or temporal_segments -- generates one clip without error."""
        strategy = FalSeedanceStrategy()
        entry = {"index": 0, "start_s": 0.0, "end_s": 3.0, "duration_s": 3.0,
                  "description": {"action": "A gnome stands at a cave mouth."}}
        fake_video = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 100
        mock_resp = MagicMock(content=fake_video, raise_for_status=MagicMock())

        with patch("fal_client.subscribe", return_value={"video": {"url": "https://example.com/c.mp4"}}):
            with patch("httpx.get", return_value=mock_resp):
                with patch("manifest.probe_duration", return_value=3.0):
                    results = strategy.generate(
                        strategy.format_prompt(entry), str(tmp_path), 0, 3.0, seed=0, entry=entry)

        assert len(results) == 1
        assert (tmp_path / "0000.mp4").exists()


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
        assert self.strategy._target_durations(10.0) == [10.0]

    def test_over_max(self):
        assert self.strategy._target_durations(35.0) == [10.0, 10.0, 10.0, 5.0]

    def test_very_short(self):
        result = self.strategy._target_durations(0.3)
        assert result == [5.0]  # Clamped to MIN_DURATION


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
        assert self.strategy._target_durations(10.0) == [10.0]

    def test_over_max(self):
        assert self.strategy._target_durations(35.0) == [10.0, 10.0, 10.0, 5.0]

    def test_much_over_max(self):
        assert self.strategy._target_durations(65.0) == [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 5.0]

    def test_very_short(self):
        assert self.strategy._target_durations(0.3) == [5.0]

    def test_zero(self):
        assert self.strategy._target_durations(0.0) == [5.0]


# ---------------------------------------------------------------------------
# RunPodWanStrategy concurrent audio pipelining
# ---------------------------------------------------------------------------

class TestRunPodWanConcurrentAudio:
    """Tests for the concurrent audio pipelining added to RunPodWanStrategy."""

    def _make_strategy(self, concurrent_audio=True, audio_capable=True):
        """Create a RunPodWanStrategy with mocked internals."""
        strategy = RunPodWanStrategy.__new__(RunPodWanStrategy)
        strategy._session = MagicMock()
        strategy._setup_done = True
        strategy._concurrent_audio = concurrent_audio
        strategy._audio_capable = audio_capable
        strategy._audio_thread = None
        strategy._audio_results = {}
        strategy._audio_failures = []
        strategy._audio_dir = "/tmp/test_audio"
        strategy._pending_audio = None
        return strategy

    def test_generate_accepts_entry_kwarg(self):
        """All strategies should accept the entry kwarg without error."""
        strategy = self._make_strategy(audio_capable=False)
        strategy._ensure_pod = MagicMock()
        strategy._generate_one_clip = MagicMock(return_value=ClipResult(
            path="/tmp/0001.mp4", actual_duration_s=5.0, cost=0.01,
        ))

        entry = {"description": {"sound": "Wind blowing"}, "duration_s": 5.0}
        results = strategy.generate("test prompt", "/tmp/clips", 1, 5.0, seed=1, entry=entry)
        assert len(results) == 1

    def test_audio_queued_when_capable(self):
        """generate() should queue audio when audio_capable and entry has sound."""
        strategy = self._make_strategy(audio_capable=True)
        strategy._ensure_pod = MagicMock()
        strategy._generate_one_clip = MagicMock(return_value=ClipResult(
            path="/tmp/0001.mp4", actual_duration_s=5.0, cost=0.01,
        ))

        entry = {"description": {"sound": "Wind blowing"}, "duration_s": 5.0}
        strategy.generate("test prompt", "/tmp/clips", 1, 5.0, seed=1, entry=entry)

        assert strategy._pending_audio is not None
        assert strategy._pending_audio[0] == 1  # shot_index
        assert strategy._pending_audio[1] == "Wind blowing"  # sound

    def test_audio_not_queued_when_disabled(self):
        """generate() should not queue audio when audio_capable is False."""
        strategy = self._make_strategy(audio_capable=False)
        strategy._ensure_pod = MagicMock()
        strategy._generate_one_clip = MagicMock(return_value=ClipResult(
            path="/tmp/0001.mp4", actual_duration_s=5.0, cost=0.01,
        ))

        entry = {"description": {"sound": "Wind blowing"}, "duration_s": 5.0}
        strategy.generate("test prompt", "/tmp/clips", 1, 5.0, seed=1, entry=entry)

        assert strategy._pending_audio is None

    def test_audio_not_queued_when_no_sound(self):
        """generate() should not queue audio when entry has no sound description."""
        strategy = self._make_strategy(audio_capable=True)
        strategy._ensure_pod = MagicMock()
        strategy._generate_one_clip = MagicMock(return_value=ClipResult(
            path="/tmp/0001.mp4", actual_duration_s=5.0, cost=0.01,
        ))

        entry = {"description": {"action": "A door opens"}, "duration_s": 5.0}
        strategy.generate("test prompt", "/tmp/clips", 1, 5.0, seed=1, entry=entry)

        assert strategy._pending_audio is None

    def test_pending_audio_fired_on_next_generate(self):
        """Pending audio from shot N should fire at the start of shot N+1."""
        strategy = self._make_strategy(audio_capable=True)
        strategy._ensure_pod = MagicMock()
        strategy._generate_one_clip = MagicMock(return_value=ClipResult(
            path="/tmp/0001.mp4", actual_duration_s=5.0, cost=0.01,
        ))
        strategy._start_audio_thread = MagicMock()

        # Set pending audio from previous shot
        strategy._pending_audio = (0, "Thunder rumbling", 4.0, 0, "/tmp/audio")

        entry = {"description": {"sound": "Rain falling"}, "duration_s": 5.0}
        strategy.generate("test prompt", "/tmp/clips", 1, 5.0, seed=1, entry=entry)

        # Previous audio should have been fired
        strategy._start_audio_thread.assert_called_once_with(0, "Thunder rumbling", 4.0, 0, "/tmp/audio")
        # New audio should be queued
        assert strategy._pending_audio[0] == 1
        assert strategy._pending_audio[1] == "Rain falling"

    def test_collect_audio_thread_noop_when_none(self):
        """_collect_audio_thread should be safe to call with no thread."""
        strategy = self._make_strategy()
        strategy._audio_thread = None
        strategy._collect_audio_thread()  # should not raise
        assert strategy._audio_thread is None

    def test_collect_audio_thread_joins_finished_thread(self):
        """_collect_audio_thread should join a completed thread."""
        strategy = self._make_strategy()
        t = threading.Thread(target=lambda: None)
        t.start()
        t.join()  # ensure it finishes
        strategy._audio_thread = t
        strategy._collect_audio_thread()
        assert strategy._audio_thread is None

    def test_finish_audio_fires_pending(self):
        """finish_audio() should fire and wait for the last pending audio."""
        strategy = self._make_strategy(audio_capable=True)
        strategy._start_audio_thread = MagicMock()
        strategy._collect_audio_thread = MagicMock()
        strategy._pending_audio = (5, "Explosion", 3.0, 5, "/tmp/audio")

        strategy.finish_audio()

        strategy._start_audio_thread.assert_called_once_with(5, "Explosion", 3.0, 5, "/tmp/audio")
        strategy._collect_audio_thread.assert_called_once()
        assert strategy._pending_audio is None

    def test_finish_audio_noop_when_nothing_pending(self):
        """finish_audio() should just collect when nothing is pending."""
        strategy = self._make_strategy(audio_capable=True)
        strategy._start_audio_thread = MagicMock()
        strategy._collect_audio_thread = MagicMock()
        strategy._pending_audio = None

        strategy.finish_audio()

        strategy._start_audio_thread.assert_not_called()
        strategy._collect_audio_thread.assert_called_once()

    def test_get_audio_results_returns_accumulated(self):
        """get_audio_results() should return results and failures."""
        strategy = self._make_strategy()
        from clip_types import AudioClipResult
        strategy._audio_results = {
            0: [AudioClipResult(path="/tmp/0000.flac", actual_duration_s=5.0, cost=0.0)],
            1: [AudioClipResult(path="/tmp/0001.flac", actual_duration_s=3.0, cost=0.0)],
        }
        strategy._audio_failures = [2]

        results, failures = strategy.get_audio_results()
        assert len(results) == 2
        assert failures == [2]

    def test_setup_audio_skips_low_vram(self):
        """_setup_audio should disable audio when GPU VRAM < 24 GB."""
        strategy = self._make_strategy(audio_capable=False)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "16000\n"
        strategy._session.ssh_cmd = MagicMock(return_value=mock_result)

        strategy._setup_audio()

        assert strategy._audio_capable is False

    def test_setup_audio_enables_on_sufficient_vram(self):
        """_setup_audio should enable audio when GPU VRAM >= 24 GB."""
        strategy = self._make_strategy(audio_capable=False)

        # Mock all ssh_cmd calls in order
        vram_result = MagicMock(returncode=0, stdout="49140\n")
        ok_result = MagicMock(returncode=0, stdout="OK\n", stderr="")
        strategy._session.ssh_cmd = MagicMock(side_effect=[
            vram_result,  # nvidia-smi
            ok_result,    # upload script
            ok_result,    # pip install
            ok_result,    # download weights
        ])

        strategy._setup_audio()

        assert strategy._audio_capable is True


# ---------------------------------------------------------------------------
# filter_speech_from_sound — SPEC-100: Speech-Cue Filter for MMAudio
# ---------------------------------------------------------------------------


class TestSpeechKeywordDetection:
    """Tests for keyword detection requirements (SPEC-100, REQ-001 to REQ-009)."""

    @pytest.mark.req("SPEC-100/REQ-001")
    @pytest.mark.parametrize("keyword", ["voice", "voices"])
    def test_detects_voice(self, keyword):
        """REQ-001: Shall detect voice/voices as speech."""
        desc = f"the {keyword} of the crowd, accompanied by wind and rain"
        result = filter_speech_from_sound(desc)
        assert result is None or keyword not in result.lower()

    @pytest.mark.req("SPEC-100/REQ-002")
    @pytest.mark.parametrize("keyword", ["speaking", "speaks", "spoken", "speech"])
    def test_detects_speaking_variants(self, keyword):
        """REQ-002: Shall detect speaking/speaks/spoken/speech."""
        desc = f"a man {keyword} softly, accompanied by the hum of engines"
        result = filter_speech_from_sound(desc)
        assert result is None or keyword not in result.lower()

    @pytest.mark.req("SPEC-100/REQ-003")
    @pytest.mark.parametrize("keyword", ["dialogue", "talking", "talks", "conversation"])
    def test_detects_dialogue_variants(self, keyword):
        """REQ-003: Shall detect dialogue, talking/talks, conversation."""
        desc = f"quiet {keyword} in the background, accompanied by rain on the roof"
        result = filter_speech_from_sound(desc)
        assert result is None or keyword not in result.lower()

    @pytest.mark.req("SPEC-100/REQ-004")
    @pytest.mark.parametrize("keyword", ["says", "saying"])
    def test_detects_says(self, keyword):
        """REQ-004: Shall detect says/saying."""
        desc = f"someone {keyword} something, wind howling outside"
        result = filter_speech_from_sound(desc)
        assert result is None or keyword not in result.lower()

    @pytest.mark.req("SPEC-100/REQ-005")
    @pytest.mark.parametrize("keyword", [
        "shout", "shouting", "whisper", "whispering",
        "scream", "screaming", "yell", "yelling",
    ])
    def test_detects_loud_speech_verbs(self, keyword):
        """REQ-005: Shall detect shout/whisper/scream/yell variants."""
        desc = f"a man {keyword} in the distance, wind howling, thunder rumbling"
        result = filter_speech_from_sound(desc)
        assert result is None or keyword not in result.lower()

    @pytest.mark.req("SPEC-100/REQ-006")
    @pytest.mark.parametrize("keyword", [
        "murmur", "murmuring", "narrates", "narrating", "narration",
    ])
    def test_detects_quiet_speech(self, keyword):
        """REQ-006: Shall detect murmur/narrate variants."""
        desc = f"soft {keyword} in the room, accompanied by ticking clocks"
        result = filter_speech_from_sound(desc)
        assert result is None or keyword not in result.lower()

    @pytest.mark.req("SPEC-100/REQ-007")
    def test_detects_vocal(self):
        """REQ-007: Shall detect vocal."""
        desc = "a vocal performance, accompanied by orchestral strings"
        result = filter_speech_from_sound(desc)
        assert result is None or "vocal" not in result.lower()

    @pytest.mark.req("SPEC-100/REQ-008")
    def test_case_insensitive(self):
        """REQ-008: Detection shall be case-insensitive."""
        desc = "LOUD SHOUTING, accompanied by the hum of engines"
        result = filter_speech_from_sound(desc)
        assert result is None or "shouting" not in result.lower()

    @pytest.mark.req("SPEC-100/REQ-009")
    @pytest.mark.parametrize("desc", [
        "The heavy mechanical breathing and the hum of engines",
        "electronic beeps and chirps of R2-D2",
        "metallic footsteps on the floor",
        "a distant explosion and rumbling thunder",
    ])
    def test_preserves_non_speech(self, desc):
        """REQ-009: Shall NOT flag breathing, beeps, chirps, footsteps, etc."""
        assert filter_speech_from_sound(desc) == desc


class TestFilteringBehavior:
    """Tests for filtering behavior requirements (SPEC-100, REQ-010 to REQ-015)."""

    @pytest.mark.req("SPEC-100/REQ-010")
    def test_pure_sfx_unchanged(self):
        """REQ-010: Pure SFX/ambient descriptions pass through unchanged."""
        desc = "The sound of laser fire and explosions"
        assert filter_speech_from_sound(desc) == desc

    @pytest.mark.req("SPEC-100/REQ-011")
    def test_entirely_speech_returns_none(self):
        """REQ-011: Only-speech descriptions return None."""
        assert filter_speech_from_sound(
            "the sound of two characters speaking in hushed dialogue"
        ) is None

    @pytest.mark.req("SPEC-100/REQ-012")
    def test_empty_returns_none(self):
        """REQ-012: Empty input returns None."""
        assert filter_speech_from_sound("") is None

    @pytest.mark.req("SPEC-100/REQ-012")
    def test_short_remainder_returns_none(self):
        """REQ-012: Remainder < 5 chars returns None."""
        assert filter_speech_from_sound("he says, ok") is None

    @pytest.mark.req("SPEC-100/REQ-013")
    def test_mixed_preserves_sfx(self):
        """REQ-013: Mixed descriptions preserve SFX, remove speech."""
        desc = (
            "The heavy mechanical breathing of Darth Vader, "
            "accompanied by the strained voice of the trooper, "
            "and the ambient hum of ship engines."
        )
        result = filter_speech_from_sound(desc)
        assert result is not None
        assert "voice" not in result
        assert "breathing" in result
        assert "hum" in result

    @pytest.mark.req("SPEC-100/REQ-013")
    def test_multiple_speech_keywords_removed(self):
        """REQ-013: All speech keywords removed, SFX kept."""
        desc = "a man shouting in the distance, wind howling, murmuring crowd"
        result = filter_speech_from_sound(desc)
        assert result is not None
        assert "howling" in result
        assert "shouting" not in result
        assert "murmuring" not in result

    @pytest.mark.req("SPEC-100/REQ-014")
    def test_speech_clause_dropped_entirely(self):
        """REQ-014: When speech is the main subject, drop the whole clause."""
        desc = (
            "C-3PO's anxious voice speaking, "
            "accompanied by electronic beeps and whistles"
        )
        result = filter_speech_from_sound(desc)
        assert result is not None
        assert "voice" not in result
        assert "beeps" in result

    @pytest.mark.req("SPEC-100/REQ-015")
    def test_speech_list_item_removed(self):
        """REQ-015: Speech list items removed, other items preserved."""
        desc = (
            "Intense, rapid-fire laser blasts, the sound of explosions, "
            "shouting, and the thud of bodies hitting the floor."
        )
        result = filter_speech_from_sound(desc)
        assert result is not None
        assert "shouting" not in result
        assert "laser blasts" in result
        assert "explosions" in result
        assert "thud" in result


class TestStructuralIntegrity:
    """Tests for structural integrity requirements (SPEC-100, REQ-020 to REQ-024)."""

    @pytest.mark.req("SPEC-100/REQ-020")
    def test_no_dangling_adjectives(self):
        """REQ-020: No dangling adjectives from possessive phrases."""
        desc = (
            "The sound of C-3PO's anxious, high-pitched voice speaking, "
            "accompanied by the rhythmic, electronic beeps and chirps of R2-D2."
        )
        result = filter_speech_from_sound(desc)
        assert result is not None
        assert "voice" not in result
        assert "beeps" in result
        assert "anxious," not in result

    @pytest.mark.req("SPEC-100/REQ-021")
    def test_and_joined_voice_removed(self):
        """REQ-021: 'and [poss] voice [verb]' removed, preceding content kept."""
        desc = (
            "The sound of C-3PO's metallic, shuffling footsteps "
            "and his distinct, anxious voice calling out, "
            "accompanied by the low, rhythmic hum of a spaceship."
        )
        result = filter_speech_from_sound(desc)
        assert result is not None
        assert "voice" not in result
        assert "footsteps" in result
        assert "hum" in result

    @pytest.mark.req("SPEC-100/REQ-022")
    def test_clause_boundaries_respected(self):
        """REQ-022: Non-speech clauses preserved across clause boundaries."""
        desc = (
            "The sound of C-3PO's metallic, anxious voice speaking, "
            "accompanied by R2-D2's characteristic electronic beeps and whistles. "
            "The background includes the low, steady hum of a starship engine."
        )
        result = filter_speech_from_sound(desc)
        assert result is not None
        assert "voice" not in result
        assert "beeps" in result
        assert "hum" in result

    @pytest.mark.req("SPEC-100/REQ-023")
    def test_capitalizes_after_removal(self):
        """REQ-023: First letter capitalized when start of text was removed."""
        desc = "dialogue between characters, accompanied by the hum of engines"
        result = filter_speech_from_sound(desc)
        assert result is not None
        assert result[0].isupper()

    @pytest.mark.req("SPEC-100/REQ-024")
    def test_cleans_punctuation_artifacts(self):
        """REQ-024: No double commas, orphaned semicolons, etc."""
        desc = (
            "orchestral score; the narrator speaking, "
            "accompanied by wind and rain"
        )
        result = filter_speech_from_sound(desc)
        assert result is not None
        assert ",," not in result
        assert ";," not in result


# ---------------------------------------------------------------------------
# vary_prompt_for_part
# ---------------------------------------------------------------------------

class TestVaryPromptForPart:
    """Tests for the temporal progression cue fallback (SPEC-300, REQ-020 to REQ-024)."""

    @pytest.mark.req("SPEC-300/REQ-024")
    def test_single_part_unchanged(self):
        """REQ-024: Single-part shot prompt returned unchanged."""
        assert vary_prompt_for_part("A ship flies.", 0, 1) == "A ship flies."

    @pytest.mark.req("SPEC-300/REQ-024")
    def test_zero_total_unchanged(self):
        """REQ-024: Zero total_parts returns prompt unchanged."""
        assert vary_prompt_for_part("A ship flies.", 0, 0) == "A ship flies."

    @pytest.mark.req("SPEC-300/REQ-020")
    def test_first_of_two(self):
        """REQ-020: First part gets 'Beginning of the action.' appended."""
        result = vary_prompt_for_part("A ship flies.", 0, 2)
        assert result == "A ship flies. Beginning of the action."

    @pytest.mark.req("SPEC-300/REQ-021")
    def test_last_of_two(self):
        """REQ-021: Last part gets 'The action concludes.' appended."""
        result = vary_prompt_for_part("A ship flies.", 1, 2)
        assert result == "A ship flies. The action concludes."

    @pytest.mark.req("SPEC-300/REQ-020")
    def test_first_of_three(self):
        """REQ-020: First part of 3 gets beginning cue."""
        result = vary_prompt_for_part("A door opens.", 0, 3)
        assert result == "A door opens. Beginning of the action."

    @pytest.mark.req("SPEC-300/REQ-022")
    def test_middle_of_three(self):
        """REQ-022: Middle part gets 'The action continues.' appended."""
        result = vary_prompt_for_part("A door opens.", 1, 3)
        assert result == "A door opens. The action continues."

    @pytest.mark.req("SPEC-300/REQ-021")
    def test_last_of_three(self):
        """REQ-021: Last part of 3 gets conclusion cue."""
        result = vary_prompt_for_part("A door opens.", 2, 3)
        assert result == "A door opens. The action concludes."

    @pytest.mark.req("SPEC-300/REQ-022")
    def test_middle_of_four(self):
        """REQ-022: Middle parts (index 1) of 4-part split get continues cue."""
        result = vary_prompt_for_part("Running.", 1, 4)
        assert result == "Running. The action continues."

    @pytest.mark.req("SPEC-300/REQ-022")
    def test_second_middle_of_four(self):
        """REQ-022: Middle parts (index 2) of 4-part split get continues cue."""
        result = vary_prompt_for_part("Running.", 2, 4)
        assert result == "Running. The action continues."

    @pytest.mark.req("SPEC-300/REQ-023")
    def test_cue_appended_with_space(self):
        """REQ-023: Cue is separated from base prompt by a single space."""
        result = vary_prompt_for_part("base", 0, 2)
        assert result.startswith("base ")


# ---------------------------------------------------------------------------
# Phase 2: Portrait generation
# ---------------------------------------------------------------------------

from decode import generate_portraits, build_character_shot_map  # noqa: E402


def _make_characters_json(tmp_path, characters):
    """Helper: write characters.json and return its path."""
    data = {"characters": characters}
    path = tmp_path / "characters.json"
    path.write_text(json.dumps(data))
    return str(path)


SAMPLE_CHARACTERS = [
    {"name": "luke", "display_name": "Luke Skywalker",
     "description": "Young man, sandy blond hair, blue eyes.", "shots": [0, 2, 4]},
    {"name": "han_solo", "display_name": "Han Solo",
     "description": "Roguish man, dark hair, vest.", "shots": [1, 3]},
]


class TestCharacterShotMap:
    """SPEC-210: Character-to-shot mapping (REQ-010 to REQ-013)."""

    @pytest.mark.req("SPEC-210/REQ-010")
    def test_maps_shot_to_characters(self):
        """REQ-010: Shall return dict mapping shot index to character names."""
        result = build_character_shot_map({"characters": SAMPLE_CHARACTERS})
        assert result[0] == ["luke"]
        assert result[1] == ["han_solo"]
        assert result[2] == ["luke"]
        assert result[4] == ["luke"]

    @pytest.mark.req("SPEC-210/REQ-011")
    def test_shared_shot_lists_all_characters(self):
        """REQ-011: A shot with multiple characters shall list all of them."""
        chars = [
            {"name": "luke", "shots": [5]},
            {"name": "leia", "shots": [5]},
        ]
        result = build_character_shot_map({"characters": chars})
        assert "luke" in result[5]
        assert "leia" in result[5]

    @pytest.mark.req("SPEC-210/REQ-012")
    def test_absent_shot_not_in_map(self):
        """REQ-012: A shot with no characters shall not appear in the mapping."""
        result = build_character_shot_map({"characters": SAMPLE_CHARACTERS})
        assert 99 not in result

    @pytest.mark.req("SPEC-210/REQ-012")
    def test_empty_characters(self):
        """REQ-012: Empty characters produces empty map."""
        assert build_character_shot_map({"characters": []}) == {}

    @pytest.mark.req("SPEC-210/REQ-013")
    def test_character_order_matches_registry(self):
        """REQ-013: Character order in shot list shall match characters.json order."""
        chars = [
            {"name": "luke", "shots": [5]},
            {"name": "leia", "shots": [5]},
        ]
        result = build_character_shot_map({"characters": chars})
        assert result[5] == ["luke", "leia"]

    def test_missing_characters_key(self):
        assert build_character_shot_map({}) == {}

    def test_character_with_no_shots(self):
        assert build_character_shot_map({"characters": [{"name": "extra", "shots": []}]}) == {}


class TestGeneratePortraits:
    """SPEC-210: Portrait generation (REQ-001 to REQ-005)."""

    @pytest.mark.req("SPEC-210/REQ-001", "SPEC-210/REQ-002")
    def test_generates_one_portrait_per_character(self, tmp_path):
        """REQ-001: One portrait per character. REQ-002: Stored as {name}.png."""
        characters_path = _make_characters_json(tmp_path, SAMPLE_CHARACTERS)
        fake_image = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

        mock_resp = MagicMock(content=fake_image, raise_for_status=MagicMock())

        with patch("fal_client.subscribe", return_value={"images": [{"url": "https://example.com/p.png"}]}):
            with patch("httpx.get", return_value=mock_resp):
                result = generate_portraits(characters_path, str(tmp_path))

        assert "luke" in result
        assert "han_solo" in result
        assert (tmp_path / "characters" / "luke.png").exists()
        assert (tmp_path / "characters" / "han_solo.png").exists()

    @pytest.mark.req("SPEC-210/REQ-003")
    def test_skips_existing_portrait(self, tmp_path):
        """REQ-003: Existing portrait shall be reused without regeneration."""
        characters_path = _make_characters_json(tmp_path, [SAMPLE_CHARACTERS[0]])
        (tmp_path / "characters").mkdir()
        (tmp_path / "characters" / "luke.png").write_bytes(b"existing")

        with patch("fal_client.subscribe") as mock_sub:
            with patch("httpx.get") as mock_get:
                result = generate_portraits(characters_path, str(tmp_path))

        mock_sub.assert_not_called()
        mock_get.assert_not_called()
        assert "luke" in result

    @pytest.mark.req("SPEC-210/REQ-004")
    def test_returns_name_to_path_dict(self, tmp_path):
        """REQ-004: Return value shall be {character_name: portrait_path}."""
        characters_path = _make_characters_json(tmp_path, [SAMPLE_CHARACTERS[0]])
        mock_resp = MagicMock(content=b"PNG", raise_for_status=MagicMock())

        with patch("fal_client.subscribe", return_value={"images": [{"url": "https://example.com/p.png"}]}):
            with patch("httpx.get", return_value=mock_resp):
                result = generate_portraits(characters_path, str(tmp_path))

        assert isinstance(result, dict)
        assert result["luke"].endswith("luke.png")

    @pytest.mark.req("SPEC-210/REQ-005")
    def test_failed_portrait_does_not_abort(self, tmp_path):
        """REQ-005: Failed generation shall skip, not abort."""
        characters_path = _make_characters_json(tmp_path, SAMPLE_CHARACTERS)

        call_count = 0

        def fake_subscribe(model, arguments, with_logs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("API error")
            return {"images": [{"url": "https://example.com/p.png"}]}

        mock_resp = MagicMock(content=b"PNG", raise_for_status=MagicMock())

        with patch("fal_client.subscribe", side_effect=fake_subscribe):
            with patch("httpx.get", return_value=mock_resp):
                result = generate_portraits(characters_path, str(tmp_path))

        # First failed, second succeeded
        assert len(result) == 1

    def test_empty_characters_returns_empty(self, tmp_path):
        characters_path = _make_characters_json(tmp_path, [])
        assert generate_portraits(characters_path, str(tmp_path)) == {}

    def test_prompt_includes_description(self, tmp_path):
        characters_path = _make_characters_json(tmp_path, [SAMPLE_CHARACTERS[0]])
        captured = {}

        def fake_subscribe(model, arguments, with_logs):
            captured["prompt"] = arguments["prompt"]
            return {"images": [{"url": "https://example.com/p.png"}]}

        mock_resp = MagicMock(content=b"PNG", raise_for_status=MagicMock())
        with patch("fal_client.subscribe", side_effect=fake_subscribe):
            with patch("httpx.get", return_value=mock_resp):
                generate_portraits(characters_path, str(tmp_path))

        assert "sandy blond hair" in captured["prompt"]


# ---------------------------------------------------------------------------
# Phase 3: VACE Strategy
# ---------------------------------------------------------------------------

import argparse  # noqa: E402

from decode import RunPodVaceStrategy, _create_vace_strategy  # noqa: E402


class TestVaceWorkflow:
    """SPEC-220: VACE workflow selection and structure (REQ-001 to REQ-005)."""

    @pytest.mark.req("SPEC-220/REQ-003")
    def test_uses_vace_diffusion_model(self):
        """REQ-003: VACE workflow shall use wan2.1_vace_1.3B_fp16."""
        strategy = RunPodVaceStrategy()
        workflow = strategy._build_vace_workflow("test", seed=1, length=81, reference_image="ref.png")
        unet_node = next(v for v in workflow.values() if v["class_type"] == "UNETLoader")
        assert "vace" in unet_node["inputs"]["unet_name"]

    @pytest.mark.req("SPEC-220/REQ-004")
    def test_has_vace_and_trim_nodes(self):
        """REQ-004: Shall include WanVaceToVideo and TrimVideoLatent nodes."""
        strategy = RunPodVaceStrategy()
        workflow = strategy._build_vace_workflow("test", seed=42, length=81, reference_image="luke.png")
        node_types = {v["class_type"] for v in workflow.values()}
        assert "WanVaceToVideo" in node_types
        assert "TrimVideoLatent" in node_types

    @pytest.mark.req("SPEC-220/REQ-005")
    def test_strength_is_one(self):
        """REQ-005: WanVaceToVideo strength shall be 1.0."""
        strategy = RunPodVaceStrategy()
        workflow = strategy._build_vace_workflow("test", seed=1, length=81, reference_image="ref.png")
        vace_node = next(v for v in workflow.values() if v["class_type"] == "WanVaceToVideo")
        assert vace_node["inputs"]["strength"] == 1.0

    def test_trim_latent_uses_vace_output(self):
        """TrimVideoLatent takes trim_amount from WanVaceToVideo output [3]."""
        strategy = RunPodVaceStrategy()
        workflow = strategy._build_vace_workflow("test", seed=1, length=81, reference_image="ref.png")
        vace_node_id = next(k for k, v in workflow.items() if v["class_type"] == "WanVaceToVideo")
        trim_node = next(v for v in workflow.values() if v["class_type"] == "TrimVideoLatent")
        assert trim_node["inputs"]["trim_amount"] == [vace_node_id, 3]

    def test_prompt_injected(self):
        strategy = RunPodVaceStrategy()
        workflow = strategy._build_vace_workflow("rebel fighter", seed=5, length=33, reference_image="leia.png")
        clip_nodes = [v for v in workflow.values()
                      if v["class_type"] == "CLIPTextEncode" and "rebel fighter" in v["inputs"].get("text", "")]
        assert len(clip_nodes) == 1

    def test_has_chinese_negative_prompt(self):
        strategy = RunPodVaceStrategy()
        workflow = strategy._build_vace_workflow("test", seed=1, length=81, reference_image="ref.png")
        neg_nodes = [v for v in workflow.values()
                     if v["class_type"] == "CLIPTextEncode"
                     and any('\u4e00' <= c <= '\u9fff' for c in v["inputs"].get("text", ""))]
        assert len(neg_nodes) == 1


class TestVaceReferenceHandling:
    """SPEC-220: Reference image handling (REQ-010 to REQ-012)."""

    @pytest.mark.req("SPEC-220/REQ-010")
    def test_primary_character_used_as_reference(self):
        """REQ-010: Primary character (first in list) shall be the reference image."""
        strategy = RunPodVaceStrategy(
            portraits={"luke": "/tmp/luke.png", "leia": "/tmp/leia.png"},
            character_shot_map={5: ["luke", "leia"]},
        )
        strategy._uploaded_portraits = {"luke": "luke.png", "leia": "leia.png"}

        # The generate method looks up char_names[0] for the reference
        char_names = strategy._character_shot_map.get(5, [])
        primary = char_names[0]
        ref = strategy._uploaded_portraits.get(primary)
        assert ref == "luke.png"

    @pytest.mark.req("SPEC-220/REQ-011")
    def test_load_image_references_portrait_filename(self):
        """REQ-011: LoadImage node shall reference the portrait filename."""
        strategy = RunPodVaceStrategy()
        workflow = strategy._build_vace_workflow("test", seed=1, length=49, reference_image="vader.png")
        load_node = next(v for v in workflow.values() if v["class_type"] == "LoadImage")
        assert load_node["inputs"]["image"] == "vader.png"


class TestVacePromptEnrichment:
    """SPEC-220: Prompt enrichment with character identity (REQ-020 to REQ-021)."""

    @pytest.mark.req("SPEC-220/REQ-020")
    def test_prompt_excludes_character_name(self):
        """REQ-020 (reversed 2026-08-05): the prompt shall NOT include the name.

        Naming a character lets the model reconstruct it from world knowledge
        instead of from the compressed description -- research 0021. The
        description still carries the identity, per REQ-021.
        """
        characters_data = {
            "characters": [
                {"name": "luke", "display_name": "Luke Skywalker",
                 "description": "Young man, sandy blond hair, blue eyes.", "shots": [5]},
            ]
        }
        strategy = RunPodVaceStrategy(
            character_shot_map={5: ["luke"]},
            portraits={"luke": "/tmp/luke.png"},
            characters_data=characters_data,
        )
        strategy._uploaded_portraits = {"luke": "luke.png"}

        entry = {
            "index": 5,
            "description": {
                "subjects": "A young man in desert robes",
                "action": "walks across sand",
                "shot_type": "medium",
                "camera_movement": "static",
            },
            "duration_s": 3.0,
        }

        prompt = strategy.format_prompt(entry)
        assert "Luke Skywalker" not in prompt
        assert "Luke" not in prompt
        assert "sandy blond hair" in prompt  # the description still arrives

    @pytest.mark.req("SPEC-220/REQ-021")
    def test_prompt_includes_character_description(self):
        """REQ-021: Prompt shall include the character's canonical description."""
        characters_data = {
            "characters": [
                {"name": "luke", "display_name": "Luke Skywalker",
                 "description": "Young man, early 20s, sandy blond hair, blue eyes, white tunic.",
                 "shots": [5]},
            ]
        }
        strategy = RunPodVaceStrategy(
            character_shot_map={5: ["luke"]},
            portraits={"luke": "/tmp/luke.png"},
            characters_data=characters_data,
        )
        strategy._uploaded_portraits = {"luke": "luke.png"}

        entry = {
            "index": 5,
            "description": {
                "subjects": "A young man in desert robes",
                "action": "walks across sand",
                "shot_type": "medium",
                "camera_movement": "static",
            },
            "duration_s": 3.0,
        }

        prompt = strategy.format_prompt(entry)
        assert "sandy blond hair" in prompt


class TestVaceStrategyIntegration:
    """SPEC-220: Strategy integration (REQ-030 to REQ-032)."""

    @pytest.mark.req("SPEC-220/REQ-030")
    def test_runpod_vace_is_valid_strategy(self):
        """REQ-030: runpod-vace shall be a valid strategy choice."""
        assert RunPodVaceStrategy.name == "runpod-vace"
        assert issubclass(RunPodVaceStrategy, RunPodWanStrategy)

    @pytest.mark.req("SPEC-220/REQ-030")
    def test_create_vace_strategy_without_characters(self, tmp_path):
        """REQ-030: Falls back gracefully when characters.json absent."""
        args = argparse.Namespace(output_dir=str(tmp_path), keep_pod=False, concurrent_audio=False)
        strategy = _create_vace_strategy(args)
        assert isinstance(strategy, RunPodVaceStrategy)
        assert strategy._portraits == {}

    @pytest.mark.req("SPEC-220/REQ-030")
    def test_create_vace_strategy_with_characters(self, tmp_path):
        """REQ-030: Loads characters.json and builds shot map."""
        chars = {"characters": [
            {"name": "luke", "display_name": "Luke Skywalker",
             "description": "Young man, sandy blond hair.", "shots": [0, 2]},
        ]}
        (tmp_path / "characters.json").write_text(json.dumps(chars))
        (tmp_path / "characters").mkdir()
        (tmp_path / "characters" / "luke.png").write_bytes(b"PNG")

        args = argparse.Namespace(output_dir=str(tmp_path), keep_pod=False, concurrent_audio=False)
        strategy = _create_vace_strategy(args)
        assert "luke" in strategy._portraits
        assert strategy._character_shot_map[0] == ["luke"]

    @pytest.mark.req("SPEC-220/REQ-031")
    def test_pipeline_includes_encode3_for_vace(self):
        """REQ-031: pipeline shall include encode3 when strategy is runpod-vace."""
        from pipeline import build_commands, STAGES
        assert "encode3" in STAGES

        args = argparse.Namespace(
            video="media/test.mp4", output="output/test", strategy="runpod-vace",
            detector=None, threshold=None, limit=None, start_index=None,
            audio_strategy=None, speech_voice=None, skip=[], dry_run=False,
        )
        commands = build_commands(args)
        assert "encode3" in commands

    @pytest.mark.req("SPEC-220/REQ-032")
    def test_pipeline_skips_encode3_for_non_vace(self):
        """REQ-032: pipeline shall skip encode3 for non-VACE strategies."""
        from pipeline import build_commands
        args = argparse.Namespace(
            video="media/test.mp4", output="output/test", strategy="runpod-wan",
            detector=None, threshold=None, limit=None, start_index=None,
            audio_strategy=None, speech_voice=None, skip=[], dry_run=True,
        )
        commands = build_commands(args)
        assert "encode3" in commands  # command exists but will be skipped at runtime

    def test_init_defaults_to_empty(self):
        strategy = RunPodVaceStrategy()
        assert strategy._portraits == {}
        assert strategy._character_shot_map == {}
        assert strategy._uploaded_portraits == {}

    def test_vace_models_include_required_weights(self):
        assert any("vace_1.3B" in url for _, url in RunPodVaceStrategy.VACE_MODELS)
        assert any("vae" in path for path, _ in RunPodVaceStrategy.VACE_MODELS)
        assert any("text_encoders" in path for path, _ in RunPodVaceStrategy.VACE_MODELS)

# ---------------------------------------------------------------------------
# run_speech — v2 global dialog path
# ---------------------------------------------------------------------------

import argparse  # noqa: E402,F811 (already imported above, harmless re-import)
from decode import run_speech  # noqa: E402


class TestRunSpeechV2:
    """Tests for the v2 (global dialog) path in run_speech()."""

    def _make_args(self, output_dir):
        return argparse.Namespace(
            output_dir=output_dir,
            start_index=None,
            limit=None,
        )

    def _make_prompts_v2(self, dialog):
        return {
            "format": "v2",
            "shots": [{"index": 0, "start_s": 0.0, "end_s": 5.0, "duration_s": 5.0}],
            "dialog": dialog,
        }

    def test_v2_generates_one_clip_per_line(self, tmp_path):
        """Each dialog line produces exactly one TTS clip."""
        dialog = [
            {"text": "Hello world", "start_s": 1.0, "end_s": 2.5},
            {"text": "Goodbye", "start_s": 3.0, "end_s": 4.0},
        ]
        (tmp_path / "shots.json").write_text(
            json.dumps(self._make_prompts_v2(dialog))
        )

        mock_strategy = MagicMock()
        mock_strategy.generate.side_effect = [
            MagicMock(path=str(tmp_path / "speech" / "0000-00.mp3"),
                      duration_s=1.5, offset_s=1.0, cost=0.001),
            MagicMock(path=str(tmp_path / "speech" / "0001-00.mp3"),
                      duration_s=1.0, offset_s=3.0, cost=0.001),
        ]

        (tmp_path / "speech").mkdir()
        run_speech(self._make_args(str(tmp_path)), mock_strategy)

        assert mock_strategy.generate.call_count == 2
        # Verify calls used dialog index as shot_index and 0 as line_index
        call_args = mock_strategy.generate.call_args_list
        assert call_args[0][0][2] == 0  # shot_index=0 (dialog index 0)
        assert call_args[0][0][3] == 0  # line_index=0
        assert call_args[1][0][2] == 1  # shot_index=1 (dialog index 1)

    def test_v2_writes_v2_progress_format(self, tmp_path):
        """speech_progress.json is written with format=v2."""
        dialog = [{"text": "Hi", "start_s": 0.5, "end_s": 1.5}]
        (tmp_path / "shots.json").write_text(
            json.dumps(self._make_prompts_v2(dialog))
        )
        (tmp_path / "speech").mkdir()

        mock_strategy = MagicMock()
        mock_strategy.generate.return_value = MagicMock(
            path=str(tmp_path / "speech" / "0000-00.mp3"),
            duration_s=1.0, offset_s=0.5, cost=0.001,
        )

        run_speech(self._make_args(str(tmp_path)), mock_strategy)

        progress = json.loads((tmp_path / "speech_progress.json").read_text())
        assert progress["format"] == "v2"
        assert 0 in progress["completed"]
        assert "start_s" in progress["clips"]["0"]
        assert "end_s" in progress["clips"]["0"]
        assert progress["clips"]["0"]["start_s"] == 0.5

    def test_v2_skips_already_completed(self, tmp_path):
        """Already-completed dialog lines are not regenerated."""
        dialog = [
            {"text": "Line 1", "start_s": 1.0, "end_s": 2.0},
            {"text": "Line 2", "start_s": 3.0, "end_s": 4.0},
        ]
        (tmp_path / "shots.json").write_text(
            json.dumps(self._make_prompts_v2(dialog))
        )
        # Pre-existing progress marks line 0 as done
        (tmp_path / "speech_progress.json").write_text(json.dumps({
            "format": "v2",
            "completed": [0],
            "failed": [],
            "total_cost_estimate": 0.001,
            "clips": {
                "0": {"path": "0000-00.mp3", "start_s": 1.0, "end_s": 2.0, "duration_s": 1.0},
            },
        }))
        (tmp_path / "speech").mkdir()

        mock_strategy = MagicMock()
        mock_strategy.generate.return_value = MagicMock(
            path=str(tmp_path / "speech" / "0001-00.mp3"),
            duration_s=1.0, offset_s=3.0, cost=0.001,
        )

        run_speech(self._make_args(str(tmp_path)), mock_strategy)

        # Only line 1 should be generated (line 0 was already done)
        assert mock_strategy.generate.call_count == 1



class TestRunSpeechVoiceMapping:
    """Per-character voice selection via speakers.json / voice_map.json."""

    def _make_args(self, output_dir, start_index=None, limit=None):
        return argparse.Namespace(
            output_dir=output_dir,
            start_index=start_index,
            limit=limit,
        )

    def _make_prompts_v2(self, shots, dialog):
        return {"format": "v2", "shots": shots, "dialog": dialog}

    def test_attributed_line_uses_mapped_voice(self, tmp_path, monkeypatch):
        dialog = [
            {"text": "Line by luke", "start_s": 1.0, "end_s": 2.0},
            {"text": "Unattributed line", "start_s": 3.0, "end_s": 4.0},
        ]
        shots = [{"index": 0, "start_s": 0.0, "end_s": 5.0, "duration_s": 5.0}]
        (tmp_path / "shots.json").write_text(json.dumps(self._make_prompts_v2(shots, dialog)))
        (tmp_path / "speakers.json").write_text(json.dumps({
            "format": "v1",
            "assignments": {"0": {"character": "luke", "confidence": 0.9}},
        }))
        (tmp_path / "voice_map.json").write_text(json.dumps({"luke": "Charlie"}))
        (tmp_path / "speech").mkdir()

        calls = []

        def fake_generate(self, text, speech_dir, shot_index, line_index, offset_s):
            calls.append((self.voice, text))
            path = os.path.join(speech_dir, f"{shot_index:04d}-{line_index:02d}.mp3")
            return SpeechClipResult(path=path, duration_s=1.0, offset_s=offset_s, cost=0.001)

        monkeypatch.setattr(SpeechStrategy, "generate", fake_generate)

        default_strategy = SpeechStrategy(voice="Roger")
        run_speech(self._make_args(str(tmp_path)), default_strategy)

        assert ("Charlie", "Line by luke") in calls
        assert ("Roger", "Unattributed line") in calls

    def test_no_speakers_file_falls_back_to_default_voice(self, tmp_path, monkeypatch):
        """Backward compatibility: no speakers.json/voice_map.json -> old behavior."""
        dialog = [{"text": "Hi", "start_s": 0.5, "end_s": 1.5}]
        shots = [{"index": 0, "start_s": 0.0, "end_s": 5.0, "duration_s": 5.0}]
        (tmp_path / "shots.json").write_text(json.dumps(self._make_prompts_v2(shots, dialog)))
        (tmp_path / "speech").mkdir()

        calls = []

        def fake_generate(self, text, speech_dir, shot_index, line_index, offset_s):
            calls.append(self.voice)
            path = os.path.join(speech_dir, f"{shot_index:04d}-{line_index:02d}.mp3")
            return SpeechClipResult(path=path, duration_s=1.0, offset_s=offset_s, cost=0.001)

        monkeypatch.setattr(SpeechStrategy, "generate", fake_generate)

        default_strategy = SpeechStrategy(voice="Roger")
        run_speech(self._make_args(str(tmp_path)), default_strategy)

        assert calls == ["Roger"]


class TestRunSpeechWindowing:
    """--start-index/--limit restrict which dialog lines get generated."""

    def test_limit_restricts_to_shot_window(self, tmp_path):
        shots = [
            {"index": 0, "start_s": 0.0, "end_s": 5.0, "duration_s": 5.0},
            {"index": 1, "start_s": 5.0, "end_s": 10.0, "duration_s": 5.0},
            {"index": 2, "start_s": 10.0, "end_s": 15.0, "duration_s": 5.0},
        ]
        dialog = [
            {"text": "In shot 0", "start_s": 1.0, "end_s": 2.0},
            {"text": "In shot 2", "start_s": 11.0, "end_s": 12.0},
        ]
        (tmp_path / "shots.json").write_text(json.dumps({
            "format": "v2", "shots": shots, "dialog": dialog,
        }))
        (tmp_path / "speech").mkdir()

        strategy = MagicMock()
        strategy.voice = "Roger"
        strategy.generate.return_value = MagicMock(
            path=str(tmp_path / "speech" / "x.mp3"), duration_s=1.0, offset_s=1.0, cost=0.001,
        )

        args = argparse.Namespace(output_dir=str(tmp_path), start_index=0, limit=1)
        run_speech(args, strategy)

        # Only the line inside shot 0's window should be generated.
        assert strategy.generate.call_count == 1
        assert strategy.generate.call_args[0][0] == "In shot 0"


class TestLtx2CharacterIdentity:
    """LTX-2 must inject canonical character identity, like runpod-wan22 does.

    The strategy originally shipped without the mixin, so every prompt reached
    the model with no idea who anyone was -- a rehearsal pass rendered Han and
    Luke as generic modern men. It also made the LTX-2 vs Wan22 quality trial
    unfair, since Wan22 had enrichment and LTX-2 did not.
    """

    ENTRY = {
        "index": 796,
        "duration_s": 4.0,
        "description": {"shot_type": "medium", "camera_movement": "static",
                        "subjects": ["a man"], "action": "He walks."},
    }
    CHARACTERS = {
        "characters": [
            {"name": "han_solo", "display_name": "Han Solo",
             "description": "A rugged smuggler in a tan tunic."},
        ]
    }

    def _strategy(self, **kwargs):
        from strategies_video import RunPodLtx2Strategy
        return RunPodLtx2Strategy(**kwargs)

    def test_prepends_identity_for_characters_in_shot(self):
        s = self._strategy(character_shot_map={796: ["han_solo"]},
                           characters_data=self.CHARACTERS)
        prompt = s.format_prompt(self.ENTRY)
        # The description leads; the name is stripped before the model sees it.
        assert prompt.startswith("A rugged smuggler in a tan tunic.")
        assert "Han Solo" not in prompt

    def test_no_identity_when_shot_has_no_characters(self):
        from prompt_format import format_prompt
        s = self._strategy(character_shot_map={}, characters_data=self.CHARACTERS)
        assert s.format_prompt(self.ENTRY) == format_prompt(self.ENTRY)

    def test_base_prompt_is_plain_not_wan_formatted(self):
        # LTX-2 takes natural language; it must not inherit RunPodWanStrategy's
        # Wan-specific formatting through the MRO now that a mixin sits above it.
        from prompt_format import format_prompt
        s = self._strategy(character_shot_map={796: ["han_solo"]},
                           characters_data=self.CHARACTERS)
        prompt = s.format_prompt(self.ENTRY)
        assert prompt.endswith(format_prompt(self.ENTRY))


class TestLtx2Chaining:
    """LTX-2 must chain split parts via I2V, or long shots jump-cut internally.

    The spike shipped _supports_i2v = False, so a 17.7s shot rendered as four
    disconnected takes. In the first dress rehearsal 52% of the segment's
    runtime sat in split shots, adding 26 cuts the film does not have.
    """

    def _wf(self, start_image=None, length=121):
        from strategies_video import RunPodLtx2Strategy
        s = RunPodLtx2Strategy.__new__(RunPodLtx2Strategy)
        return s._build_workflow("prompt", seed=1, length=length, start_image=start_image)

    def test_strategy_declares_i2v_support(self):
        from strategies_video import RunPodLtx2Strategy
        assert RunPodLtx2Strategy._supports_i2v is True

    def test_t2v_uses_empty_latent(self):
        wf = self._wf()
        assert wf["9"]["inputs"]["video_latent"] == ["5", 0]
        assert wf["6"]["inputs"]["positive"] == ["3", 0]
        assert "14" not in wf and "15" not in wf

    def test_i2v_routes_latent_and_conditioning_through_img_to_video(self):
        wf = self._wf(start_image="chain_0846_00.png")
        assert wf["15"]["class_type"] == "LTXVImgToVideo"
        assert wf["14"]["inputs"]["image"] == "chain_0846_00.png"
        # conditioning must flow through the image node, not straight from the
        # text encoders, or the start frame is ignored
        assert wf["6"]["inputs"]["positive"] == ["15", 0]
        assert wf["6"]["inputs"]["negative"] == ["15", 1]
        # the AV concat must take the conditioned latent
        assert wf["9"]["inputs"]["video_latent"] == ["15", 2]

    def test_i2v_supersedes_the_empty_latent(self):
        wf = self._wf(start_image="chain_0846_00.png")
        refs = [(n, k) for n, node in wf.items()
                for k, v in node["inputs"].items()
                if isinstance(v, list) and len(v) == 2 and v[0] == "5"]
        assert refs == [], f"empty latent still feeds {refs}"

    def test_i2v_length_and_resolution_match_the_video_latent(self):
        wf = self._wf(start_image="x.png", length=97)
        assert wf["15"]["inputs"]["length"] == 97
        assert wf["15"]["inputs"]["width"] == wf["5"]["inputs"]["width"]
        assert wf["15"]["inputs"]["height"] == wf["5"]["inputs"]["height"]

    def test_all_node_references_resolve(self):
        for wf in (self._wf(), self._wf(start_image="x.png")):
            for nid, node in wf.items():
                for k, v in node["inputs"].items():
                    if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                        assert v[0] in wf, f"node {nid}.{k} -> missing {v[0]}"


class TestSeamTrim:
    """Split parts are trimmed at internal seams to cut the motion hitch.

    The model renders each part with a slow-in/slow-out envelope and sometimes
    a frozen final frame, so a chained long shot decelerates and re-accelerates
    at every boundary -- a rhythmic hitch through elongated scenes. Chaining
    fixes the content jump; this fixes the motion one.
    """

    def _ltx(self):
        from strategies_video import RunPodLtx2Strategy
        return RunPodLtx2Strategy.__new__(RunPodLtx2Strategy)

    def test_trimming_is_off_by_default_for_other_strategies(self):
        from strategies_video import RunPodWanStrategy
        w = RunPodWanStrategy.__new__(RunPodWanStrategy)
        assert w.SEAM_TRIM_HEAD == 0 and w.SEAM_TRIM_TAIL == 0
        assert w._usable_frames_per_part() == w.MAX_FRAMES

    def test_ltx_usable_frames_account_for_the_trim(self):
        s = self._ltx()
        assert s._usable_frames_per_part() == s.MAX_FRAMES - s.SEAM_TRIM_HEAD - s.SEAM_TRIM_TAIL

    def test_part_planning_accounts_for_whatever_trim_is_configured(self):
        """With trimming off, parts cover the full clip; with it on, more parts."""
        s = self._ltx()
        baseline = len(s._target_durations(17.7))
        s.SEAM_TRIM_HEAD, s.SEAM_TRIM_TAIL = 0, 12
        assert len(s._target_durations(17.7)) > baseline

    def test_short_single_part_shot_is_unaffected(self):
        s = self._ltx()
        assert len(s._target_durations(3.0)) == 1

    def test_single_part_shot_is_never_trimmed(self, tmp_path):
        from clip_types import ClipResult
        s = self._ltx()
        c = ClipResult(path=str(tmp_path / "x.mp4"), actual_duration_s=4.0, cost=0.0)
        s._trim_seam_frames(c, 0, 1)  # no file needed -- must return before touching it
        assert c.actual_duration_s == 4.0

    def test_shot_edges_keep_their_natural_ease(self, monkeypatch, tmp_path):
        """The first part trims its tail; the shot's final tail is left alone."""
        from clip_types import ClipResult
        s = self._ltx()
        seen = []

        def fake_run(cmd, **kw):
            seen.append(" ".join(cmd))
            class R:
                returncode = 1  # bail out after recording the filter
            return R()
        monkeypatch.setattr("strategies_video.subprocess.run", fake_run)

        s.SEAM_TRIM_TAIL = 12  # mechanism is currently disabled; exercise it directly
        c = ClipResult(path=str(tmp_path / "a.mp4"), actual_duration_s=4.84, cost=0.0)
        s._trim_seam_frames(c, 0, 3)          # first part: tail trimmed
        assert len(seen) == 1
        assert "between(n\\,0\\," in seen[-1], "head is never trimmed while chaining"
        assert str(round(4.84 * s.FPS) - s.SEAM_TRIM_TAIL - 1) in seen[-1]

        # last part: nothing to trim (head is 0, and its tail is the shot's own end)
        before = len(seen)
        s._trim_seam_frames(c, 2, 3)
        assert len(seen) == before, "the shot's final tail must keep its natural ease"

    def test_clip_too_short_to_trim_is_left_alone(self, tmp_path):
        from clip_types import ClipResult
        s = self._ltx()
        c = ClipResult(path=str(tmp_path / "x.mp4"), actual_duration_s=0.5, cost=0.0)
        s._trim_seam_frames(c, 1, 3)
        assert c.actual_duration_s == 0.5


class TestChainUploadIsNonFatal:
    """A failed chain-frame upload must degrade to an unchained part.

    A stalled scp raised TimeoutExpired straight through the generate loop and
    aborted a render 84 clips in. Chaining is an enhancement; losing it costs
    one visible seam, losing the run costs hours.
    """

    def _strategy(self, tmp_path, raiser):
        from strategies_video import RunPodLtx2Strategy
        s = RunPodLtx2Strategy.__new__(RunPodLtx2Strategy)

        class Session:
            ssh_host = "1.2.3.4"
            def scp_to(self, *a, **kw):
                raise raiser
        s._session = Session()
        s._extract_last_frame = lambda clip, out: True
        return s

    def test_timeout_returns_none_instead_of_raising(self, tmp_path):
        import subprocess as sp
        s = self._strategy(tmp_path, sp.TimeoutExpired(cmd="scp", timeout=60))
        assert s._upload_start_frame("clip.mp4", 846, 3) is None

    def test_unexpected_error_returns_none_instead_of_raising(self, tmp_path):
        s = self._strategy(tmp_path, OSError("connection reset"))
        assert s._upload_start_frame("clip.mp4", 846, 3) is None


class TestHeadTrimMustNotBreakChaining:
    """Trimming a chained part's head throws away the frames that make it continuous.

    Continuity lives in the first frames -- they are the ones conditioned on
    the previous part's last frame. A rehearsal render with an 18-frame head
    trim measured seam jumps of 25-43 against 2.6 untrimmed.
    """

    def test_ltx_does_not_trim_the_head_while_chaining(self):
        from strategies_video import RunPodLtx2Strategy
        s = RunPodLtx2Strategy
        if s._supports_i2v:
            assert s.SEAM_TRIM_HEAD == 0, (
                "head trimming discards the conditioned frames and reopens the seam"
            )

    def test_trimming_is_disabled_after_measuring_no_benefit(self):
        # Surveyed across every multi-part shot: chaining takes the seam from
        # 44.6 to 10.4, and a 12-frame tail trim moved it to 11.5 -- no gain,
        # at the cost of frames and extra parts. The mechanism stays for a
        # future attempt; the constants are zero.
        from strategies_video import RunPodLtx2Strategy
        assert RunPodLtx2Strategy.SEAM_TRIM_TAIL == 0


class TestCrossfadeJoins:
    """Chained parts are dissolved together so the join stops reading as a hitch.

    Chaining takes a join from ~8.5x a normal frame step to ~2.2x, but the
    residual is still visible and there are ~617 joins across the film -- one
    every ~12s. The two sides of a chained join are already visually close,
    which is the condition a short dissolve hides well.
    """

    def test_crossfade_is_off_by_default(self):
        # It masked a join hitch that turned out to be the 4-frame rewind bug.
        # With that fixed the dissolve only costs a doubled-edge artifact.
        import stitch
        assert stitch.CROSSFADE_S == 0

    def test_parts_shorter_than_the_dissolve_are_not_merged(self, tmp_path, monkeypatch):
        import stitch
        monkeypatch.setattr(stitch, "_probe_duration", lambda p: 0.1)
        out = str(tmp_path / "m.mp4")
        assert stitch._crossfade_parts(["a.mp4", "b.mp4"], out, 0.24) is None

    def test_filter_chain_normalises_frame_rate_before_dissolving(self, tmp_path, monkeypatch):
        """Retiming leaves parts at differing rates and xfade rejects those."""
        import stitch
        seen = {}
        monkeypatch.setattr(stitch, "_probe_duration", lambda p: 4.0)
        monkeypatch.setattr(stitch, "_run_ffmpeg",
                            lambda cmd, label: seen.setdefault("cmd", cmd))
        monkeypatch.setattr(stitch.os.path, "exists", lambda p: True)
        monkeypatch.setattr(stitch.os.path, "getsize", lambda p: 100)
        stitch._crossfade_parts(["a.mp4", "b.mp4", "c.mp4"], str(tmp_path / "m.mp4"), 0.24)
        fc = seen["cmd"][seen["cmd"].index("-filter_complex") + 1]
        assert fc.count(f"fps={stitch.XFADE_FPS}") == 3, "every part must be normalised"
        assert fc.count("xfade") == 2, "two joins for three parts"

    def test_dissolve_offsets_account_for_earlier_fades(self, tmp_path, monkeypatch):
        import stitch
        seen = {}
        monkeypatch.setattr(stitch, "_probe_duration", lambda p: 4.0)
        monkeypatch.setattr(stitch, "_run_ffmpeg",
                            lambda cmd, label: seen.setdefault("cmd", cmd))
        monkeypatch.setattr(stitch.os.path, "exists", lambda p: True)
        monkeypatch.setattr(stitch.os.path, "getsize", lambda p: 100)
        stitch._crossfade_parts(["a.mp4", "b.mp4", "c.mp4"], str(tmp_path / "m.mp4"), 0.5)
        fc = seen["cmd"][seen["cmd"].index("-filter_complex") + 1]
        # first join at 4.0 - 0.5; second at (4.0 + 4.0 - 0.5) - 0.5
        assert "offset=3.5" in fc
        assert "offset=7.0" in fc


class TestChainConditioningFrame:
    """The frame handed to the next part must be the clip's LAST frame.

    `-sseof -N -frames:v 1` takes the FIRST frame of the trailing window, not
    the last -- with -0.2 that is ~5 frames early at 25fps. Every chained part
    therefore restarted 4 frames back and replayed them, undoing motion at each
    join: measured across four joins, the next part's first frame best-matched
    the frame 4 before the end (diff ~2.5) against 8-19 for the true last one.
    """

    def _cmd_for(self, monkeypatch):
        import strategies_video
        from strategies_video import RunPodLtx2Strategy
        seen = {}

        class R:
            returncode = 0
        monkeypatch.setattr(strategies_video.subprocess, "run",
                            lambda cmd, **kw: (seen.setdefault("cmd", cmd), R())[1])
        monkeypatch.setattr(strategies_video.os.path, "exists", lambda p: True)
        monkeypatch.setattr(strategies_video.os.path, "getsize", lambda p: 100)
        s = RunPodLtx2Strategy.__new__(RunPodLtx2Strategy)
        s._extract_last_frame("clip.mp4", "out.png")
        return seen["cmd"]

    def test_does_not_take_the_first_frame_of_the_window(self, monkeypatch):
        cmd = self._cmd_for(monkeypatch)
        assert "-frames:v" not in cmd, (
            "-frames:v 1 grabs the first frame after the seek, which is several "
            "frames before the end -- that is the motion-rewind bug"
        )

    def test_uses_update_so_the_last_frame_wins(self, monkeypatch):
        cmd = self._cmd_for(monkeypatch)
        assert "-update" in cmd, "every frame must overwrite so the final one remains"

    def test_seeks_from_the_end(self, monkeypatch):
        cmd = self._cmd_for(monkeypatch)
        assert "-sseof" in cmd
        assert float(cmd[cmd.index("-sseof") + 1]) < 0


class TestClipRetry:
    """A failed clip is retried rather than failing its whole shot.

    The first submission after a fresh pod boot has been observed to fail even
    though wait_for_comfyui()'s /system_stats probe already returned 200 --
    that probe proves the server is up, not that the model is loaded. Before
    the retry, generate() dropped an entire shot on a pod already paid for.
    """

    def _strategy(self, monkeypatch):
        monkeypatch.setattr("strategies_video.time.sleep", lambda s: None)
        return RunPodWanStrategy()

    def test_a_failed_attempt_is_retried(self, monkeypatch, tmp_path):
        strategy = self._strategy(monkeypatch)
        calls = []

        def fake_attempt(prompt, clips_dir, clip_name, frames, seed, start_image=None):
            calls.append(seed)
            return None if len(calls) == 1 else "clip"

        monkeypatch.setattr(strategy, "_attempt_one_clip", fake_attempt)
        result = strategy._generate_one_clip("p", str(tmp_path), "c.mp4", 81, 7)

        assert result == "clip"
        assert len(calls) == 2

    def test_the_retry_reuses_the_same_seed(self, monkeypatch, tmp_path):
        """A retry is the same clip again, not a different one."""
        strategy = self._strategy(monkeypatch)
        seeds = []

        def fake_attempt(prompt, clips_dir, clip_name, frames, seed, start_image=None):
            seeds.append(seed)
            return None

        monkeypatch.setattr(strategy, "_attempt_one_clip", fake_attempt)
        strategy._generate_one_clip("p", str(tmp_path), "c.mp4", 81, 7)

        assert seeds == [7] * strategy.CLIP_ATTEMPTS

    def test_it_gives_up_after_the_attempt_limit(self, monkeypatch, tmp_path):
        strategy = self._strategy(monkeypatch)
        calls = []

        def fake_attempt(prompt, clips_dir, clip_name, frames, seed, start_image=None):
            calls.append(seed)
            return None

        monkeypatch.setattr(strategy, "_attempt_one_clip", fake_attempt)
        result = strategy._generate_one_clip("p", str(tmp_path), "c.mp4", 81, 7)

        assert result is None
        assert len(calls) == strategy.CLIP_ATTEMPTS

    def test_a_first_time_success_does_not_retry(self, monkeypatch, tmp_path):
        strategy = self._strategy(monkeypatch)
        calls = []

        def fake_attempt(prompt, clips_dir, clip_name, frames, seed, start_image=None):
            calls.append(seed)
            return "clip"

        monkeypatch.setattr(strategy, "_attempt_one_clip", fake_attempt)
        assert strategy._generate_one_clip("p", str(tmp_path), "c.mp4", 81, 7) == "clip"
        assert len(calls) == 1

    def test_the_start_image_is_carried_into_the_retry(self, monkeypatch, tmp_path):
        """Chained parts must stay conditioned on the previous part's frame."""
        strategy = self._strategy(monkeypatch)
        seen = []

        def fake_attempt(prompt, clips_dir, clip_name, frames, seed, start_image=None):
            seen.append(start_image)
            return None

        monkeypatch.setattr(strategy, "_attempt_one_clip", fake_attempt)
        strategy._generate_one_clip("p", str(tmp_path), "c.mp4", 81, 7, start_image="http://f.jpg")

        assert seen == ["http://f.jpg"] * strategy.CLIP_ATTEMPTS


from decode import _record_success  # noqa: E402


class TestRecordSuccessClearsFailures:
    """A shot that succeeds must stop being reported as failed.

    Only the in-run retry path used to clear the failed list, so a *resumed*
    run left shots sitting in `completed` and `failed` at once. The full run's
    first checkpoint showed exactly that for shots 10-14, which makes the
    failed list useless as a QC signal across 2069 shots.
    """

    def _result(self, path="0010.mp4"):
        return ClipResult(path=path, cost=0.01, actual_duration_s=2.0)

    def test_a_shot_failed_on_an_earlier_run_is_cleared_on_success(self):
        progress = {"completed": [], "failed": [10, 11], "total_cost_estimate": 0.0, "clips": {}}

        _record_success(progress, set(), 10, [self._result()])

        assert progress["failed"] == [11]
        assert progress["completed"] == [10]

    def test_other_shots_failures_are_left_alone(self):
        progress = {"completed": [], "failed": [7, 8, 9], "total_cost_estimate": 0.0, "clips": {}}

        _record_success(progress, set(), 42, [self._result("0042.mp4")])

        assert progress["failed"] == [7, 8, 9]

    def test_cost_and_clips_are_still_recorded(self):
        progress = {"completed": [], "failed": [10], "total_cost_estimate": 1.5, "clips": {}}

        _record_success(progress, set(), 10, [self._result(), self._result("0010-02.mp4")])

        assert progress["total_cost_estimate"] == pytest.approx(1.52)
        assert [c["path"] for c in progress["clips"]["10"]] == ["0010.mp4", "0010-02.mp4"]


class TestClipRetryBackoffGrows:
    """A retry delay that grows, because not every failure is the pod's.

    The full run's checkpoint lost shot 56 to `[Errno 49] Can't assign
    requested address` -- local socket exhaustion. Two attempts 20s apart
    could not outlast it; the shot was abandoned and the run moved on.
    """

    def _strategy(self, monkeypatch):
        s = RunPodWanStrategy.__new__(RunPodWanStrategy)
        return s

    def test_the_delay_grows_with_each_attempt(self, monkeypatch, tmp_path):
        s = self._strategy(monkeypatch)
        slept = []
        monkeypatch.setattr("strategies_video.time.sleep", lambda d: slept.append(d))
        monkeypatch.setattr(s, "_attempt_one_clip",
                            lambda *a, **k: None)

        assert s._generate_one_clip("p", str(tmp_path), "c.mp4", 81, 7) is None
        assert slept == [s.CLIP_RETRY_DELAY_S * n for n in range(1, s.CLIP_ATTEMPTS)]
        assert slept == sorted(slept) and len(set(slept)) == len(slept)

    def test_there_are_at_least_three_attempts(self):
        assert RunPodWanStrategy.CLIP_ATTEMPTS >= 3


class TestLetterboxStripping:
    """Baked-in mattes are cropped at stitch time, not re-rendered.

    13 of the full run's first 50 shots (26%) came back with ~23% of frame
    height as black bars, scattered rather than clustered, so the film jumped
    aspect ratio at the cut. The bars are in the pixels, so only a crop
    removes them.
    """

    def _detect(self, monkeypatch, crop_line, size="1280x704"):
        import stitch

        def fake_ffmpeg(cmd, context):
            out = MagicMock()
            out.stdout = size if cmd[0] == "ffprobe" else ""
            out.stderr = crop_line
            return out

        monkeypatch.setattr(stitch, "_run_ffmpeg", fake_ffmpeg)
        return stitch._detect_letterbox("clip.mp4")

    def test_a_real_matte_is_detected(self, monkeypatch):
        # what cropdetect actually reported for shot 0058
        assert self._detect(monkeypatch, "[Parsed_cropdetect] crop=1280:534:0:86") \
            == (1280, 534, 0, 86)

    def test_a_full_frame_clip_is_left_alone(self, monkeypatch):
        assert self._detect(monkeypatch, "[Parsed_cropdetect] crop=1280:704:0:0") is None

    def test_a_few_dark_rows_are_not_a_matte(self, monkeypatch):
        """Night skies and shadowed ceilings must not zoom the shot."""
        assert self._detect(monkeypatch, "[Parsed_cropdetect] crop=1280:690:0:7") is None

    def test_an_implausibly_deep_crop_is_refused(self, monkeypatch):
        """Past a point it is a dark shot, not a matte."""
        assert self._detect(monkeypatch, "[Parsed_cropdetect] crop=1280:200:0:252") is None

    def test_pillarboxing_is_out_of_scope(self, monkeypatch):
        assert self._detect(monkeypatch, "[Parsed_cropdetect] crop=900:704:190:0") is None

    def test_unparseable_output_is_survivable(self, monkeypatch):
        assert self._detect(monkeypatch, "no crop here at all") is None

    def test_every_part_of_a_shot_shares_one_crop(self, monkeypatch, tmp_path):
        """Per-part detection would move the jump into the middle of a shot."""
        import stitch

        boxes = []
        monkeypatch.setattr(stitch, "_detect_letterbox", lambda p: (1280, 534, 0, 86))

        def fake_strip(src, dst, box=None):
            boxes.append(box)
            open(dst, "w").close()
            return True

        monkeypatch.setattr(stitch, "_strip_letterbox", fake_strip)
        parts = []
        for n in (1, 2, 3):
            p = tmp_path / f"0846-0{n}.mp4"
            p.write_text("x")
            parts.append(str(p))

        out = stitch._strip_letterbox_group(parts, str(tmp_path / "work"))
        assert len(out) == 3
        assert boxes == [(1280, 534, 0, 86)] * 3  # detected once, applied to all

    def test_a_clean_shot_passes_through_untouched(self, monkeypatch, tmp_path):
        import stitch
        monkeypatch.setattr(stitch, "_detect_letterbox", lambda p: None)
        parts = [str(tmp_path / "0011.mp4")]
        assert stitch._strip_letterbox_group(parts, str(tmp_path / "work")) == parts

    def test_a_failed_crop_keeps_the_original_rather_than_losing_the_shot(
            self, monkeypatch, tmp_path):
        import stitch
        monkeypatch.setattr(stitch, "_detect_letterbox", lambda p: (1280, 534, 0, 86))

        def boom(src, dst, box=None):
            raise RuntimeError("ffmpeg fell over")

        monkeypatch.setattr(stitch, "_strip_letterbox", boom)
        src = str(tmp_path / "0058.mp4")
        assert stitch._strip_letterbox_group([src], str(tmp_path / "work")) == [src]


from prompt_format import strip_character_names  # noqa: E402


class TestStripCharacterNames:
    """Names are kept in the encode for stage 3, and removed before the model.

    Naming a character lets the model reconstruct it from world knowledge
    rather than from the compressed description (research 0021). The names
    cannot simply be banned at encode time: `_text_match_cast` matches the
    TMDB cast against the `subjects` text to work out who is in each shot.
    So they survive in shots.json and die here.
    """

    NAMES = ["C-3PO", "R2-D2", "Luke Skywalker", "Luke", "Darth Vader", "Vader", "Han"]

    def strip(self, text):
        return strip_character_names(text, self.NAMES)

    def test_a_parenthetical_gloss_is_dropped_whole(self):
        assert self.strip(
            "A golden humanoid droid (C-3PO) and a blue astromech droid (R2-D2) wait."
        ) == "A golden humanoid droid and a blue astromech droid wait."

    def test_a_comma_appositive_loses_both_commas(self):
        """'a droid, C-3PO, stands' must not become 'a droid, stands'."""
        assert self.strip("A golden humanoid droid, C-3PO, stands center frame.") \
            == "A golden humanoid droid stands center frame."

    def test_a_bare_name_after_a_noun_just_goes(self):
        assert self.strip("The droid R2-D2 is in the foreground.") \
            == "The droid is in the foreground."

    def test_a_governing_preposition_goes_with_it(self):
        """Otherwise 'similar in appearance to C-3PO but silver' strands 'to'."""
        assert self.strip("A droid, similar in appearance to C-3PO but silver, waits.") \
            == "A droid, similar in appearance but silver, waits."

    def test_a_stranded_copula_is_repaired_and_recapitalised(self):
        """Canonical descriptions read 'C-3PO is a tall golden droid'."""
        assert self.strip("C-3PO is a tall, golden humanoid droid.") \
            == "A tall, golden humanoid droid."

    def test_the_identity_block_label_is_removed(self):
        assert self.strip("C-3PO: a tall golden droid. Cinematic wide shot.") \
            == "A tall golden droid. Cinematic wide shot."

    def test_a_full_name_is_preferred_over_its_parts(self):
        """Stripping 'Luke' first would strand 'Skywalker'."""
        assert "Skywalker" not in self.strip("Luke Skywalker walks into the hangar.")

    def test_text_with_no_names_is_untouched(self):
        text = "A corridor with no characters at all."
        assert self.strip(text) == text

    def test_an_empty_name_list_changes_nothing(self):
        text = "C-3PO and R2-D2 wait."
        assert strip_character_names(text, []) == text

    def test_empty_text_is_survivable(self):
        assert strip_character_names("", self.NAMES) == ""


class TestIdentityBlockCarriesNoName:
    """The identity block used to prepend '<display_name>: <description>'.

    That put a character name in front of 1828 of 2069 shots (88.4%) -- far
    more exposure than the 389 whose `subjects` text names anyone.
    """

    def _strategy(self):
        from strategies_video import CharacterIdentityMixin

        s = CharacterIdentityMixin.__new__(CharacterIdentityMixin)
        s._init_character_identity(
            {7: ["c_3po"]},
            {"characters": [{
                "name": "c_3po",
                "display_name": "C-3PO",
                "description": "C-3PO is a tall, golden humanoid droid.",
                "shots": [7],
            }]},
        )
        return s

    def test_the_description_arrives_without_the_name(self):
        out = self._strategy()._prepend_identity({"index": 7}, "Cinematic wide shot.")
        assert "C-3PO" not in out
        assert "tall, golden humanoid droid" in out
        assert out.startswith("A tall")  # copula repaired, not left as "is a tall"

    def test_a_shot_with_no_identity_block_is_still_stripped(self):
        """Returning early here leaked Greedo in 757 and Vader in 1977."""
        out = self._strategy()._prepend_identity(
            {"index": 999}, "A protocol droid, C-3PO, waits.")
        assert "C-3PO" not in out

    def test_a_shot_with_no_index_is_still_stripped(self):
        out = self._strategy()._prepend_identity({}, "C-3PO waits.")
        assert "C-3PO" not in out


class TestDesignNamesSurviveTheStrip:
    """Not every name is contamination.

    "Luke Skywalker" makes the model recall an actor's face -- that is what
    0021 objects to. "Stormtrooper" is not a person; it is the most compact
    description that armour has. Stripping it turned a corridor of
    stormtroopers into generic soldiers in white, even though the prompt
    still read "Imperial soldiers clad in distinctive white armor". The one
    word carried more than the paragraph.
    """

    def _strategy(self):
        from strategies_video import CharacterIdentityMixin

        s = CharacterIdentityMixin.__new__(CharacterIdentityMixin)
        s._init_character_identity(
            {1: ["stormtrooper"], 2: ["luke"], 3: ["stormtrooper", "luke"]},
            {"characters": [
                {"name": "stormtrooper", "display_name": "Stormtrooper",
                 "description": "Imperial soldiers in white armor.",
                 "shots": [1, 3], "keep_name": True},
                {"name": "luke", "display_name": "Luke Skywalker",
                 "description": "A young man with sandy hair.",
                 "shots": [2, 3]},
            ]},
        )
        return s

    def test_a_design_name_is_kept_and_labelled(self):
        out = self._strategy()._prepend_identity({"index": 1}, "Cinematic wide shot.")
        assert out.startswith("Stormtrooper: Imperial soldiers in white armor.")

    def test_a_personal_name_is_still_stripped(self):
        out = self._strategy()._prepend_identity({"index": 2}, "Cinematic wide shot.")
        assert "Luke" not in out and "Skywalker" not in out
        assert "young man with sandy hair" in out

    def test_both_rules_apply_in_one_shot(self):
        out = self._strategy()._prepend_identity({"index": 3}, "Cinematic wide shot.")
        assert "Stormtrooper" in out
        assert "Luke" not in out and "Skywalker" not in out

    def test_a_design_name_survives_in_the_shot_text_too(self):
        """The describe pass writes "stormtroopers in white armor" -- keep it."""
        out = self._strategy()._prepend_identity(
            {"index": 1}, "Three Stormtrooper figures advance.")
        assert "Stormtrooper figures advance" in out

    def test_only_flagged_characters_are_spared(self):
        s = self._strategy()
        assert "Luke Skywalker" in s._all_character_names()
        assert "Stormtrooper" not in s._all_character_names()


class TestStaleAudioIsRefusedAtStitch:
    """The stitch must not mux audio from a different encode.

    Auto-discovery sweeps every directory under audio/, and clips are keyed
    only by shot index -- nothing in the filename says which encode made them.
    Re-encoding renumbers shots, so March audio for "shot 794" silently became
    audio for entirely different footage. The middle-third render shipped with
    speech and ambience built from clips dated March, June and 1 August, none
    of them generated for that encode, and the mux reported no error.
    """

    def _dirs(self, tmp_path, strategy="mmaudio"):
        import manifest, os

        os.makedirs(manifest.audio_dir(str(tmp_path), strategy), exist_ok=True)
        return str(tmp_path)

    def _shots(self, out):
        import json, os

        p = os.path.join(out, "shots.json")
        with open(p, "w") as f:
            json.dump({"format": "v2", "shots": [
                {"index": 0, "start_s": 0.0, "end_s": 1.0, "duration_s": 1.0}],
                "dialog": []}, f)
        return p

    def test_audio_from_the_current_encode_is_accepted(self, tmp_path):
        import json, manifest, stitch

        out = self._dirs(tmp_path)
        self._shots(out)
        with open(manifest.audio_progress_path(out, "mmaudio"), "w") as f:
            json.dump({"encode_fingerprint": manifest.encode_fingerprint(out)}, f)

        assert stitch._audio_matches_encode(out, "mmaudio") is True

    def test_audio_from_a_different_encode_is_refused(self, tmp_path):
        import json, manifest, stitch

        out = self._dirs(tmp_path)
        self._shots(out)
        with open(manifest.audio_progress_path(out, "mmaudio"), "w") as f:
            json.dump({"encode_fingerprint": "deadbeefcafe"}, f)

        assert stitch._audio_matches_encode(out, "mmaudio") is False

    def test_unstamped_audio_older_than_the_encode_is_refused(self, tmp_path):
        """The March/June/August dirs had no progress file at all."""
        import os, time, manifest, stitch

        out = self._dirs(tmp_path)
        audio = manifest.audio_dir(out, "mmaudio")
        old = time.time() - 86_400
        os.utime(audio, (old, old))
        self._shots(out)  # written now, so newer than the audio

        assert stitch._audio_matches_encode(out, "mmaudio") is False

    def test_a_corrupt_progress_file_is_refused_rather_than_trusted(self, tmp_path):
        import manifest, stitch

        out = self._dirs(tmp_path)
        self._shots(out)
        with open(manifest.audio_progress_path(out, "mmaudio"), "w") as f:
            f.write("{ not json")

        assert stitch._audio_matches_encode(out, "mmaudio") is False


class TestReprovisionWhenPodDiesMidRun:
    """with_setup_retry only covers setup. A pod can die mid-generation.

    16 hours into a 690-shot render the account balance hit zero and RunPod
    reclaimed the pod. Every request 404'd, and the clip retry loop kept
    retrying a host that no longer existed -- three attempts a shot, heading
    for "Too many errors" 20 shots later.
    """

    def _strategy(self, alive, monkeypatch):
        from strategies_video import RunPodWanStrategy

        s = RunPodWanStrategy.__new__(RunPodWanStrategy)

        class Session:
            def __init__(self):
                self.forgotten = False
                self.pod_id = "p1"        # we had a pod; the question is whether it lives
            def pod_alive(self):
                return alive
            def forget_pod(self):
                self.forgotten = True
                self.pod_id = None

        s._session = Session()
        s._setup_done = True
        s.setup_calls = []
        monkeypatch.setattr(type(s), "_ensure_pod",
                            lambda self: self.setup_calls.append(1), raising=False)
        return s

    def test_a_dead_pod_is_replaced(self, monkeypatch):
        s = self._strategy(alive=False, monkeypatch=monkeypatch)

        assert s._reprovision_if_pod_gone() is True
        assert s._session.forgotten is True      # no misleading terminate()
        assert s._setup_done is False            # forces a real re-setup
        assert s.setup_calls == [1]

    def test_a_live_pod_is_left_alone(self, monkeypatch):
        s = self._strategy(alive=True, monkeypatch=monkeypatch)

        assert s._reprovision_if_pod_gone() is False
        assert s._session.forgotten is False
        assert s.setup_calls == []

    def test_a_strategy_with_no_session_does_not_crash(self):
        from strategies_video import RunPodWanStrategy

        s = RunPodWanStrategy.__new__(RunPodWanStrategy)
        assert s._reprovision_if_pod_gone() is False

    def test_the_clip_loop_retries_once_on_the_new_pod(self, monkeypatch, tmp_path):
        """A replaced pod gets an immediate attempt, not another 40s wait."""
        from strategies_video import RunPodWanStrategy

        s = self._strategy(alive=False, monkeypatch=monkeypatch)
        attempts = []

        def fake_attempt(prompt, clips_dir, clip_name, frames, seed, start_image=None):
            attempts.append(len(attempts))
            return "clip" if len(attempts) == 2 else None

        monkeypatch.setattr(s, "_attempt_one_clip", fake_attempt)
        monkeypatch.setattr("strategies_video.time.sleep", lambda _: None)

        assert s._generate_one_clip("p", str(tmp_path), "c.mp4", 81, 7) == "clip"
        assert len(attempts) == 2  # original, then straight onto the fresh pod

    def test_a_session_that_never_had_a_pod_is_not_reprovisioned(self, monkeypatch):
        """No pod id means setup has not run -- not that a pod died under us."""
        from strategies_video import RunPodWanStrategy

        s = RunPodWanStrategy.__new__(RunPodWanStrategy)

        class NeverProvisioned:
            pod_id = None
            def pod_alive(self):
                raise AssertionError("must not be consulted")

        s._session = NeverProvisioned()
        assert s._reprovision_if_pod_gone() is False
