"""Tests for decode.py strategy pattern and duration logic."""

import json
import os
import threading
from unittest.mock import MagicMock, patch

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
