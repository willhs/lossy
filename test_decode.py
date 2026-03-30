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
    filter_speech_from_sound,
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
# Phase 2: Portrait generation
# ---------------------------------------------------------------------------

from decode import generate_portraits, build_character_shot_map  # noqa: E402


class TestBuildCharacterShotMap:
    """Tests for build_character_shot_map — pure function, no mocks needed."""

    def test_basic_mapping(self):
        characters_data = {
            "characters": [
                {"name": "luke", "shots": [0, 2, 4]},
                {"name": "han_solo", "shots": [1, 3]},
            ]
        }
        result = build_character_shot_map(characters_data)
        assert result[0] == ["luke"]
        assert result[1] == ["han_solo"]
        assert result[2] == ["luke"]
        assert result[3] == ["han_solo"]
        assert result[4] == ["luke"]

    def test_shared_shot(self):
        characters_data = {
            "characters": [
                {"name": "luke", "shots": [5]},
                {"name": "leia", "shots": [5]},
            ]
        }
        result = build_character_shot_map(characters_data)
        assert "luke" in result[5]
        assert "leia" in result[5]

    def test_empty_characters(self):
        assert build_character_shot_map({"characters": []}) == {}

    def test_character_with_no_shots(self):
        characters_data = {"characters": [{"name": "extra", "shots": []}]}
        assert build_character_shot_map(characters_data) == {}

    def test_missing_characters_key(self):
        assert build_character_shot_map({}) == {}


class TestGeneratePortraits:
    """Tests for generate_portraits with mocked fal_client."""

    def _make_characters_json(self, tmp_path, characters):
        data = {"characters": characters}
        path = tmp_path / "characters.json"
        path.write_text(json.dumps(data))
        return str(path)

    def test_generates_portrait_and_saves_file(self, tmp_path):
        characters_path = self._make_characters_json(tmp_path, [
            {
                "name": "luke",
                "display_name": "Luke Skywalker",
                "description": "Young man, sandy blond hair, blue eyes.",
                "shots": [0, 2],
            }
        ])

        fake_image_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

        mock_response = MagicMock()
        mock_response.content = fake_image_bytes
        mock_response.raise_for_status = MagicMock()

        with patch("fal_client.subscribe", return_value={"images": [{"url": "https://example.com/portrait.png"}]}):
            with patch("httpx.get", return_value=mock_response):
                result = generate_portraits(characters_path, str(tmp_path))

        assert "luke" in result
        portrait_path = tmp_path / "characters" / "luke.png"
        assert portrait_path.exists()
        assert portrait_path.read_bytes() == fake_image_bytes

    def test_skips_existing_portrait(self, tmp_path):
        characters_path = self._make_characters_json(tmp_path, [
            {
                "name": "vader",
                "display_name": "Darth Vader",
                "description": "Tall figure in black armor.",
                "shots": [10],
            }
        ])

        # Pre-create the portrait file
        characters_dir = tmp_path / "characters"
        characters_dir.mkdir()
        portrait_file = characters_dir / "vader.png"
        portrait_file.write_bytes(b"existing")

        with patch("fal_client.subscribe") as mock_subscribe:
            with patch("httpx.get") as mock_get:
                result = generate_portraits(characters_path, str(tmp_path))

        mock_subscribe.assert_not_called()
        mock_get.assert_not_called()
        assert result["vader"] == str(portrait_file)

    def test_handles_fal_error_gracefully(self, tmp_path):
        characters_path = self._make_characters_json(tmp_path, [
            {
                "name": "han_solo",
                "display_name": "Han Solo",
                "description": "Roguish man, dark hair.",
                "shots": [1],
            }
        ])

        with patch("fal_client.subscribe", side_effect=RuntimeError("API error")):
            result = generate_portraits(characters_path, str(tmp_path))

        assert result == {}

    def test_returns_empty_for_no_characters(self, tmp_path):
        characters_path = self._make_characters_json(tmp_path, [])
        result = generate_portraits(characters_path, str(tmp_path))
        assert result == {}

    def test_prompt_includes_description(self, tmp_path):
        characters_path = self._make_characters_json(tmp_path, [
            {
                "name": "leia",
                "display_name": "Princess Leia",
                "description": "Young woman, brown hair in buns, white dress.",
                "shots": [3],
            }
        ])

        captured_args = {}

        def fake_subscribe(model, arguments, with_logs):
            captured_args["prompt"] = arguments["prompt"]
            return {"images": [{"url": "https://example.com/leia.png"}]}

        mock_response = MagicMock()
        mock_response.content = b"PNG"
        mock_response.raise_for_status = MagicMock()

        with patch("fal_client.subscribe", side_effect=fake_subscribe):
            with patch("httpx.get", return_value=mock_response):
                generate_portraits(characters_path, str(tmp_path))

        assert "brown hair in buns" in captured_args["prompt"]


# ---------------------------------------------------------------------------
# Phase 3: VACE Strategy
# ---------------------------------------------------------------------------

import argparse  # noqa: E402

from decode import RunPodVaceStrategy, _create_vace_strategy  # noqa: E402


class TestRunPodVaceStrategy:
    """Tests for RunPodVaceStrategy."""

    def test_name(self):
        assert RunPodVaceStrategy.name == "runpod-vace"

    def test_inherits_from_runpod_wan(self):
        assert issubclass(RunPodVaceStrategy, RunPodWanStrategy)

    def test_vace_models_list(self):
        assert any("vace_1.3B" in url for _, url in RunPodVaceStrategy.VACE_MODELS)
        assert any("vae" in path for path, _ in RunPodVaceStrategy.VACE_MODELS)
        assert any("text_encoders" in path for path, _ in RunPodVaceStrategy.VACE_MODELS)

    def test_build_vace_workflow_has_required_nodes(self):
        strategy = RunPodVaceStrategy()
        workflow = strategy._build_vace_workflow("a hero walks", seed=42, length=81, reference_image="luke.png")

        node_types = {v["class_type"] for v in workflow.values()}
        assert "WanVaceToVideo" in node_types
        assert "TrimVideoLatent" in node_types
        assert "LoadImage" in node_types
        assert "UNETLoader" in node_types
        assert "KSampler" in node_types
        assert "VAEDecode" in node_types
        assert "SaveWEBM" in node_types

    def test_build_vace_workflow_reference_image_injected(self):
        strategy = RunPodVaceStrategy()
        workflow = strategy._build_vace_workflow("test", seed=1, length=49, reference_image="vader.png")

        load_image_node = next(
            v for v in workflow.values() if v["class_type"] == "LoadImage"
        )
        assert load_image_node["inputs"]["image"] == "vader.png"

    def test_build_vace_workflow_prompt_injected(self):
        strategy = RunPodVaceStrategy()
        workflow = strategy._build_vace_workflow("rebel fighter", seed=5, length=33, reference_image="leia.png")

        clip_encode_node = next(
            v for v in workflow.values()
            if v["class_type"] == "CLIPTextEncode" and "rebel fighter" in v["inputs"].get("text", "")
        )
        assert clip_encode_node is not None

    def test_build_vace_workflow_uses_vace_model(self):
        strategy = RunPodVaceStrategy()
        workflow = strategy._build_vace_workflow("test", seed=1, length=81, reference_image="ref.png")

        unet_node = next(v for v in workflow.values() if v["class_type"] == "UNETLoader")
        assert "vace" in unet_node["inputs"]["unet_name"]

    def test_build_vace_workflow_strength_is_one(self):
        strategy = RunPodVaceStrategy()
        workflow = strategy._build_vace_workflow("test", seed=1, length=81, reference_image="ref.png")

        vace_node = next(v for v in workflow.values() if v["class_type"] == "WanVaceToVideo")
        assert vace_node["inputs"]["strength"] == 1.0

    def test_build_vace_workflow_trim_latent_uses_vace_output(self):
        """TrimVideoLatent must take its trim_amount from WanVaceToVideo output [3]."""
        strategy = RunPodVaceStrategy()
        workflow = strategy._build_vace_workflow("test", seed=1, length=81, reference_image="ref.png")

        vace_node_id = next(k for k, v in workflow.items() if v["class_type"] == "WanVaceToVideo")
        trim_node = next(v for v in workflow.values() if v["class_type"] == "TrimVideoLatent")
        assert trim_node["inputs"]["trim_amount"] == [vace_node_id, 3]

    def test_build_vace_workflow_has_chinese_negative_prompt(self):
        strategy = RunPodVaceStrategy()
        workflow = strategy._build_vace_workflow("test", seed=1, length=81, reference_image="ref.png")

        neg_nodes = [
            v for v in workflow.values()
            if v["class_type"] == "CLIPTextEncode"
            and any('\u4e00' <= c <= '\u9fff' for c in v["inputs"].get("text", ""))
        ]
        assert len(neg_nodes) == 1

    def test_init_stores_portraits_and_shot_map(self):
        portraits = {"luke": "/tmp/luke.png"}
        shot_map = {0: ["luke"], 2: ["luke"]}
        strategy = RunPodVaceStrategy(portraits=portraits, character_shot_map=shot_map)
        assert strategy._portraits == portraits
        assert strategy._character_shot_map == shot_map

    def test_init_defaults_to_empty(self):
        strategy = RunPodVaceStrategy()
        assert strategy._portraits == {}
        assert strategy._character_shot_map == {}
        assert strategy._uploaded_portraits == {}


class TestCreateVaceStrategy:
    """Tests for _create_vace_strategy helper."""

    def test_without_characters_json(self, tmp_path):
        """Falls back gracefully when characters.json is absent."""
        args = argparse.Namespace(
            output_dir=str(tmp_path),
            keep_pod=False,
            concurrent_audio=False,
        )
        strategy = _create_vace_strategy(args)
        assert isinstance(strategy, RunPodVaceStrategy)
        assert strategy._portraits == {}
        assert strategy._character_shot_map == {}

    def test_with_characters_json(self, tmp_path):
        """Loads characters.json, generates portraits, builds shot map."""
        characters_data = {
            "characters": [
                {"name": "luke", "display_name": "Luke Skywalker",
                 "description": "Young man, sandy blond hair.", "shots": [0, 2]},
            ]
        }
        (tmp_path / "characters.json").write_text(json.dumps(characters_data))

        # Pre-create portrait so generate_portraits skips fal_client
        characters_dir = tmp_path / "characters"
        characters_dir.mkdir()
        (characters_dir / "luke.png").write_bytes(b"PNG")

        args = argparse.Namespace(
            output_dir=str(tmp_path),
            keep_pod=False,
            concurrent_audio=False,
        )
        strategy = _create_vace_strategy(args)
        assert isinstance(strategy, RunPodVaceStrategy)
        assert "luke" in strategy._portraits
        assert strategy._character_shot_map[0] == ["luke"]
        assert strategy._character_shot_map[2] == ["luke"]
