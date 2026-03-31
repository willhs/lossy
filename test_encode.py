"""Tests for encode.py data transformation functions."""

import argparse
import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from encode import (
    aggregate_shot_audio,
    align_subtitles_to_shots,
    class_to_bucket,
    classify_motion,
    frames_for_duration,
    parse_srt,
    run_stage3,
)


# ---------------------------------------------------------------------------
# frames_for_duration
# ---------------------------------------------------------------------------

class TestFramesForDuration:
    def test_very_short(self):
        assert frames_for_duration(0.5) == 2

    def test_boundary_1s(self):
        assert frames_for_duration(0.99) == 2
        assert frames_for_duration(1.0) == 4

    def test_medium(self):
        assert frames_for_duration(2.5) == 4

    def test_boundary_3s(self):
        assert frames_for_duration(2.99) == 4
        assert frames_for_duration(3.0) == 6

    def test_longer(self):
        assert frames_for_duration(5.0) == 6

    def test_boundary_8s(self):
        assert frames_for_duration(7.99) == 6
        assert frames_for_duration(8.0) == 8

    def test_very_long(self):
        assert frames_for_duration(30.0) == 8
        assert frames_for_duration(200.0) == 8


# ---------------------------------------------------------------------------
# parse_srt
# ---------------------------------------------------------------------------

class TestParseSrt:
    def test_basic(self, tmp_path):
        srt = tmp_path / "test.srt"
        srt.write_text(
            "1\n"
            "00:01:00,000 --> 00:01:02,500\n"
            "Hello there.\n"
            "\n"
            "2\n"
            "00:01:03,000 --> 00:01:05,000\n"
            "General Kenobi.\n"
        )
        entries = parse_srt(str(srt))
        assert len(entries) == 2
        assert entries[0]["text"] == "Hello there."
        assert entries[0]["start_s"] == 60.0
        assert entries[0]["end_s"] == 62.5
        assert entries[1]["text"] == "General Kenobi."

    def test_multiline_dialogue(self, tmp_path):
        srt = tmp_path / "test.srt"
        srt.write_text(
            "1\n"
            "00:00:10,000 --> 00:00:12,000\n"
            "Line one.\n"
            "Line two.\n"
        )
        entries = parse_srt(str(srt))
        assert len(entries) == 1
        assert entries[0]["text"] == "Line one. Line two."

    def test_html_tags_stripped(self, tmp_path):
        srt = tmp_path / "test.srt"
        srt.write_text(
            "1\n"
            "00:02:49,710 --> 00:02:52,500\n"
            '<font size="24">They shut down the main reactor.\n'
            "We'll be destroyed for sure.\n"
            "</font>\n"
        )
        entries = parse_srt(str(srt))
        assert len(entries) == 1
        assert "<font" not in entries[0]["text"]
        assert "</font>" not in entries[0]["text"]
        assert "They shut down" in entries[0]["text"]

    def test_dot_separator(self, tmp_path):
        """Some SRT files use . instead of , for milliseconds."""
        srt = tmp_path / "test.srt"
        srt.write_text(
            "1\n"
            "00:00:01.500 --> 00:00:03.750\n"
            "Dot separator.\n"
        )
        entries = parse_srt(str(srt))
        assert len(entries) == 1
        assert entries[0]["start_s"] == 1.5
        assert entries[0]["end_s"] == 3.75

    def test_empty_file(self, tmp_path):
        srt = tmp_path / "test.srt"
        srt.write_text("")
        entries = parse_srt(str(srt))
        assert entries == []

    def test_malformed_timestamp_skipped(self, tmp_path):
        srt = tmp_path / "test.srt"
        srt.write_text(
            "1\n"
            "NOT A TIMESTAMP\n"
            "Some text.\n"
            "\n"
            "2\n"
            "00:00:05,000 --> 00:00:07,000\n"
            "Valid entry.\n"
        )
        entries = parse_srt(str(srt))
        assert len(entries) == 1
        assert entries[0]["text"] == "Valid entry."


# ---------------------------------------------------------------------------
# align_subtitles_to_shots
# ---------------------------------------------------------------------------

class TestAlignSubtitlesToShots:
    def _scene(self, index, start_s, end_s):
        return {"index": index, "start_s": start_s, "end_s": end_s}

    def _sub(self, start_s, end_s, text):
        return {"start_s": start_s, "end_s": end_s, "text": text}

    def test_subtitle_within_shot(self):
        scenes = [self._scene(0, 10.0, 20.0)]
        subs = [self._sub(12.0, 15.0, "Hello")]
        result = align_subtitles_to_shots(subs, scenes)
        assert 0 in result
        assert result[0] == [{"text": "Hello", "start_s": 2.0, "end_s": 5.0}]

    def test_subtitle_spanning_two_shots(self):
        scenes = [
            self._scene(0, 10.0, 15.0),
            self._scene(1, 15.0, 20.0),
        ]
        subs = [self._sub(14.0, 16.0, "Spanning")]
        result = align_subtitles_to_shots(subs, scenes)
        assert 0 in result
        assert 1 in result
        # Shot 0: clamped to [4.0, 5.0] relative to shot start 10.0
        assert result[0][0]["text"] == "Spanning"
        assert result[0][0]["start_s"] == 4.0
        assert result[0][0]["end_s"] == 5.0
        # Shot 1: clamped to [0.0, 1.0] relative to shot start 15.0
        assert result[1][0]["text"] == "Spanning"
        assert result[1][0]["start_s"] == 0.0
        assert result[1][0]["end_s"] == 1.0

    def test_no_dialogue(self):
        scenes = [self._scene(0, 0.0, 10.0)]
        subs = [self._sub(20.0, 22.0, "Late")]
        result = align_subtitles_to_shots(subs, scenes)
        assert result == {}

    def test_multiple_subs_in_one_shot(self):
        scenes = [self._scene(0, 0.0, 10.0)]
        subs = [
            self._sub(1.0, 3.0, "First"),
            self._sub(5.0, 7.0, "Second"),
        ]
        result = align_subtitles_to_shots(subs, scenes)
        assert 0 in result
        assert len(result[0]) == 2
        assert result[0][0] == {"text": "First", "start_s": 1.0, "end_s": 3.0}
        assert result[0][1] == {"text": "Second", "start_s": 5.0, "end_s": 7.0}

    def test_exact_boundary_no_overlap(self):
        """Subtitle ends exactly when shot starts -- no overlap."""
        scenes = [self._scene(0, 10.0, 20.0)]
        subs = [self._sub(5.0, 10.0, "Before")]
        result = align_subtitles_to_shots(subs, scenes)
        assert result == {}

    def test_exact_boundary_overlap(self):
        """Subtitle starts exactly when shot starts -- overlaps."""
        scenes = [self._scene(0, 10.0, 20.0)]
        subs = [self._sub(10.0, 12.0, "Exact")]
        result = align_subtitles_to_shots(subs, scenes)
        assert 0 in result
        assert result[0] == [{"text": "Exact", "start_s": 0.0, "end_s": 2.0}]


# ---------------------------------------------------------------------------
# classify_motion
# ---------------------------------------------------------------------------

class TestClassifyMotion:
    def test_static(self):
        angles = np.zeros(1000)
        mags = np.ones(1000) * 0.5  # below default threshold of 2.0
        assert classify_motion(angles, mags) == "static"

    def test_pan_right(self):
        """Uniform rightward flow (angle ~0)."""
        angles = np.zeros(1000)  # 0 radians = rightward
        mags = np.ones(1000) * 5.0
        assert classify_motion(angles, mags) == "pan right"

    def test_pan_left(self):
        """Uniform leftward flow (angle ~pi)."""
        angles = np.full(1000, np.pi)
        mags = np.ones(1000) * 5.0
        assert classify_motion(angles, mags) == "pan left"

    def test_tilt_down(self):
        """Uniform downward flow (angle ~pi/2)."""
        angles = np.full(1000, np.pi / 2)
        mags = np.ones(1000) * 5.0
        assert classify_motion(angles, mags) == "tilt down"

    def test_tilt_up(self):
        """Uniform upward flow (angle ~3pi/2)."""
        angles = np.full(1000, 3 * np.pi / 2)
        mags = np.ones(1000) * 5.0
        assert classify_motion(angles, mags) == "tilt up"

    def test_zoom(self):
        """High magnitude variance suggests zoom."""
        angles = np.random.uniform(0, 2 * np.pi, 1000)
        # High std relative to mean triggers zoom
        mags = np.concatenate([np.ones(500) * 1.0, np.ones(500) * 10.0])
        result = classify_motion(angles, mags)
        assert result == "zoom"

    def test_custom_threshold(self):
        angles = np.zeros(1000)
        mags = np.ones(1000) * 3.0
        assert classify_motion(angles, mags, threshold=5.0) == "static"
        assert classify_motion(angles, mags, threshold=2.0) == "pan right"


# ---------------------------------------------------------------------------
# class_to_bucket
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# aggregate_shot_audio
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# run_stage3
# ---------------------------------------------------------------------------

class TestStage3:
    """Tests for encode stage 3 — character registry."""

    def test_stage3_produces_characters_json(self, tmp_path, monkeypatch):
        """Stage 3 reads prompts.json subjects and writes characters.json."""
        prompts = [
            {"index": 0, "description": {"subjects": "Luke, a young man with sandy blond hair"}},
            {"index": 1, "description": {"subjects": "Han Solo, a roguish man in a vest"}},
            {"index": 2, "description": {"subjects": "Luke wearing a white tunic"}},
            {"index": 3, "description": {"subjects": "Han Solo shooting a blaster"}},
        ]
        (tmp_path / "prompts.json").write_text(json.dumps(prompts))

        mock_response = {
            "characters": [
                {
                    "name": "luke",
                    "display_name": "Luke Skywalker",
                    "description": "Young man, early 20s, sandy blond hair...",
                    "shots": [0, 2],
                },
                {
                    "name": "han_solo",
                    "display_name": "Han Solo",
                    "description": "Roguish man, mid 30s, dark hair...",
                    "shots": [1, 3],
                },
            ]
        }

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(
            text=json.dumps(mock_response)
        )
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setattr("google.genai.Client", lambda **kwargs: mock_client)

        args = argparse.Namespace(output_dir=str(tmp_path))
        run_stage3(args)

        characters_path = tmp_path / "characters.json"
        assert characters_path.exists()
        data = json.loads(characters_path.read_text())
        assert len(data["characters"]) == 2
        assert data["characters"][0]["name"] == "luke"
        assert 0 in data["characters"][0]["shots"]

    def test_stage3_missing_prompts_exits(self, tmp_path):
        """stage3 exits with error if prompts.json is missing."""
        args = argparse.Namespace(output_dir=str(tmp_path))
        with pytest.raises(SystemExit):
            run_stage3(args)

    def test_stage3_no_subjects_exits(self, tmp_path, monkeypatch):
        """stage3 exits if prompts.json has no subjects fields."""
        prompts = [
            {"index": 0, "description": {"action": "A door opens."}},
        ]
        (tmp_path / "prompts.json").write_text(json.dumps(prompts))
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        args = argparse.Namespace(output_dir=str(tmp_path))
        with pytest.raises(SystemExit):
            run_stage3(args)

    def test_stage3_missing_api_key_exits(self, tmp_path, monkeypatch):
        """stage3 exits if GEMINI_API_KEY is not set."""
        prompts = [
            {"index": 0, "description": {"subjects": "Luke"}},
        ]
        (tmp_path / "prompts.json").write_text(json.dumps(prompts))
        # Set to empty string (falsy) so setdefault() won't overwrite it with
        # the real key from the .env file.
        monkeypatch.setenv("GEMINI_API_KEY", "")

        args = argparse.Namespace(output_dir=str(tmp_path))
        with pytest.raises(SystemExit):
            run_stage3(args)

# ---------------------------------------------------------------------------
# run_stage3 — v2 format compat
# ---------------------------------------------------------------------------

class TestStage3V2Compat:
    """Stage 3 handles both v1 (flat array) and v2 (object) prompts.json."""

    def test_stage3_reads_v2_format(self, tmp_path, monkeypatch):
        """stage3 correctly reads subjects from v2-format prompts.json."""
        prompts_v2 = {
            "format": "v2",
            "shots": [
                {"index": 0, "description": {"subjects": "Luke, a young man with sandy blond hair"}},
                {"index": 1, "description": {"subjects": "Luke wearing a white tunic"}},
            ],
            "dialog": [{"text": "May the Force be with you.", "start_s": 5.1, "end_s": 7.0}],
        }
        (tmp_path / "prompts.json").write_text(json.dumps(prompts_v2))

        mock_response = {
            "characters": [
                {"name": "luke", "display_name": "Luke Skywalker",
                 "description": "Young man, early 20s, sandy blond hair.", "shots": [0, 1]},
            ]
        }
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(
            text=json.dumps(mock_response)
        )
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setattr("google.genai.Client", lambda **kwargs: mock_client)

        args = argparse.Namespace(output_dir=str(tmp_path))
        run_stage3(args)

        data = json.loads((tmp_path / "characters.json").read_text())
        assert data["characters"][0]["name"] == "luke"
