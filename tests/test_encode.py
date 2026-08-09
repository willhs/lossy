"""Tests for encode.py data transformation functions."""

import argparse
import json
from unittest.mock import MagicMock

import numpy as np
import pytest

import manifest
from encode import (
    _build_shot_character_map,
    _candidate_characters_for_line,
    _character_dedupe_key,
    _collapse_duplicate_characters,
    _normalize_character_name,
    _refine_shot_assignments,
    _run_supervised_stage3,
    _text_match_cast,
    _to_name_key,
    aggregate_shot_audio,
    align_subtitles_to_shots,
    class_to_bucket,
    classify_motion,
    frames_for_duration,
    generate_prompts,
    generate_temporal_segments,
    parse_srt,
    run_stage3,
    run_stage4,
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

def _mock_stage3(tmp_path, monkeypatch, prompts_data, mock_response):
    """Helper: write shots.json, mock Gemini, run stage3, return parsed characters.json."""
    (tmp_path / "shots.json").write_text(json.dumps(prompts_data))

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = MagicMock(
        text=json.dumps(mock_response)
    )
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr("google.genai.Client", lambda **kwargs: mock_client)

    args = argparse.Namespace(output_dir=str(tmp_path))
    run_stage3(args)

    return json.loads((tmp_path / "characters.json").read_text())


# Shared test fixtures
PROMPTS_V1 = [
    {"index": 0, "description": {"subjects": "Luke, a young man with sandy blond hair"}},
    {"index": 1, "description": {"subjects": "Han Solo, a roguish man in a vest"}},
    {"index": 2, "description": {"subjects": "Luke wearing a white tunic"}},
    {"index": 3, "description": {"subjects": "Han Solo shooting a blaster"}},
]

MOCK_CHARACTERS = {
    "characters": [
        {
            "name": "luke",
            "display_name": "Luke Skywalker",
            "description": "Young man, early 20s, sandy blond hair, blue eyes, wearing a white tunic and utility belt.",
            "shots": [0, 2],
        },
        {
            "name": "han_solo",
            "display_name": "Han Solo",
            "description": "Roguish man, mid 30s, dark hair, wearing a black vest over white shirt.",
            "shots": [1, 3],
        },
    ]
}


class TestStage3OutputStructure:
    """SPEC-200: Character registry output structure (REQ-001 to REQ-007)."""

    @pytest.mark.req("SPEC-200/REQ-001")
    def test_writes_characters_json(self, tmp_path, monkeypatch):
        """REQ-001: Stage 3 shall write a characters.json file."""
        _mock_stage3(tmp_path, monkeypatch, PROMPTS_V1, MOCK_CHARACTERS)
        assert (tmp_path / "characters.json").exists()

    @pytest.mark.req("SPEC-200/REQ-002")
    def test_has_characters_array(self, tmp_path, monkeypatch):
        """REQ-002: characters.json shall contain a "characters" array."""
        data = _mock_stage3(tmp_path, monkeypatch, PROMPTS_V1, MOCK_CHARACTERS)
        assert isinstance(data["characters"], list)

    @pytest.mark.req("SPEC-200/REQ-003")
    def test_name_is_lowercase_underscored(self, tmp_path, monkeypatch):
        """REQ-003: name shall be lowercase with underscores, no spaces."""
        data = _mock_stage3(tmp_path, monkeypatch, PROMPTS_V1, MOCK_CHARACTERS)
        for char in data["characters"]:
            assert char["name"] == char["name"].lower()
            assert " " not in char["name"]

    @pytest.mark.req("SPEC-200/REQ-004")
    def test_has_display_name(self, tmp_path, monkeypatch):
        """REQ-004: Each character shall have a display_name."""
        data = _mock_stage3(tmp_path, monkeypatch, PROMPTS_V1, MOCK_CHARACTERS)
        for char in data["characters"]:
            assert "display_name" in char
            assert len(char["display_name"]) > 0

    @pytest.mark.req("SPEC-200/REQ-005")
    def test_has_description(self, tmp_path, monkeypatch):
        """REQ-005: Each character shall have a description."""
        data = _mock_stage3(tmp_path, monkeypatch, PROMPTS_V1, MOCK_CHARACTERS)
        for char in data["characters"]:
            assert "description" in char
            assert len(char["description"]) > 0

    @pytest.mark.req("SPEC-200/REQ-006")
    def test_shots_is_int_list(self, tmp_path, monkeypatch):
        """REQ-006: shots shall be a list of integer shot indices."""
        data = _mock_stage3(tmp_path, monkeypatch, PROMPTS_V1, MOCK_CHARACTERS)
        for char in data["characters"]:
            assert isinstance(char["shots"], list)
            assert all(isinstance(s, int) for s in char["shots"])

    @pytest.mark.req("SPEC-200/REQ-007")
    def test_shot_indices_match_prompts(self, tmp_path, monkeypatch):
        """REQ-007: Shot indices shall correspond to indices in shots.json."""
        data = _mock_stage3(tmp_path, monkeypatch, PROMPTS_V1, MOCK_CHARACTERS)
        valid_indices = {e["index"] for e in PROMPTS_V1}
        for char in data["characters"]:
            for shot_idx in char["shots"]:
                assert shot_idx in valid_indices


class TestStage3CharacterSelection:
    """SPEC-200: Character selection rules (REQ-010 to REQ-013)."""

    @pytest.mark.req("SPEC-200/REQ-010")
    def test_min_two_shots_in_prompt(self, tmp_path, monkeypatch):
        """REQ-010: Only characters in at least 2 shots shall be included.

        Verified via the system prompt sent to Gemini.
        """
        (tmp_path / "shots.json").write_text(json.dumps(PROMPTS_V1))
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(
            text=json.dumps(MOCK_CHARACTERS)
        )
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setattr("google.genai.Client", lambda **kwargs: mock_client)

        run_stage3(argparse.Namespace(output_dir=str(tmp_path)))

        call_args = mock_client.models.generate_content.call_args
        config = call_args.kwargs["config"]
        assert "at least 2 shots" in config.system_instruction

    @pytest.mark.req("SPEC-200/REQ-011")
    def test_max_five_characters_in_prompt(self, tmp_path, monkeypatch):
        """REQ-011: At most 5 characters in the registry.

        Verified via the system prompt sent to Gemini.
        """
        (tmp_path / "shots.json").write_text(json.dumps(PROMPTS_V1))
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(
            text=json.dumps(MOCK_CHARACTERS)
        )
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setattr("google.genai.Client", lambda **kwargs: mock_client)

        run_stage3(argparse.Namespace(output_dir=str(tmp_path)))

        call_args = mock_client.models.generate_content.call_args
        config = call_args.kwargs["config"]
        assert "5 most prominent" in config.system_instruction

    @pytest.mark.req("SPEC-200/REQ-013")
    def test_all_subjects_sent_to_gemini(self, tmp_path, monkeypatch):
        """REQ-013: All shot subjects shall be sent to Gemini for merging."""
        (tmp_path / "shots.json").write_text(json.dumps(PROMPTS_V1))
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(
            text=json.dumps(MOCK_CHARACTERS)
        )
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setattr("google.genai.Client", lambda **kwargs: mock_client)

        run_stage3(argparse.Namespace(output_dir=str(tmp_path)))

        call_args = mock_client.models.generate_content.call_args
        contents = call_args.kwargs["contents"]
        user_text = contents[0].parts[0].text
        # All subject descriptions should appear in the prompt
        assert "sandy blond hair" in user_text
        assert "roguish man" in user_text
        assert "Shot 0" in user_text
        assert "Shot 3" in user_text


class TestStage3InputHandling:
    """SPEC-200: Input handling (REQ-020 to REQ-024)."""

    @pytest.mark.req("SPEC-200/REQ-020")
    def test_reads_v1_format(self, tmp_path, monkeypatch):
        """REQ-020: Stage 3 shall support v1 (flat array) shots.json."""
        data = _mock_stage3(tmp_path, monkeypatch, PROMPTS_V1, MOCK_CHARACTERS)
        assert len(data["characters"]) == 2

    @pytest.mark.req("SPEC-200/REQ-021")
    def test_reads_v2_format(self, tmp_path, monkeypatch):
        """REQ-021: Stage 3 shall support v2 format shots.json."""
        prompts_v2 = {
            "format": "v2",
            "shots": PROMPTS_V1,
            "dialog": [],
        }
        data = _mock_stage3(tmp_path, monkeypatch, prompts_v2, MOCK_CHARACTERS)
        assert len(data["characters"]) == 2

    @pytest.mark.req("SPEC-200/REQ-022")
    def test_missing_prompts_exits(self, tmp_path):
        """REQ-022: Stage 3 shall exit if shots.json does not exist."""
        with pytest.raises(SystemExit):
            run_stage3(argparse.Namespace(output_dir=str(tmp_path)))

    @pytest.mark.req("SPEC-200/REQ-023")
    def test_no_subjects_exits(self, tmp_path, monkeypatch):
        """REQ-023: Stage 3 shall exit if no subjects are found."""
        prompts = [{"index": 0, "description": {"action": "A door opens."}}]
        (tmp_path / "shots.json").write_text(json.dumps(prompts))
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        with pytest.raises(SystemExit):
            run_stage3(argparse.Namespace(output_dir=str(tmp_path)))

    @pytest.mark.req("SPEC-200/REQ-024")
    def test_missing_api_key_exits(self, tmp_path, monkeypatch):
        """REQ-024: Stage 3 shall exit if GEMINI_API_KEY is not set."""
        prompts = [{"index": 0, "description": {"subjects": "Luke"}}]
        (tmp_path / "shots.json").write_text(json.dumps(prompts))
        monkeypatch.setenv("GEMINI_API_KEY", "")

        with pytest.raises(SystemExit):
            run_stage3(argparse.Namespace(output_dir=str(tmp_path)))


# ---------------------------------------------------------------------------
# Shot assignment refinement
# ---------------------------------------------------------------------------


class TestRefineShotAssignments:
    """SPEC-200: Shot assignment refinement (REQ-030 to REQ-032)."""

    def _make_mock_client(self, refinement_response):
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(
            text=json.dumps(refinement_response)
        )
        return mock_client

    @pytest.mark.req("SPEC-200/REQ-030", "SPEC-200/REQ-031")
    def test_assigns_unmatched_shots(self):
        """REQ-030/031: Unassigned shots matching a character shall be added."""
        characters_data = {
            "characters": [
                {"name": "luke", "display_name": "Luke Skywalker",
                 "description": "Young man, blond hair, desert robes.",
                 "shots": [0, 2]},
            ]
        }
        prompts = [
            {"index": 0, "description": {"subjects": "Luke Skywalker"}},
            {"index": 1, "description": {"subjects": "A young man in desert robes"}},
            {"index": 2, "description": {"subjects": "Luke in white tunic"}},
            {"index": 3, "description": {"subjects": "A young man on a ridge"}},
        ]
        subjects_by_shot = [
            "Shot 0: Luke Skywalker",
            "Shot 1: A young man in desert robes",
            "Shot 2: Luke in white tunic",
            "Shot 3: A young man on a ridge",
        ]

        refinement = {
            "assignments": [
                {"shot": 1, "characters": ["luke"]},
                {"shot": 3, "characters": ["luke"]},
            ]
        }
        client = self._make_mock_client(refinement)

        result = _refine_shot_assignments(client, characters_data, prompts, subjects_by_shot)
        luke = result["characters"][0]
        assert 1 in luke["shots"]
        assert 3 in luke["shots"]

    @pytest.mark.req("SPEC-200/REQ-032")
    def test_preserves_existing_assignments(self):
        """REQ-032: Refinement shall not remove existing shot assignments."""
        characters_data = {
            "characters": [
                {"name": "luke", "display_name": "Luke Skywalker",
                 "description": "Young man, blond hair.",
                 "shots": [0, 2, 4]},
            ]
        }
        prompts = [
            {"index": 1, "description": {"subjects": "A young man"}},
        ]
        subjects_by_shot = ["Shot 1: A young man"]

        refinement = {"assignments": [{"shot": 1, "characters": ["luke"]}]}
        client = self._make_mock_client(refinement)

        result = _refine_shot_assignments(client, characters_data, prompts, subjects_by_shot)
        luke = result["characters"][0]
        assert 0 in luke["shots"]
        assert 2 in luke["shots"]
        assert 4 in luke["shots"]
        assert 1 in luke["shots"]

    @pytest.mark.req("SPEC-200/REQ-030")
    def test_no_unassigned_shots_skips_refinement(self):
        """REQ-030: If all shots are assigned, no refinement call is made."""
        characters_data = {
            "characters": [
                {"name": "luke", "display_name": "Luke",
                 "description": "Young man.", "shots": [0, 1]},
            ]
        }
        prompts = [
            {"index": 0, "description": {"subjects": "Luke"}},
            {"index": 1, "description": {"subjects": "Luke again"}},
        ]

        client = MagicMock()
        _refine_shot_assignments(client, characters_data, prompts, [])
        client.models.generate_content.assert_not_called()

    def test_invalid_json_response_skips(self):
        """Gracefully handles bad refinement response."""
        characters_data = {
            "characters": [
                {"name": "luke", "display_name": "Luke",
                 "description": "Young man.", "shots": [0]},
            ]
        }
        prompts = [{"index": 1, "description": {"subjects": "A man"}}]

        client = MagicMock()
        client.models.generate_content.return_value = MagicMock(text="not json")

        result = _refine_shot_assignments(client, characters_data, prompts, ["Shot 1: A man"])
        assert result["characters"][0]["shots"] == [0]  # unchanged


# ---------------------------------------------------------------------------
# TMDB helper functions
# ---------------------------------------------------------------------------


class TestNormalizeCharacterName:
    def test_strips_voice(self):
        assert _normalize_character_name("Darth Vader (voice)") == "Darth Vader"

    def test_strips_uncredited(self):
        assert _normalize_character_name("Jabba the Hutt (uncredited)") == "Jabba the Hutt"

    def test_strips_archive_footage(self):
        assert _normalize_character_name("Obi-Wan Kenobi (archive footage)") == "Obi-Wan Kenobi"

    def test_takes_first_alt_name(self):
        assert _normalize_character_name("Han Solo / Narrator") == "Han Solo"

    def test_no_annotation(self):
        assert _normalize_character_name("Luke Skywalker") == "Luke Skywalker"

    def test_empty_string(self):
        assert _normalize_character_name("") == ""

    def test_strips_and_splits(self):
        assert _normalize_character_name("Yoda (voice) / The Jedi Master") == "Yoda"


class TestToNameKey:
    def test_basic(self):
        assert _to_name_key("Luke Skywalker") == "luke_skywalker"

    def test_special_chars(self):
        assert _to_name_key("Obi-Wan Kenobi") == "obi_wan_kenobi"

    def test_single_word(self):
        assert _to_name_key("Chewbacca") == "chewbacca"

    def test_with_annotation(self):
        # _to_name_key normalizes first
        assert _to_name_key("R2-D2 (voice)") == "r2_d2"

    def test_numbers_preserved(self):
        assert _to_name_key("C-3PO") == "c_3po"


class TestTextMatchCast:
    CAST = [
        {"display_name": "Luke Skywalker", "actor": "Mark Hamill", "name_key": "luke_skywalker"},
        {"display_name": "Han Solo", "actor": "Harrison Ford", "name_key": "han_solo"},
        {"display_name": "Jawas", "actor": "", "name_key": "jawas"},
    ]

    def test_exact_match(self):
        subjects = ["Shot 0: Luke Skywalker in a desert"]
        pre, unmatched = _text_match_cast(self.CAST, subjects)
        assert 0 in pre
        assert pre[0][0]["name_key"] == "luke_skywalker"
        assert unmatched == []

    def test_no_match(self):
        subjects = ["Shot 5: A small hooded creature"]
        pre, unmatched = _text_match_cast(self.CAST, subjects)
        assert pre == {}
        assert len(unmatched) == 1

    def test_word_boundary_no_partial(self):
        """'Han' should not match in 'Hangar'."""
        subjects = ["Shot 2: Wide shot of the Hangar bay"]
        pre, unmatched = _text_match_cast(self.CAST, subjects)
        assert pre == {}
        assert len(unmatched) == 1

    def test_case_insensitive(self):
        subjects = ["Shot 1: han solo shoots first"]
        pre, unmatched = _text_match_cast(self.CAST, subjects)
        assert 1 in pre
        assert pre[1][0]["name_key"] == "han_solo"

    def test_multiple_matches_in_one_shot(self):
        subjects = ["Shot 3: Luke Skywalker and Han Solo face off"]
        pre, unmatched = _text_match_cast(self.CAST, subjects)
        assert 3 in pre
        assert len(pre[3]) == 2
        keys = {e["name_key"] for e in pre[3]}
        assert keys == {"luke_skywalker", "han_solo"}

    def test_mixed_matched_and_unmatched(self):
        subjects = [
            "Shot 0: Luke Skywalker looking at twin suns",
            "Shot 1: A young man stares into the distance",
        ]
        pre, unmatched = _text_match_cast(self.CAST, subjects)
        assert 0 in pre
        assert len(unmatched) == 1
        assert "Shot 1" in unmatched[0]


# ---------------------------------------------------------------------------
# Supervised stage 3
# ---------------------------------------------------------------------------


def _make_supervised_mock_client(gemini_response):
    """Build a mock Gemini client that returns the given response for all calls."""
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = MagicMock(
        text=json.dumps(gemini_response)
    )
    return mock_client


CAST_ENTRIES = [
    {"display_name": "Luke Skywalker", "actor": "Mark Hamill", "name_key": "luke_skywalker"},
    {"display_name": "Jawas", "actor": "", "name_key": "jawas"},
]

DROID_CAST_ENTRIES = [
    {"display_name": "C-3PO", "actor": "Anthony Daniels", "name_key": "c_3po"},
]

DROID_PROMPTS = [
    {"index": 0, "description": {"subjects": "C-3PO, a tall gold protocol droid"}},
    {"index": 1, "description": {"subjects": "A tall golden humanoid droid gesturing"}},
    {"index": 2, "description": {"subjects": "C-3PO walking through the desert"}},
    {"index": 3, "description": {"subjects": "A polished gold droid with a visor"}},
]

PROMPTS_WITH_SUBJECTS = [
    {"index": 0, "description": {"subjects": "Luke Skywalker, young man in white tunic"}},
    {"index": 1, "description": {"subjects": "Small hooded creatures scavenging droids"}},
    {"index": 2, "description": {"subjects": "Luke Skywalker at moisture vaporator"}},
    {"index": 3, "description": {"subjects": "Small hooded figures in desert canyon"}},
]


class TestCharacterDedupeKey:
    def test_collapses_separator_variants(self):
        assert _character_dedupe_key("c_3po") == _character_dedupe_key("c3po")
        assert _character_dedupe_key("C-3PO") == _character_dedupe_key("c3po")
        assert _character_dedupe_key("r2_d2") == _character_dedupe_key("r2d2")

    def test_keeps_distinct_characters_apart(self):
        assert _character_dedupe_key("luke_skywalker") != _character_dedupe_key("han_solo")
        # a real near-miss: don't collapse two different Red squadron pilots
        assert _character_dedupe_key("red_leader") != _character_dedupe_key("red_two")

    def test_handles_empty_and_none(self):
        assert _character_dedupe_key("") == ""
        assert _character_dedupe_key(None) == ""


class TestCollapseDuplicateCharacters:
    def test_merges_separator_variant_ids(self):
        """The Star Wars IV regression: c3po and c_3po were two registry entries
        for one droid, with disjoint shots and separate locked descriptions."""
        collapsed = _collapse_duplicate_characters([
            {"name": "c3po", "display_name": "C-3PO", "description": "short", "shots": [1, 3]},
            {"name": "c_3po", "display_name": "C-3PO",
             "description": "a much longer canonical appearance description", "shots": [2, 4]},
        ])
        assert len(collapsed) == 1
        entry = collapsed[0]
        # canonical id comes from display_name via _to_name_key
        assert entry["name"] == "c_3po"
        assert entry["shots"] == [1, 2, 3, 4], "shot lists must union, not overwrite"
        assert entry["description"] == "a much longer canonical appearance description"

    def test_leaves_distinct_characters_alone(self):
        collapsed = _collapse_duplicate_characters([
            {"name": "red_leader", "display_name": "Red Leader", "description": "a", "shots": [1]},
            {"name": "red_two", "display_name": "Red Two", "description": "b", "shots": [2]},
        ])
        assert len(collapsed) == 2

    def test_preserves_order(self):
        collapsed = _collapse_duplicate_characters([
            {"name": "luke_skywalker", "display_name": "Luke Skywalker", "description": "a", "shots": [1]},
            {"name": "han_solo", "display_name": "Han Solo", "description": "b", "shots": [2]},
        ])
        assert [c["name"] for c in collapsed] == ["luke_skywalker", "han_solo"]


class TestRunSupervisedStage3:
    def test_model_id_variant_does_not_create_duplicate(self):
        """Regression: the TMDB path derives `c_3po` from the credited name while
        the model returns `c3po` for the same droid. An exact-string compare in
        the pre-assignment merge appended a duplicate stub, so the droid ended up
        with two locked descriptions and two voices."""
        gemini_response = {
            "characters": [
                {
                    "name": "c3po",  # model's own id — differs from _to_name_key("C-3PO")
                    "display_name": "C-3PO",
                    "description": "A tall humanoid droid with a polished gold exterior.",
                    "shots": [1, 3],
                }
            ]
        }
        client = _make_supervised_mock_client(gemini_response)
        subjects = [
            "Shot 0: C-3PO, a tall gold protocol droid",
            "Shot 1: A tall golden humanoid droid gesturing",
            "Shot 2: C-3PO walking through the desert",
            "Shot 3: A polished gold droid with a visor",
        ]
        result = _run_supervised_stage3(client, DROID_CAST_ENTRIES, subjects, DROID_PROMPTS)

        droids = [c for c in result["characters"]
                  if _character_dedupe_key(c["display_name"]) == "c3po"]
        assert len(droids) == 1, f"expected one C-3PO entry, got {[d['name'] for d in droids]}"
        # text-matched shots 0 and 2 merge with the model's 1 and 3
        assert sorted(droids[0]["shots"]) == [0, 1, 2, 3]

    def test_gemini_results_merged_with_text_match(self):
        """Shots 0 and 2 are text-matched (Luke); shots 1 and 3 go to Gemini (Jawas)."""
        gemini_response = {
            "characters": [
                {
                    "name": "jawas",
                    "display_name": "Jawas",
                    "description": "Small hooded creatures in brown robes with glowing eyes.",
                    "shots": [1, 3],
                }
            ]
        }
        client = _make_supervised_mock_client(gemini_response)
        subjects = [
            "Shot 0: Luke Skywalker, young man in white tunic",
            "Shot 1: Small hooded creatures scavenging droids",
            "Shot 2: Luke Skywalker at moisture vaporator",
            "Shot 3: Small hooded figures in desert canyon",
        ]
        result = _run_supervised_stage3(client, CAST_ENTRIES, subjects, PROMPTS_WITH_SUBJECTS)
        chars = {c["name"]: c for c in result["characters"]}

        # Jawas found by Gemini
        assert "jawas" in chars
        assert sorted(chars["jawas"]["shots"]) == [1, 3]

        # Luke found by text match and merged in
        assert "luke_skywalker" in chars
        assert sorted(chars["luke_skywalker"]["shots"]) == [0, 2]

    def test_min_two_shots_filter(self):
        """Characters with only 1 shot after merge should be excluded."""
        gemini_response = {
            "characters": [
                {
                    "name": "jawas",
                    "display_name": "Jawas",
                    "description": "Small creatures.",
                    "shots": [1],  # only 1 shot from Gemini; no text-match supplement
                }
            ]
        }
        client = _make_supervised_mock_client(gemini_response)
        subjects = ["Shot 1: Small hooded creatures"]
        prompts = [{"index": 1, "description": {"subjects": "Small hooded creatures"}}]
        result = _run_supervised_stage3(client, CAST_ENTRIES, subjects, prompts)
        names = [c["name"] for c in result["characters"]]
        assert "jawas" not in names

    def test_stub_description_filled(self):
        """Text-match-only characters (no Gemini match) get descriptions generated."""
        # All shots are text-matched (0 unmatched), so no batch call occurs.
        # The only Gemini call is the description-generation call for Luke.
        desc_response = "Young man in his early twenties with sandy blond hair."
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(text=desc_response)
        subjects = [
            "Shot 0: Luke Skywalker in white tunic",
            "Shot 2: Luke Skywalker at vaporator",
        ]
        prompts = [
            {"index": 0, "description": {"subjects": "Luke Skywalker in white tunic"}},
            {"index": 2, "description": {"subjects": "Luke Skywalker at vaporator"}},
        ]
        result = _run_supervised_stage3(mock_client, CAST_ENTRIES, subjects, prompts)
        chars = {c["name"]: c for c in result["characters"]}
        assert "luke_skywalker" in chars
        assert chars["luke_skywalker"]["description"] == desc_response

    def test_batches_large_unmatched_set(self):
        """More than 200 unmatched shots triggers multiple Gemini batch calls."""
        # 250 unmatched shots (none text-matched)
        subjects = [f"Shot {i}: Generic subject" for i in range(250)]
        prompts = [{"index": i, "description": {"subjects": "Generic subject"}} for i in range(250)]

        mock_client = MagicMock()
        # First batch: 200 shots → 1 char; second batch: 50 shots → same char
        batch1 = {"characters": [{"name": "jawas", "display_name": "Jawas",
                                   "description": "Small creatures.", "shots": list(range(200))}]}
        batch2 = {"characters": [{"name": "jawas", "display_name": "Jawas",
                                   "description": "Small hooded creatures.", "shots": list(range(200, 250))}]}
        mock_client.models.generate_content.side_effect = [
            MagicMock(text=json.dumps(batch1)),
            MagicMock(text=json.dumps(batch2)),
        ]

        result = _run_supervised_stage3(mock_client, CAST_ENTRIES, subjects, prompts)
        assert mock_client.models.generate_content.call_count == 2
        chars = {c["name"]: c for c in result["characters"]}
        assert "jawas" in chars
        assert len(chars["jawas"]["shots"]) == 250

    def test_invalid_batch_json_skipped(self):
        """A batch returning invalid JSON is skipped without crashing."""
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(text="not json")
        subjects = ["Shot 1: Generic subject", "Shot 2: Generic subject"]
        prompts = [{"index": 1, "description": {"subjects": "s"}},
                   {"index": 2, "description": {"subjects": "s"}}]
        result = _run_supervised_stage3(mock_client, CAST_ENTRIES, subjects, prompts)
        # No crash; empty registry (no chars met 2-shot threshold)
        assert result == {"characters": []}


# ---------------------------------------------------------------------------
# run_stage3 with --tmdb-id
# ---------------------------------------------------------------------------


class TestRunStage3WithTmdb:
    def _write_prompts(self, tmp_path, prompts_data=None):
        if prompts_data is None:
            prompts_data = PROMPTS_WITH_SUBJECTS
        (tmp_path / "shots.json").write_text(json.dumps(prompts_data))

    def test_supervised_path_invoked_when_tmdb_id_set(self, tmp_path, monkeypatch):
        """When --tmdb-id and TMDB_API_KEY are set, the supervised path is used."""
        self._write_prompts(tmp_path)

        gemini_response = {
            "characters": [
                {"name": "jawas", "display_name": "Jawas",
                 "description": "Small creatures.", "shots": [1, 3]},
            ]
        }
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(
            text=json.dumps(gemini_response)
        )
        monkeypatch.setenv("GEMINI_API_KEY", "test-gemini")
        monkeypatch.setenv("TMDB_API_KEY", "test-tmdb")
        monkeypatch.setattr("google.genai.Client", lambda **kwargs: mock_client)

        tmdb_cast = [
            {"display_name": "Luke Skywalker", "actor": "Mark Hamill", "name_key": "luke_skywalker"},
            {"display_name": "Jawas", "actor": "", "name_key": "jawas"},
        ]
        monkeypatch.setattr("encode.fetch_tmdb_cast", lambda *a, **kw: tmdb_cast)

        args = argparse.Namespace(
            output_dir=str(tmp_path), tmdb_id="11", tmdb_type="movie"
        )
        run_stage3(args)

        data = json.loads((tmp_path / "characters.json").read_text())
        names = {c["name"] for c in data["characters"]}
        # Luke text-matched; Jawas from Gemini
        assert "luke_skywalker" in names or "jawas" in names

    def test_missing_tmdb_api_key_falls_back(self, tmp_path, monkeypatch, capsys):
        """When TMDB_API_KEY is absent, falls back to unsupervised mode."""
        self._write_prompts(tmp_path)

        unsupervised_response = {
            "characters": [
                {"name": "luke", "display_name": "Luke Skywalker",
                 "description": "Young man.", "shots": [0, 2]},
            ]
        }
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(
            text=json.dumps(unsupervised_response)
        )
        monkeypatch.setenv("GEMINI_API_KEY", "test-gemini")
        monkeypatch.delenv("TMDB_API_KEY", raising=False)
        monkeypatch.setattr("google.genai.Client", lambda **kwargs: mock_client)

        args = argparse.Namespace(
            output_dir=str(tmp_path), tmdb_id="11", tmdb_type="movie"
        )
        run_stage3(args)

        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "unsupervised" in captured.out.lower()

    def test_tmdb_fetch_failure_falls_back(self, tmp_path, monkeypatch, capsys):
        """When TMDB fetch raises an exception, falls back to unsupervised mode."""
        self._write_prompts(tmp_path)

        unsupervised_response = {
            "characters": [
                {"name": "luke", "display_name": "Luke Skywalker",
                 "description": "Young man.", "shots": [0, 2]},
            ]
        }
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(
            text=json.dumps(unsupervised_response)
        )
        monkeypatch.setenv("GEMINI_API_KEY", "test-gemini")
        monkeypatch.setenv("TMDB_API_KEY", "bad-key")
        monkeypatch.setattr("google.genai.Client", lambda **kwargs: mock_client)
        monkeypatch.setattr("encode.fetch_tmdb_cast", lambda *a, **kw: (_ for _ in ()).throw(
            Exception("HTTP 401")
        ))

        args = argparse.Namespace(
            output_dir=str(tmp_path), tmdb_id="11", tmdb_type="movie"
        )
        run_stage3(args)

        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "falling back" in captured.out.lower()

    def test_no_tmdb_id_uses_unsupervised(self, tmp_path, monkeypatch):
        """Without --tmdb-id, the original unsupervised path runs unchanged."""
        self._write_prompts(tmp_path)

        unsupervised_response = {
            "characters": [
                {"name": "luke", "display_name": "Luke Skywalker",
                 "description": "Young man.", "shots": [0, 2]},
            ]
        }
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(
            text=json.dumps(unsupervised_response)
        )
        monkeypatch.setenv("GEMINI_API_KEY", "test-gemini")
        monkeypatch.setattr("google.genai.Client", lambda **kwargs: mock_client)

        args = argparse.Namespace(output_dir=str(tmp_path), tmdb_id=None, tmdb_type="movie")
        run_stage3(args)

        data = json.loads((tmp_path / "characters.json").read_text())
        assert data["characters"][0]["name"] == "luke"


# ---------------------------------------------------------------------------
# generate_temporal_segments
# ---------------------------------------------------------------------------

class TestGenerateTemporalSegments:
    """Tests for per-segment description generation for long shots (SPEC-300, REQ-001 to REQ-008)."""

    def _make_scene(self, duration_s=19.2):
        return {"duration_s": duration_s, "start_s": 0.0, "end_s": duration_s}

    def _make_mock_client(self, responses):
        """Create a mock Gemini client that returns responses in order."""
        from unittest.mock import MagicMock
        client = MagicMock()
        client.models.generate_content.side_effect = [
            MagicMock(text=json.dumps(r)) for r in responses
        ]
        return client

    @pytest.mark.req("SPEC-300/REQ-002")
    def test_returns_n_segment_descriptions(self, tmp_path):
        """REQ-002: Returns exactly n_segments description dicts for a shot with >= 2 keyframes."""
        for name in ["0001-01.jpg", "0001-02.jpg", "0001-03.jpg", "0001-04.jpg"]:
            (tmp_path / name).write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        descs = [{"action": f"Segment {i} action.", "shot_type": "wide"} for i in range(3)]
        client = self._make_mock_client(descs)

        frame_files = ["0001-01.jpg", "0001-02.jpg", "0001-03.jpg", "0001-04.jpg"]
        result = generate_temporal_segments(
            self._make_scene(), {"action": "Whole shot."}, frame_files, str(tmp_path), client,
            n_segments=3,
        )

        assert len(result) == 3
        assert result[0]["action"] == "Segment 0 action."
        assert result[2]["action"] == "Segment 2 action."

    @pytest.mark.req("SPEC-300/REQ-002")
    def test_defaults_to_two_segments(self, tmp_path):
        """REQ-002: Default n_segments=2 returns exactly 2 descriptions."""
        for name in ["0001-01.jpg", "0001-02.jpg"]:
            (tmp_path / name).write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        descs = [{"action": "First."}, {"action": "Second."}]
        client = self._make_mock_client(descs)

        result = generate_temporal_segments(
            self._make_scene(), {"action": "Whole."}, ["0001-01.jpg", "0001-02.jpg"], str(tmp_path), client
        )

        assert len(result) == 2

    @pytest.mark.req("SPEC-300/REQ-006")
    def test_returns_empty_for_single_keyframe(self, tmp_path):
        """REQ-006: Returns [] when only 1 keyframe — can't split into two halves."""
        (tmp_path / "0001-01.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
        from unittest.mock import MagicMock
        client = MagicMock()

        result = generate_temporal_segments(
            self._make_scene(), {"action": "Test."}, ["0001-01.jpg"], str(tmp_path), client
        )
        assert result == []
        client.models.generate_content.assert_not_called()

    @pytest.mark.req("SPEC-300/REQ-006")
    def test_returns_empty_for_no_keyframes(self, tmp_path):
        """REQ-006: Returns [] when frame_files is empty."""
        from unittest.mock import MagicMock
        client = MagicMock()

        result = generate_temporal_segments(
            self._make_scene(), {"action": "Test."}, [], str(tmp_path), client
        )
        assert result == []
        client.models.generate_content.assert_not_called()

    @pytest.mark.req("SPEC-300/REQ-007")
    def test_returns_empty_on_gemini_failure(self, tmp_path):
        """REQ-007: Returns [] when a Gemini call raises — no partial segments stored."""
        from unittest.mock import MagicMock
        for name in ["0001-01.jpg", "0001-02.jpg"]:
            (tmp_path / name).write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
        client = MagicMock()
        client.models.generate_content.side_effect = Exception("API error")

        result = generate_temporal_segments(
            self._make_scene(), {"action": "Test."}, ["0001-01.jpg", "0001-02.jpg"],
            str(tmp_path), client,
        )
        assert result == []

    @pytest.mark.req("SPEC-300/REQ-003", "SPEC-300/REQ-004")
    def test_frames_split_into_n_groups(self, tmp_path):
        """REQ-003/REQ-004: One Gemini call made per segment, frames divided evenly."""
        frames = [f"0001-0{i}.jpg" for i in range(1, 9)]  # 8 frames
        for name in frames:
            (tmp_path / name).write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        seg_desc = {"action": "Desc."}
        client = self._make_mock_client([seg_desc] * 5)

        generate_temporal_segments(
            self._make_scene(), {"action": "Whole."}, frames, str(tmp_path), client,
            n_segments=5,
        )

        assert client.models.generate_content.call_count == 5

    @pytest.mark.req("SPEC-300/REQ-002")
    def test_n_segments_capped_at_keyframe_count(self, tmp_path):
        """REQ-002: n_segments capped at available keyframes — 3 frames with n_segments=5 gives 3."""
        frames = ["0001-01.jpg", "0001-02.jpg", "0001-03.jpg"]
        for name in frames:
            (tmp_path / name).write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        descs = [{"action": f"Seg {i}."} for i in range(3)]
        client = self._make_mock_client(descs)

        result = generate_temporal_segments(
            self._make_scene(), {"action": "Whole."}, frames, str(tmp_path), client,
            n_segments=5,
        )

        assert len(result) == 3
        assert client.models.generate_content.call_count == 3


# ---------------------------------------------------------------------------
# generate_prompts — resume via the manifest contract
# ---------------------------------------------------------------------------

class TestGeneratePromptsResume:
    """generate_prompts must resume/save through manifest.load_shots/save_shots
    (v2), not hand-rolled JSON — a v1-shaped resume crashes on a v2 shots.json.
    """

    def _make_scene(self, idx):
        return {
            "index": idx, "start_s": float(idx), "end_s": float(idx) + 1.0,
            "duration_s": 1.0, "keyframes": [f"{idx:04d}-01.jpg"],
        }

    def _write_keyframe(self, tmp_path, idx):
        keyframes_dir = tmp_path / "keyframes"
        keyframes_dir.mkdir(exist_ok=True)
        (keyframes_dir / f"{idx:04d}-01.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

    def test_resumes_from_v2_shots_json_without_crashing(self, tmp_path, monkeypatch):
        """A v2 shots.json (the only shape stage2 ever writes) must resume cleanly.

        Regression test: the old resume path did `json.load` then
        `{p["index"] for p in existing}`, which raises TypeError against the
        v2 dict shape — re-running stage2 on a completed/interrupted dir
        always crashed.
        """
        existing_dialog = [{"text": "hello", "start_s": 0.0, "end_s": 0.5}]
        manifest.save_shots(
            str(tmp_path),
            [{"index": 0, "start_s": 0.0, "end_s": 1.0, "duration_s": 1.0,
              "camera_motion_detected": "static", "audio_detected": None,
              "dialogue": None, "description": {"action": "Shot 0."}}],
            existing_dialog,
        )

        self._write_keyframe(tmp_path, 1)
        scenes = [self._make_scene(0), self._make_scene(1)]

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(
            text=json.dumps({"action": "Shot 1."}), usage_metadata=None,
        )
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setattr("google.genai.Client", lambda **kwargs: mock_client)

        prompts = generate_prompts(
            scenes, {}, {}, {}, str(tmp_path), "gemini", dialog=existing_dialog,
        )

        # Shot 0 was skipped (already in shots.json); only shot 1 hit Gemini.
        assert mock_client.models.generate_content.call_count == 1
        assert {p["index"] for p in prompts} == {0, 1}

    def test_incremental_and_final_save_round_trip_via_manifest(self, tmp_path, monkeypatch):
        """An interrupted stage2 leaves a v2 shots.json that manifest.load_shots accepts."""
        for idx in range(11):
            self._write_keyframe(tmp_path, idx)
        scenes = [self._make_scene(idx) for idx in range(11)]

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(
            text=json.dumps({"action": "Shot."}), usage_metadata=None,
        )
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setattr("google.genai.Client", lambda **kwargs: mock_client)

        dialog = [{"text": "line", "start_s": 0.0, "end_s": 0.5}]
        prompts = generate_prompts(
            scenes, {}, {}, {}, str(tmp_path), "gemini", dialog=dialog,
        )
        manifest.save_shots(str(tmp_path), prompts, dialog)

        # The incremental save at shot 10 (11th shot) must already be v2-shaped.
        shots, loaded_dialog = manifest.load_shots(str(tmp_path))
        assert {s["index"] for s in shots} == set(range(11))
        assert loaded_dialog == dialog

        # Interrupt-and-resume: a fresh call against the same output_dir must
        # not crash and must not re-describe already-completed shots.
        mock_client.models.generate_content.reset_mock()
        resumed = generate_prompts(
            scenes, {}, {}, {}, str(tmp_path), "gemini", dialog=dialog,
        )
        assert mock_client.models.generate_content.call_count == 0
        assert {p["index"] for p in resumed} == set(range(11))


class TestGeneratePromptsPreviousShotContext:
    """Each shot's Gemini call should be given the previous shot's setting/
    lighting continuity fields (but NOT its subjects/action, where Gemini's
    world-knowledge bias hallucinates character identity), and a resumed run
    must seed that context from the last already-described shot rather than
    starting cold.
    """

    def _make_scene(self, idx):
        return {
            "index": idx, "start_s": float(idx), "end_s": float(idx) + 1.0,
            "duration_s": 1.0, "keyframes": [f"{idx:04d}-01.jpg"],
        }

    def _write_keyframe(self, tmp_path, idx):
        keyframes_dir = tmp_path / "keyframes"
        keyframes_dir.mkdir(exist_ok=True)
        (keyframes_dir / f"{idx:04d}-01.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

    def _user_text(self, call_args) -> str:
        """Extract the joined text part from a generate_content call's contents."""
        contents = call_args.kwargs["contents"]
        parts = contents[0].parts
        return parts[-1].text

    def test_second_shot_receives_first_shots_setting_but_not_its_subjects(self, tmp_path, monkeypatch):
        for idx in (0, 1):
            self._write_keyframe(tmp_path, idx)
        scenes = [self._make_scene(0), self._make_scene(1)]

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            MagicMock(text=json.dumps({
                "action": "Luke looks at the twin suns.",
                "subjects": "Luke Skywalker, a young man in a white tunic",
                "setting": "Tatooine, a desert homestead at dusk",
                "lighting": "warm orange backlight from the setting suns",
            }), usage_metadata=None),
            MagicMock(text=json.dumps({"action": "Luke walks inside."}), usage_metadata=None),
        ]
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setattr("google.genai.Client", lambda **kwargs: mock_client)

        generate_prompts(scenes, {}, {}, {}, str(tmp_path), "gemini")

        assert mock_client.models.generate_content.call_count == 2
        first_call, second_call = mock_client.models.generate_content.call_args_list
        second_text = self._user_text(second_call)
        # Continuity fields (setting/lighting) should carry forward...
        assert "Tatooine, a desert homestead at dusk" in second_text
        assert "warm orange backlight from the setting suns" in second_text
        # ...but identity-bearing fields (subjects/action) must not.
        assert "Luke looks at the twin suns." not in second_text
        assert "Luke Skywalker, a young man in a white tunic" not in second_text
        assert "Luke looks at the twin suns." not in self._user_text(first_call)

    def test_resumed_run_seeds_context_from_last_existing_shot(self, tmp_path, monkeypatch):
        manifest.save_shots(
            str(tmp_path),
            [{"index": 0, "start_s": 0.0, "end_s": 1.0, "duration_s": 1.0,
              "camera_motion_detected": "static", "audio_detected": None,
              "dialogue": None,
              "description": {
                  "action": "R2-D2 beeps in the corridor.",
                  "subjects": "R2-D2, a blue and white droid",
                  "setting": "a dim starship corridor",
              }}],
            [],
        )
        self._write_keyframe(tmp_path, 1)
        scenes = [self._make_scene(0), self._make_scene(1)]

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(
            text=json.dumps({"action": "Leia enters the frame."}), usage_metadata=None,
        )
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setattr("google.genai.Client", lambda **kwargs: mock_client)

        generate_prompts(scenes, {}, {}, {}, str(tmp_path), "gemini")

        assert mock_client.models.generate_content.call_count == 1
        call = mock_client.models.generate_content.call_args_list[0]
        user_text = self._user_text(call)
        assert "a dim starship corridor" in user_text
        assert "R2-D2 beeps in the corridor." not in user_text
        assert "R2-D2, a blue and white droid" not in user_text


# ---------------------------------------------------------------------------
# Stage 4: speaker attribution
# ---------------------------------------------------------------------------


CHARACTERS = [
    {"name": "luke", "display_name": "Luke Skywalker",
     "description": "Young man, blond hair.", "shots": [0, 1]},
    {"name": "leia", "display_name": "Princess Leia",
     "description": "Young woman, hologram.", "shots": [2]},
]

SHOTS = [
    {"index": 0, "start_s": 0.0, "end_s": 5.0, "duration_s": 5.0},
    {"index": 1, "start_s": 5.0, "end_s": 10.0, "duration_s": 5.0},
    {"index": 2, "start_s": 10.0, "end_s": 15.0, "duration_s": 5.0},
]


class TestBuildShotCharacterMap:
    def test_reverses_shots_lists(self):
        shot_map = _build_shot_character_map(CHARACTERS)
        assert shot_map[0] == ["luke"]
        assert shot_map[1] == ["luke"]
        assert shot_map[2] == ["leia"]

    def test_empty_characters(self):
        assert _build_shot_character_map([]) == {}


class TestCandidateCharactersForLine:
    def test_line_within_one_shot(self):
        shot_map = _build_shot_character_map(CHARACTERS)
        line = {"start_s": 1.0, "end_s": 2.0}
        assert _candidate_characters_for_line(line, SHOTS, shot_map) == ["luke"]

    def test_line_spanning_two_shots_unions_candidates(self):
        shot_map = _build_shot_character_map(CHARACTERS)
        line = {"start_s": 9.0, "end_s": 11.0}
        assert _candidate_characters_for_line(line, SHOTS, shot_map) == ["luke", "leia"]

    def test_no_overlapping_shot_returns_empty(self):
        shot_map = _build_shot_character_map(CHARACTERS)
        line = {"start_s": 100.0, "end_s": 101.0}
        assert _candidate_characters_for_line(line, SHOTS, shot_map) == []


class TestRunStage4:
    def _write_inputs(self, tmp_path):
        manifest.save_shots(
            str(tmp_path),
            SHOTS,
            [
                {"text": "Help me.", "start_s": 1.0, "end_s": 2.0},
                {"text": "You're my only hope.", "start_s": 11.0, "end_s": 12.0},
            ],
        )
        (tmp_path / "characters.json").write_text(
            json.dumps({"characters": CHARACTERS})
        )

    def test_writes_speakers_json_gated_to_registry(self, tmp_path, monkeypatch):
        self._write_inputs(tmp_path)
        gemini_response = {
            "assignments": [
                {"line": 0, "character": "luke", "confidence": 0.9},
                {"line": 1, "character": "leia", "confidence": 0.8},
            ]
        }
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(
            text=json.dumps(gemini_response)
        )
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setattr("google.genai.Client", lambda **kwargs: mock_client)

        args = argparse.Namespace(output_dir=str(tmp_path))
        run_stage4(args)

        speakers = manifest.load_speakers(str(tmp_path))
        assert speakers[0] == "luke"
        assert speakers[1] == "leia"

    def test_unregistered_name_falls_back_to_narrator(self, tmp_path, monkeypatch):
        self._write_inputs(tmp_path)
        gemini_response = {
            "assignments": [
                {"line": 0, "character": "obi_wan", "confidence": 0.5},
                {"line": 1, "character": "leia", "confidence": 0.9},
            ]
        }
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(
            text=json.dumps(gemini_response)
        )
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setattr("google.genai.Client", lambda **kwargs: mock_client)

        args = argparse.Namespace(output_dir=str(tmp_path))
        run_stage4(args)

        speakers = manifest.load_speakers(str(tmp_path))
        assert speakers[0] == "narrator"
        assert speakers[1] == "leia"

    def test_missing_characters_json_exits(self, tmp_path, monkeypatch, capsys):
        manifest.save_shots(str(tmp_path), SHOTS, [{"text": "hi", "start_s": 0.0, "end_s": 1.0}])
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        args = argparse.Namespace(output_dir=str(tmp_path))
        with pytest.raises(SystemExit):
            run_stage4(args)

class TestGeneratePromptsDoesNotPropagateHallucinatedIdentity:
    """Regression test for the sw_r2_leia shots 6-7 incident: Gemini mislabelled
    the Leia hologram as 'Obi-Wan Kenobi' on shot 6 (world-knowledge bias), and
    because the full description was forwarded as context, shot 7 echoed the
    same wrong name and came out byte-identical to shot 6. Character identity
    must never be part of the propagated context.
    """

    def _make_scene(self, idx):
        return {
            "index": idx, "start_s": float(idx), "end_s": float(idx) + 1.0,
            "duration_s": 1.0, "keyframes": [f"{idx:04d}-01.jpg"],
        }

    def _write_keyframe(self, tmp_path, idx):
        keyframes_dir = tmp_path / "keyframes"
        keyframes_dir.mkdir(exist_ok=True)
        (keyframes_dir / f"{idx:04d}-01.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

    def _user_text(self, call_args) -> str:
        contents = call_args.kwargs["contents"]
        parts = contents[0].parts
        return parts[-1].text

    def test_hallucinated_name_in_shot_6_does_not_reach_shot_7_prompt(self, tmp_path, monkeypatch):
        for idx in (6, 7):
            self._write_keyframe(tmp_path, idx)
        scenes = [self._make_scene(6), self._make_scene(7)]

        shot6_description = {
            "shot_type": "medium wide",
            "camera_movement": "static",
            "subjects": "A blue, translucent holographic figure of Obi-Wan Kenobi, wearing a hooded robe",
            "action": "The holographic figure flickers and shifts slightly.",
            "lighting": "dim, with the primary light source being the hologram itself",
            "setting": "a dark chamber aboard a starship",
            "color_palette": "blue and black",
        }
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            MagicMock(text=json.dumps(shot6_description), usage_metadata=None),
            MagicMock(text=json.dumps({
                "subjects": "A blue, translucent holographic figure of a woman in a hooded robe",
                "action": "She raises her hand slightly.",
            }), usage_metadata=None),
        ]
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setattr("google.genai.Client", lambda **kwargs: mock_client)

        generate_prompts(scenes, {}, {}, {}, str(tmp_path), "gemini")

        assert mock_client.models.generate_content.call_count == 2
        _, second_call = mock_client.models.generate_content.call_args_list
        second_text = self._user_text(second_call)

        assert "Obi-Wan Kenobi" not in second_text
        # Continuity (setting/lighting) still passed through.
        assert "a dark chamber aboard a starship" in second_text
        assert "dim, with the primary light source being the hologram itself" in second_text


class TestCanonicalDescriptionsExcludeWardrobe:
    """Canonical character descriptions must carry identity, not costume.

    Stage 3 derives one description per character from all their shots and the
    decoder prepends it verbatim to every shot they appear in. When it named a
    costume, that costume followed the character through the whole film: Luke's
    description said "typically seen in an orange flight suit... yellow
    goggles" (derived from the trench run), so he wore X-wing pilot gear on
    Tatooine and the Death Star -- wrong in 334 of his 572 shots.

    Wardrobe belongs in the per-shot description, which is already accurate.
    """

    def _prompts(self):
        import encode
        import inspect
        src = inspect.getsource(encode)
        return src

    def test_stage3_prompts_forbid_costume_for_humans(self):
        # The rule used to be copy-pasted into all three prompts and was
        # checked by counting occurrences. It now lives in one shared spec,
        # so the guarantee is structural rather than a headcount.
        import encode

        assert "Do NOT describe clothing or costume" in encode.IDENTITY_DESCRIPTION_SPEC
        assert encode.IDENTITY_DESCRIPTION_SPEC in encode.STAGE3_SYSTEM_PROMPT

    def test_stage3_prompts_keep_the_costume_is_identity_exception(self):
        # droids, masked and armoured figures never change, and stripping
        # their shell would strip their identity (C-3PO becomes a man)
        import encode

        assert "costume or shell IS the character" in encode.IDENTITY_DESCRIPTION_SPEC

    def test_stage3_prompts_still_ask_for_stable_identity_traits(self):
        src = self._prompts()
        for trait in ("age range", "hair colour", "bearing"):
            assert trait in src or trait.replace("colour", "color") in src


class TestGeminiClientHasADeadline:
    """Every Gemini call must be able to give up.

    The client was built with no http_options, so generate_content had no
    timeout. Re-encoding the film hung on shot ~400 and sat there 21 hours:
    process alive, 0% CPU, nothing written, no error -- indistinguishable
    from working. It costs wall-clock rather than money, so nothing else
    catches it either.
    """

    def test_the_client_is_built_with_a_timeout(self):
        import encode

        client = encode._gemini_client("dummy-key")
        assert client._api_client._http_options.timeout == encode.GEMINI_TIMEOUT_MS

    def test_the_timeout_is_a_hang_detector_not_a_latency_budget(self):
        import encode

        # Generous enough that a slow-but-working call is never killed, short
        # enough that a hang cannot eat a night.
        assert 60_000 <= encode.GEMINI_TIMEOUT_MS <= 600_000

    def test_no_bare_client_construction_survives(self):
        """A bare genai.Client() would silently reintroduce the hang."""
        import inspect
        import encode

        source = inspect.getsource(encode)
        assert "genai.Client(api_key=api_key)" not in source


class TestPoisonedKeyframeRecovery:
    """One bad frame must not cost the whole shot.

    Shot 451 of Star Wars IV -- the Tusken Raider standing over Luke -- came
    back empty on every encode this project has run, and its absence is the
    cause of the known "index diverges from list position from 451 onward"
    defect. The block is BlockedReason.OTHER, which relaxing safety_settings
    does not lift. But only the FIRST of its four keyframes trips it; the
    other three each describe fine alone.
    """

    class _Part:
        def __init__(self, inline_data=None):
            self.inline_data = inline_data

    class _Resp:
        def __init__(self, text):
            self.text = text

    def _content(self, n_images=4):
        parts = [self._Part(inline_data=f"img{i}") for i in range(n_images)]
        return parts + [self._Part()]  # trailing text part

    def _client(self, succeed_when):
        """A client that returns text only when succeed_when(parts) is true."""
        outer = self

        class Models:
            calls = []

            def generate_content(self, model, contents, config):
                parts = contents[0].parts
                Models.calls.append(parts)
                return outer._Resp("{}" if succeed_when(parts) else "")

        class Client:
            models = Models()

        return Client()

    def _types(self):
        import types as _t

        class Content:
            def __init__(self, role, parts):
                self.role, self.parts = role, parts

        ns = _t.SimpleNamespace(Content=Content)
        return ns

    def test_the_shot_is_recovered_by_dropping_the_bad_frame(self):
        import encode

        content = self._content()
        bad = content[0]
        client = self._client(lambda parts: bad not in parts)

        resp = encode._retry_without_poisoned_frame(
            client, self._types(), content, config=None, idx=451)

        assert resp is not None and resp.text == "{}"

    def test_the_largest_subset_is_tried_first(self):
        """Exactly one frame is dropped, so the shot keeps maximum coverage."""
        import encode

        content = self._content()
        bad = content[0]
        client = self._client(lambda parts: bad not in parts)

        encode._retry_without_poisoned_frame(
            client, self._types(), content, config=None, idx=451)

        first_attempt = client.models.calls[0]
        images = [p for p in first_attempt if p.inline_data is not None]
        assert len(images) == 3  # 4 - exactly one

    def test_a_shot_blocked_on_every_frame_still_gives_up(self):
        import encode

        client = self._client(lambda parts: False)
        assert encode._retry_without_poisoned_frame(
            client, self._types(), self._content(), config=None, idx=451) is None

    def test_a_single_frame_shot_is_not_worth_retrying(self):
        """Dropping the only image leaves nothing to describe."""
        import encode

        client = self._client(lambda parts: True)
        assert encode._retry_without_poisoned_frame(
            client, self._types(), self._content(n_images=1), config=None, idx=7) is None

    def test_an_api_error_on_one_subset_does_not_abort_the_rest(self):
        import encode

        content = self._content()
        good = content[2]

        class Models:
            def generate_content(self, model, contents, config):
                parts = contents[0].parts
                if good in parts and content[0] in parts:
                    raise RuntimeError("transient")
                return TestPoisonedKeyframeRecovery._Resp(
                    "{}" if content[0] not in parts else "")

        class Client:
            models = Models()

        resp = encode._retry_without_poisoned_frame(
            Client(), self._types(), content, config=None, idx=451)
        assert resp is not None


class TestIdentityDescriptionSpec:
    """The description is the only identity signal the model gets.

    Names are stripped before the prompt is sent, so a description made of
    personality words leaves nothing to draw. Han came back as "a smuggler
    and pilot: cocky and cynical in manner, quick-moving and physically
    confident" -- half of it unrenderable, silently dropped by the model,
    which fell back on a generic handsome lead.
    """

    def test_it_demands_drawable_geometry(self):
        import encode

        spec = encode.IDENTITY_DESCRIPTION_SPEC.lower()
        for feature in ("face shape", "brow", "nose", "jaw"):
            assert feature in spec

    def test_it_forbids_the_unrenderable(self):
        import encode

        spec = encode.IDENTITY_DESCRIPTION_SPEC.lower()
        for banned in ("personality", "temperament", "profession", "backstory"):
            assert banned in spec

    def test_it_keeps_the_costume_rule(self):
        """Wardrobe is per-shot, except where the shell IS the character."""
        import encode

        spec = encode.IDENTITY_DESCRIPTION_SPEC.lower()
        assert "clothing or costume" in spec
        assert "droids" in spec

    def test_all_three_stage3_prompts_share_one_spec(self):
        """Three copies of this instruction used to drift apart."""
        import encode

        assert "__IDENTITY_SPEC__" not in encode.STAGE3_SYSTEM_PROMPT
        assert encode.IDENTITY_DESCRIPTION_SPEC in encode.STAGE3_SYSTEM_PROMPT


class TestCuratedFieldsSurviveRegeneration:
    """Stage 3 rewrites characters.json wholesale.

    `keep_name` marks the characters whose name is their design rather than a
    person, so it must survive the prompt-time strip. Regenerating the
    registry silently dropped it, and the damage only showed up as
    stormtroopers rendering as generic soldiers, several dollars later.
    """

    def _existing(self, tmp_path, entries):
        import json

        p = tmp_path / "characters.json"
        p.write_text(json.dumps({"characters": entries}))
        return str(p)

    def test_a_hand_set_flag_is_carried_forward(self, tmp_path):
        import encode

        path = self._existing(tmp_path, [{"name": "stormtrooper", "keep_name": True}])
        fresh = {"characters": [{"name": "stormtrooper", "description": "new"}]}

        encode._preserve_curated_fields(path, fresh)

        assert fresh["characters"][0]["keep_name"] is True
        assert fresh["characters"][0]["description"] == "new"  # regenerated text wins

    def test_characters_without_the_flag_are_untouched(self, tmp_path):
        import encode

        path = self._existing(tmp_path, [{"name": "luke"}])
        fresh = {"characters": [{"name": "luke", "description": "new"}]}

        encode._preserve_curated_fields(path, fresh)

        assert "keep_name" not in fresh["characters"][0]

    def test_a_character_that_no_longer_exists_is_ignored(self, tmp_path):
        import encode

        path = self._existing(tmp_path, [{"name": "gone", "keep_name": True}])
        fresh = {"characters": [{"name": "luke"}]}

        encode._preserve_curated_fields(path, fresh)  # must not raise
        assert "keep_name" not in fresh["characters"][0]

    def test_a_first_run_with_no_existing_registry_is_fine(self, tmp_path):
        import encode

        fresh = {"characters": [{"name": "luke"}]}
        encode._preserve_curated_fields(str(tmp_path / "nope.json"), fresh)
        assert fresh["characters"] == [{"name": "luke"}]

    def test_a_corrupt_registry_does_not_abort_stage3(self, tmp_path):
        import encode

        p = tmp_path / "characters.json"
        p.write_text("{ not json")
        fresh = {"characters": [{"name": "luke"}]}
        encode._preserve_curated_fields(str(p), fresh)  # must not raise
