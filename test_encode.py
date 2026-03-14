"""Tests for encode.py data transformation functions."""

import numpy as np
import pytest

from encode import (
    align_subtitles_to_shots,
    classify_motion,
    frames_for_duration,
    parse_srt,
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
        assert result == {0: ["Hello"]}

    def test_subtitle_spanning_two_shots(self):
        scenes = [
            self._scene(0, 10.0, 15.0),
            self._scene(1, 15.0, 20.0),
        ]
        subs = [self._sub(14.0, 16.0, "Spanning")]
        result = align_subtitles_to_shots(subs, scenes)
        assert 0 in result
        assert 1 in result
        assert result[0] == ["Spanning"]
        assert result[1] == ["Spanning"]

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
        assert result == {0: ["First", "Second"]}

    def test_exact_boundary_no_overlap(self):
        """Subtitle ends exactly when shot starts — no overlap."""
        scenes = [self._scene(0, 10.0, 20.0)]
        subs = [self._sub(5.0, 10.0, "Before")]
        result = align_subtitles_to_shots(subs, scenes)
        assert result == {}

    def test_exact_boundary_overlap(self):
        """Subtitle starts exactly when shot starts — overlaps."""
        scenes = [self._scene(0, 10.0, 20.0)]
        subs = [self._sub(10.0, 12.0, "Exact")]
        result = align_subtitles_to_shots(subs, scenes)
        assert result == {0: ["Exact"]}


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
