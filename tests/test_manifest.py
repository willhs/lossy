"""Tests for manifest.py's speakers.json / voice_map.json sidecar helpers."""

import json

import manifest


class TestSpeakersPath:
    def test_load_missing_file_returns_empty(self, tmp_path):
        assert manifest.load_speakers(str(tmp_path)) == {}

    def test_load_parses_assignments_by_int_index(self, tmp_path):
        (tmp_path / "speakers.json").write_text(json.dumps({
            "format": "v1",
            "assignments": {
                "0": {"character": "luke", "confidence": 0.9},
                "5": {"character": "narrator", "confidence": 0.4},
            },
        }))
        speakers = manifest.load_speakers(str(tmp_path))
        assert speakers == {0: "luke", 5: "narrator"}
        assert isinstance(next(iter(speakers)), int)


class TestVoiceMapPath:
    def test_load_missing_file_returns_empty(self, tmp_path):
        assert manifest.load_voice_map(str(tmp_path)) == {}

    def test_load_returns_raw_mapping(self, tmp_path):
        (tmp_path / "voice_map.json").write_text(json.dumps({
            "luke": "Charlie", "leia": "Alice", "narrator": "Roger",
        }))
        assert manifest.load_voice_map(str(tmp_path)) == {
            "luke": "Charlie", "leia": "Alice", "narrator": "Roger",
        }


class TestPathHelpers:
    def test_speakers_path_location(self, tmp_path):
        assert manifest.speakers_path(str(tmp_path)) == str(tmp_path / "speakers.json")

    def test_voice_map_path_location(self, tmp_path):
        assert manifest.voice_map_path(str(tmp_path)) == str(tmp_path / "voice_map.json")
