"""Tests for voice_casting.py: proposed casting + sample generation."""

import json
from unittest.mock import MagicMock

import manifest
from voice_casting import (
    _sample_line_for,
    main,
    propose_casting,
)

CHARACTERS = [
    {"name": "luke", "display_name": "Luke Skywalker",
     "description": "Young man, blond hair.", "shots": [0]},
    {"name": "leia", "display_name": "Princess Leia",
     "description": "Young woman.", "shots": [1]},
]


class TestProposeCasting:
    def test_returns_parsed_json(self):
        gemini_response = {
            "casting": {
                "luke": {"voice": "Charlie", "reasoning": "young male"},
                "leia": {"voice": "Alice", "reasoning": "confident female"},
            },
            "narrator": {"voice": "Roger", "reasoning": "default"},
        }
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(
            text=json.dumps(gemini_response)
        )
        result = propose_casting(mock_client, CHARACTERS)
        assert result["casting"]["luke"]["voice"] == "Charlie"
        assert result["narrator"]["voice"] == "Roger"


class TestSampleLineFor:
    def test_uses_attributed_line_when_available(self, tmp_path):
        manifest.save_shots(
            str(tmp_path),
            [{"index": 0, "start_s": 0.0, "end_s": 5.0, "duration_s": 5.0}],
            [{"text": "Help me, Obi-Wan.", "start_s": 1.0, "end_s": 2.0}],
        )
        (tmp_path / "speakers.json").write_text(json.dumps({
            "format": "v1",
            "assignments": {"0": {"character": "leia", "confidence": 0.9}},
        }))
        assert _sample_line_for("leia", str(tmp_path)) == "Help me, Obi-Wan."

    def test_falls_back_to_generic_sample_when_no_speakers(self, tmp_path):
        line = _sample_line_for("luke", str(tmp_path))
        assert "luke" in line.lower()


class TestMainWritesVoiceMap:
    def test_writes_voice_map_gated_to_curated_bank(self, tmp_path, monkeypatch):
        (tmp_path / "characters.json").write_text(json.dumps({"characters": CHARACTERS}))
        gemini_response = {
            "casting": {
                "luke": {"voice": "Charlie", "reasoning": "young male"},
                "leia": {"voice": "NotInBank", "reasoning": "bad name"},
            },
            "narrator": {"voice": "Roger", "reasoning": "default"},
        }
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(
            text=json.dumps(gemini_response)
        )
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setattr("google.genai.Client", lambda **kwargs: mock_client)
        monkeypatch.setattr("sys.argv", ["voice_casting.py", str(tmp_path)])

        main()

        voice_map = manifest.load_voice_map(str(tmp_path))
        assert voice_map["luke"] == "Charlie"
        # "leia" entry is dropped since its proposed voice isn't in the curated bank
        assert "leia" not in voice_map
        assert voice_map["narrator"] == "Roger"
        assert (tmp_path / "voice_casting.md").exists()

    def test_missing_characters_json_exits(self, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.argv", ["voice_casting.py", str(tmp_path)])
        try:
            main()
            assert False, "expected SystemExit"
        except SystemExit:
            pass
