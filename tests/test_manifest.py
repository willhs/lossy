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


class TestExistingClipParts:
    """Resume must not adopt a shot whose parts were cut short mid-generation."""

    def _touch(self, tmp_path, name):
        (tmp_path / name).write_bytes(b"")

    def test_single_part_shot_present(self, tmp_path):
        self._touch(tmp_path, "0804.mp4")
        parts = manifest.existing_clip_parts(str(tmp_path), 804, 1)
        assert [p.split("/")[-1] for p in parts] == ["0804.mp4"]

    def test_single_part_shot_absent(self, tmp_path):
        assert manifest.existing_clip_parts(str(tmp_path), 804, 1) == []

    def test_split_shot_all_parts_present(self, tmp_path):
        for name in ("0846-01.mp4", "0846-02.mp4", "0846-03.mp4"):
            self._touch(tmp_path, name)
        parts = manifest.existing_clip_parts(str(tmp_path), 846, 3)
        assert [p.split("/")[-1] for p in parts] == [
            "0846-01.mp4", "0846-02.mp4", "0846-03.mp4"]

    def test_split_shot_interrupted_after_first_part_is_not_resumable(self, tmp_path):
        # The regression: a run killed mid-shot leaves part 1 on disk. Adopting
        # it would truncate the shot, and with no clips_meta written the stitch
        # would drop the shot entirely.
        self._touch(tmp_path, "0846-01.mp4")
        assert manifest.existing_clip_parts(str(tmp_path), 846, 3) == []
        # ...even though the old first-part-only check calls it done.
        assert manifest.clip_exists_for_shot(str(tmp_path), 846) is True

    def test_split_shot_missing_middle_part(self, tmp_path):
        self._touch(tmp_path, "0846-01.mp4")
        self._touch(tmp_path, "0846-03.mp4")
        assert manifest.existing_clip_parts(str(tmp_path), 846, 3) == []
