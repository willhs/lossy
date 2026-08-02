"""Tests for manifest.py's speakers.json / voice_map.json sidecar helpers."""

import json
import os

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


class TestEncodeFingerprint:
    """Generated artifacts must be pinned to the encode that produced them.

    Clips, audio and speech are keyed by shot (or dialog-line) index, and
    nothing in those filenames says which encode they came from. A re-encode
    of Star Wars IV renumbered 1618 of 2069 shots, so March audio for "shot
    794" silently became audio for different footage -- and a stitch muxed it
    with no error.
    """

    def _write_shots(self, tmp_path, n=3, offset=0.0):
        shots = [{"index": i, "start_s": i * 2.0 + offset, "end_s": i * 2.0 + 2.0 + offset,
                  "duration_s": 2.0, "description": {}} for i in range(n)]
        manifest.save_shots(str(tmp_path), shots, [])

    def test_fingerprint_is_stable_for_the_same_shot_list(self, tmp_path):
        self._write_shots(tmp_path)
        assert manifest.encode_fingerprint(str(tmp_path)) == manifest.encode_fingerprint(str(tmp_path))

    def test_fingerprint_changes_when_shots_are_renumbered(self, tmp_path):
        self._write_shots(tmp_path, n=3)
        before = manifest.encode_fingerprint(str(tmp_path))
        self._write_shots(tmp_path, n=4)          # a re-encode finding another shot
        assert manifest.encode_fingerprint(str(tmp_path)) != before

    def test_fingerprint_changes_when_boundaries_shift(self, tmp_path):
        self._write_shots(tmp_path)
        before = manifest.encode_fingerprint(str(tmp_path))
        self._write_shots(tmp_path, offset=0.5)   # same count, different timings
        assert manifest.encode_fingerprint(str(tmp_path)) != before

    def test_matching_progress_is_accepted(self, tmp_path):
        self._write_shots(tmp_path)
        prog = {"encode_fingerprint": manifest.encode_fingerprint(str(tmp_path))}
        assert manifest.check_encode_fingerprint(prog, str(tmp_path), "test") is True

    def test_progress_from_a_different_encode_is_rejected(self, tmp_path):
        self._write_shots(tmp_path)
        prog = {"encode_fingerprint": "deadbeef1234"}
        assert manifest.check_encode_fingerprint(prog, str(tmp_path), "test") is False

    def test_unstamped_progress_gets_stamped(self, tmp_path):
        self._write_shots(tmp_path)
        prog = {}
        assert manifest.check_encode_fingerprint(prog, str(tmp_path), "test") is True
        assert prog["encode_fingerprint"] == manifest.encode_fingerprint(str(tmp_path))

    def test_unstamped_progress_older_than_the_encode_is_rejected(self, tmp_path):
        """The retroactive check: the March-audio-vs-July-encode case."""
        self._write_shots(tmp_path)
        shots_mtime = os.path.getmtime(manifest.shots_path(str(tmp_path)))
        prog = {"_mtime": shots_mtime - 3600}     # progress written an hour earlier
        assert manifest.check_encode_fingerprint(prog, str(tmp_path), "test") is False
