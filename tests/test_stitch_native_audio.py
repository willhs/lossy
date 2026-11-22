"""--audio-strategy native and --scale (task 462c92a3: the 30 Nov arms A and B).

Arm A keeps the soundtrack LTX-2.5 generates with each clip, which the stitch
otherwise drops (clips are retimed with -an). The extraction runs real ffmpeg
on tiny synthetic clips (skipped when ffmpeg is absent).
"""

import json
import shutil
import subprocess

import pytest

import stitch

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _clip(path, seconds, with_audio):
    cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=gray:s=64x36:d={seconds}"]
    if with_audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", "-shortest"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, check=True, capture_output=True)
    return str(path)


def _output_dir(tmp_path):
    shots = [{"index": i, "start_s": float(i), "end_s": float(i + 1), "duration_s": 1.0}
             for i in range(3)]
    (tmp_path / "shots.json").write_text(json.dumps({"format": "v2", "shots": shots, "dialog": []}))
    return shots


@needs_ffmpeg
def test_extracts_each_parts_audio_and_skips_silent_clips(tmp_path):
    shots = _output_dir(tmp_path)
    groups = [
        (shots[0], [_clip(tmp_path / "0000.mp4", 1, True)]),
        (shots[1], [_clip(tmp_path / "0001-01.mp4", 1, True), _clip(tmp_path / "0001-02.mp4", 1, True)]),
        (shots[2], [_clip(tmp_path / "0002.mp4", 1, False)]),
    ]

    assert stitch.extract_native_audio(str(tmp_path), groups) == 2

    progress = json.loads((tmp_path / "audio_progress_native.json").read_text())
    assert [c["path"] for c in progress["clips"]["0"]] == ["0000.flac"]
    assert [c["path"] for c in progress["clips"]["1"]] == ["0001-01.flac", "0001-02.flac"]
    assert "2" not in progress["clips"]
    assert progress["encode_fingerprint"]
    for c in progress["clips"]["1"]:
        assert (tmp_path / "audio" / "native" / c["path"]).exists()
        assert c["duration_s"] == pytest.approx(1.0, abs=0.1)


@needs_ffmpeg
def test_rerun_is_idempotent(tmp_path):
    shots = _output_dir(tmp_path)
    groups = [(shots[0], [_clip(tmp_path / "0000.mp4", 1, True)])]
    stitch.extract_native_audio(str(tmp_path), groups)
    flac = tmp_path / "audio" / "native" / "0000.flac"
    before = flac.stat().st_mtime_ns

    assert stitch.extract_native_audio(str(tmp_path), groups) == 1
    assert flac.stat().st_mtime_ns == before


def test_native_audio_is_never_auto_mixed_into_a_plain_stitch(tmp_path, monkeypatch):
    (tmp_path / "audio" / "native").mkdir(parents=True)
    (tmp_path / "audio" / "runpod-mmaudio").mkdir(parents=True)
    seen = []
    monkeypatch.setattr(stitch, "_stitch_audio", lambda _od, strat, _p, _si: seen.append(strat))
    monkeypatch.setattr(stitch, "_audio_matches_encode", lambda _od, _s: True)

    stitch._mux_audio(str(tmp_path), str(tmp_path / "v.mp4"), [], [], None, None)

    assert seen == ["runpod-mmaudio"]


@pytest.mark.parametrize("bad", ["1280", "1280x", "x704", "1280*704"])
def test_scale_rejects_malformed_sizes(bad):
    with pytest.raises(ValueError, match="WxH"):
        stitch._scale_filter(bad)


def test_scale_is_lanczos():
    assert stitch._scale_filter("1280x704") == "scale=1280:704:flags=lanczos"
