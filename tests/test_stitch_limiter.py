#!/usr/bin/env python3
"""Final-mix peak control in stitch._mux_audio (research 0023 defect 12, 0026).

Both the single-track and multi-track audio paths must end in the same
true-peak limiter before the AAC mux. The filter-chain tests drive
_mux_audio with ffmpeg and track building mocked out; the measurement test
runs real ffmpeg (skipped when it is absent).
"""

import re
import shutil
import subprocess
from unittest.mock import patch

import pytest

import stitch


def _mux(tmp_path, *, with_sfx):
    """Run _mux_audio with mocks; return {step label: ffmpeg argv}."""
    (tmp_path / "speech.wav").write_bytes(b"")
    (tmp_path / "sfx.wav").write_bytes(b"")
    calls = {}

    def fake_run(cmd, label):
        calls[label] = cmd

    video = str(tmp_path / "out.mp4")
    (tmp_path / "out.mp4").write_bytes(b"")
    prompts = [{"index": 0, "duration_s": 1.0}]
    with patch.object(stitch, "_run_ffmpeg", fake_run), \
         patch.object(stitch, "_probe_duration", return_value=1.0), \
         patch.object(stitch, "_audio_matches_encode", return_value=True), \
         patch.object(stitch, "_stitch_speech",
                      return_value=str(tmp_path / "speech.wav")), \
         patch.object(stitch, "_stitch_audio",
                      return_value=str(tmp_path / "sfx.wav")), \
         patch.object(stitch.os, "replace"):
        stitch._mux_audio(
            str(tmp_path), video, [str(tmp_path / "0-a.mp4")], prompts,
            "sfx" if with_sfx else None, "voice",
        )
    return calls


def _limiter_stage(chain):
    """The 3-stage limiter suffix must be present, in order, at the chain's end."""
    return chain.endswith(stitch._true_peak_limiter())


class TestFilterChains:
    def test_single_track_path_gain_then_limiter(self, tmp_path):
        calls = _mux(tmp_path, with_sfx=False)
        cmd = calls["single audio track volume filter"]
        chain = cmd[cmd.index("-af") + 1]
        # speech-only mix: +6 dB speech gain, then the limiter
        assert chain == "volume=6dB," + stitch._true_peak_limiter()
        assert "level=disabled" in chain

    def test_multi_track_path_amix_then_limiter(self, tmp_path):
        calls = _mux(tmp_path, with_sfx=True)
        cmd = calls["combined audio mix"]
        graph = cmd[cmd.index("-filter_complex") + 1]
        assert "amix=inputs=2:duration=longest:normalize=0," in graph
        assert _limiter_stage(graph.removesuffix("[out]"))
        assert graph.endswith("[out]")

    def test_both_paths_share_one_limiter(self, tmp_path):
        single = _mux(tmp_path, with_sfx=False)["single audio track volume filter"]
        multi = _mux(tmp_path, with_sfx=True)["combined audio mix"]
        for cmd in (single, multi):
            joined = " ".join(cmd)
            assert "alimiter=limit=0.8:level=disabled" in joined
            assert f"aresample={stitch.LIMITER_OVERSAMPLE_RATE}" in joined
            assert joined.count("alimiter") == 1

    def test_no_default_autolevel(self):
        # alimiter's auto-level default re-normalises the output and defeats the ceiling
        assert "level=disabled" in stitch._true_peak_limiter()
        assert "level=1" not in stitch._true_peak_limiter()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
class TestMeasuredTruePeak:
    @pytest.mark.parametrize("graph", ["single", "multi"])
    def test_hot_mix_stays_below_minus_1_dbtp_after_aac(self, tmp_path, graph):
        src = tmp_path / "hot.wav"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
             "anoisesrc=d=6:c=pink:a=0.7:r=44100:seed=12345", "-ac", "2", str(src)],
            check=True,
        )
        wav, aac = tmp_path / "mix.wav", tmp_path / "mix.m4a"
        if graph == "single":
            cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(src),
                   "-af", stitch._single_track_filter(6), str(wav)]
        else:
            cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-i", str(src),
                   "-filter_complex", stitch._mix_filter_complex([0, 6]),
                   "-map", "[out]", str(wav)]
        subprocess.run(cmd, check=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav),
                        "-c:a", "aac", "-b:a", "192k", str(aac)], check=True)
        out = subprocess.run(
            ["ffmpeg", "-nostats", "-i", str(aac), "-af", "ebur128=peak=true",
             "-f", "null", "-"], capture_output=True, text=True,
        ).stderr
        peak = float(re.findall(r"Peak:\s+(-?[\d.]+) dBFS", out)[-1])
        assert peak <= -1.0


# -- Review of dfe0a70 (research 0026 note, 2026-11-15) ------------------------
# The test above feeds pink noise at 0.7 and gains it +6 dB: that peaks at about
# -1.1 dBFS in an earlier sample. These use full-scale input, and state what the
# limiter does to loudness.

def _render(tmp_path, graph, src_expr, gain_db=6, *, limited=True):
    """src -> final-mix filter (single or multi path) -> float wav. Returns the path."""
    src = tmp_path / "src.wav"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", src_expr, "-ac", "2",
         "-ar", "44100", str(src)], check=True)
    wav = tmp_path / f"{graph}_{'lim' if limited else 'nolim'}.wav"
    if graph == "single":
        af = stitch._single_track_filter(gain_db) if limited else f"volume={gain_db}dB"
        cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-af", af]
    else:
        fc = stitch._mix_filter_complex([gain_db, gain_db])
        if not limited:
            fc = fc.replace("," + stitch._true_peak_limiter(), "")
        cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-i", str(src),
               "-filter_complex", fc, "-map", "[out]"]
    subprocess.run(cmd + ["-c:a", "pcm_f32le", str(wav)], check=True)
    return wav


def _ebur(path):
    out = subprocess.run(
        ["ffmpeg", "-nostats", "-i", str(path), "-af", "ebur128=peak=true", "-f", "null", "-"],
        capture_output=True, text=True, check=True).stderr
    lufs = float(re.findall(r"I:\s+(-?[\d.]+) LUFS", out)[-1])
    peak = float(re.findall(r"Peak:\s+(-?[\d.]+) dBFS", out)[-1])
    return lufs, peak


# Full-scale noise: +6 dB gain takes it to about +2 dBFS before the limiter.
FULL_SCALE_NOISE = "anoisesrc=d=6:c=pink:a=1.0:r=44100:seed=12345"
# Worst case for AAC: a hard-edged full-scale square (see the xfail below).
FULL_SCALE_SQUARE = (
    "aevalsrc='0.999*sgn(sin(2*PI*220*t))|0.999*sgn(sin(2*PI*220*t))':d=6:s=44100")
# Dialogue-like bursts peaking at 0.25 (-12 dBFS); +6 dB leaves them at -6 dBFS,
# under the 0.8 ceiling, so the limiter should have nothing to do.
QUIET_BURSTS = "aevalsrc='0.2*sin(2*PI*200*t)*pow(abs(sin(2*PI*2.3*t)),3)':d=10:s=44100"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
class TestFullScaleInputAndLoudness:
    @pytest.mark.parametrize("graph", ["single", "multi"])
    def test_full_scale_square_is_limited_before_aac(self, tmp_path, graph):
        _, peak = _ebur(_render(tmp_path, graph, FULL_SCALE_SQUARE))
        assert peak <= -1.0

    def _after_aac_peak(self, tmp_path, graph, src):
        _lufs, raw_peak = _ebur(_render(tmp_path, graph, src, limited=False))
        assert raw_peak > 0                     # without the limiter this really clips
        aac = tmp_path / "mix.m4a"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(_render(tmp_path, graph, src)),
                        "-c:a", "aac", "-b:a", "192k", str(aac)], check=True)
        return _ebur(aac)[1]

    @pytest.mark.parametrize("graph", ["single", pytest.param("multi", marks=pytest.mark.xfail(
        strict=True, reason="KNOWN GAP: deterministic hot-noise multi-track AAC peaks at -0.6 dBTP"))])
    def test_full_scale_input_is_held_below_minus_1_dbtp_after_aac(self, tmp_path, graph):
        assert self._after_aac_peak(tmp_path, graph, FULL_SCALE_NOISE) <= -1.0

    @pytest.mark.xfail(strict=True, reason=(
        "KNOWN GAP (research 0026 note): ffmpeg's AAC encoder overshoots the "
        "limiter ceiling on a hard-edged full-scale square. "
        "Whether the 0.8 ceiling needs to drop depends on real programme audio."))
    @pytest.mark.parametrize("graph", ["single", "multi"])
    def test_hard_square_after_aac_stays_below_minus_1_dbtp(self, tmp_path, graph):
        assert self._after_aac_peak(tmp_path, graph, FULL_SCALE_SQUARE) <= -1.0

    @pytest.mark.parametrize("graph", ["single", "multi"])
    def test_programme_below_the_ceiling_keeps_its_loudness(self, tmp_path, graph):
        """The stated bound: material that never reaches the 0.8 ceiling comes
        out within 0.2 LU of where it went in (measured: 0.0 LU)."""
        before, _ = _ebur(_render(tmp_path, graph, QUIET_BURSTS, limited=False))
        after, _ = _ebur(_render(tmp_path, graph, QUIET_BURSTS))
        print(f"{graph} quiet programme: {before:.1f} -> {after:.1f} LUFS")
        assert abs(after - before) <= 0.2

    @pytest.mark.xfail(strict=True, reason=(
        "KNOWN GAP: sustained hot input loses more than 1 LU; a peak limiter "
        "alone cannot guarantee programme loudness. See research 0026 follow-ups."))
    @pytest.mark.parametrize("graph", ["single", "multi"])
    def test_hot_programme_loudness_stays_within_one_lu(self, tmp_path, graph):
        before, _ = _ebur(_render(tmp_path, graph, FULL_SCALE_SQUARE, limited=False))
        after, _ = _ebur(_render(tmp_path, graph, FULL_SCALE_SQUARE))
        print(f"{graph} hot programme: {before:.1f} -> {after:.1f} LUFS")
        assert abs(after - before) <= 1.0
