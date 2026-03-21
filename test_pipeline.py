#!/usr/bin/env python3
"""Tests for pipeline.py orchestrator."""

import argparse
import pytest
from unittest.mock import patch, MagicMock
from pipeline import build_commands, STAGES


@pytest.fixture
def base_args():
    return argparse.Namespace(
        video="media/film.mp4",
        output="output/film",
        strategy="fal-seedance",
        audio_strategy=None,
        speech_voice=None,
        detector=None,
        threshold=None,
        start_index=None,
        limit=None,
        skip=None,
        dry_run=False,
    )


class TestBuildCommands:
    def test_basic_commands(self, base_args):
        cmds = build_commands(base_args)
        assert "encode1" in cmds
        assert "encode2" in cmds
        assert "decode" in cmds
        assert "stitch" in cmds

    def test_encode1_has_video_and_output(self, base_args):
        cmds = build_commands(base_args)
        cmd = cmds["encode1"]
        assert "encode.py" in cmd[1]
        assert "stage1" in cmd
        assert "media/film.mp4" in cmd
        assert "-o" in cmd
        assert "output/film" in cmd

    def test_encode2_has_output_dir(self, base_args):
        cmds = build_commands(base_args)
        cmd = cmds["encode2"]
        assert "encode.py" in cmd[1]
        assert "stage2" in cmd
        assert "output/film" in cmd

    def test_decode_has_strategy(self, base_args):
        cmds = build_commands(base_args)
        cmd = cmds["decode"]
        assert "decode.py" in cmd[1]
        assert "--strategy" in cmd
        assert "fal-seedance" in cmd
        assert "--stitch" not in cmd

    def test_stitch_has_strategy_and_flag(self, base_args):
        cmds = build_commands(base_args)
        cmd = cmds["stitch"]
        assert "decode.py" in cmd[1]
        assert "--strategy" in cmd
        assert "fal-seedance" in cmd
        assert "--stitch" in cmd

    def test_optional_detector(self, base_args):
        base_args.detector = "content"
        cmds = build_commands(base_args)
        assert "-d" in cmds["encode1"]
        assert "content" in cmds["encode1"]

    def test_optional_threshold(self, base_args):
        base_args.threshold = 3.5
        cmds = build_commands(base_args)
        assert "-t" in cmds["encode1"]
        assert "3.5" in cmds["encode1"]

    def test_optional_limit(self, base_args):
        base_args.limit = 10
        cmds = build_commands(base_args)
        assert "--limit" in cmds["encode2"]
        assert "10" in cmds["encode2"]
        assert "--limit" in cmds["decode"]
        assert "10" in cmds["decode"]

    def test_optional_start_index(self, base_args):
        base_args.start_index = 5
        cmds = build_commands(base_args)
        assert "--start-index" in cmds["decode"]
        assert "5" in cmds["decode"]

    def test_build_commands_with_audio(self, base_args):
        base_args.audio_strategy = "elevenlabs"
        commands = build_commands(base_args)
        assert "audio" in commands
        audio_cmd = commands["audio"]
        assert "--audio" in audio_cmd
        assert "--audio-strategy" in audio_cmd
        assert "elevenlabs" in audio_cmd

    def test_stitch_includes_audio_strategy(self, base_args):
        base_args.audio_strategy = "elevenlabs"
        commands = build_commands(base_args)
        stitch_cmd = commands["stitch"]
        assert "--audio-strategy" in stitch_cmd
        assert "elevenlabs" in stitch_cmd


class TestRunPipeline:
    @patch("pipeline.subprocess.run")
    def test_all_stages_run_in_order(self, mock_run, base_args):
        mock_run.return_value = MagicMock(returncode=0)
        from pipeline import run_pipeline
        run_pipeline(base_args)
        assert mock_run.call_count == 4

    @patch("pipeline.subprocess.run")
    def test_stops_on_failure(self, mock_run, base_args):
        mock_run.side_effect = [
            MagicMock(returncode=0),
            MagicMock(returncode=1),
        ]
        from pipeline import run_pipeline
        with pytest.raises(SystemExit) as exc:
            run_pipeline(base_args)
        assert exc.value.code == 1
        assert mock_run.call_count == 2

    @patch("pipeline.subprocess.run")
    def test_skip_stages(self, mock_run, base_args):
        mock_run.return_value = MagicMock(returncode=0)
        base_args.skip = ["encode1", "encode2"]
        from pipeline import run_pipeline
        run_pipeline(base_args)
        assert mock_run.call_count == 2

    def test_dry_run_no_subprocess(self, base_args, capsys):
        base_args.dry_run = True
        from pipeline import run_pipeline
        run_pipeline(base_args)
        output = capsys.readouterr().out
        assert "[dry-run]" in output
        assert "encode.py" in output
        assert "decode.py" in output
