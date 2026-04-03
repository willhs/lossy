---
id: plan-0003
type: spec
purpose: "Implementation plan for single-command pipeline orchestration."
tags: ["plan", "cli", "pipeline"]
related: ["./task.md"]
---

# Pipeline Orchestration Implementation Plan

## Overview

Add `pipeline.py` — a standalone script that chains encode stage1 → encode stage2 → decode → stitch via subprocess calls. Each stage runs as a child process, preserving ADR-002's stateless CLI model. The pipeline script is purely orchestration: no imports from encode.py or decode.py.

**Primary Goal**: `python pipeline.py media/film.mp4 -o output/film --strategy fal-seedance` runs the full pipeline.

**Approach**: Subprocess calls to existing scripts. Stop on first non-zero exit. Print timing summary at end.

## Current State Analysis

- `encode.py` has two subcommands: `stage1` (video → manifest + keyframes) and `stage2` (manifest → prompts). Both call `sys.exit(1)` on failure.
- `decode.py` has a flat argparse with `--stitch` flag toggling between generation and stitching modes. Also calls `sys.exit(1)` on failure.
- All stages use `sys.exit(1)` for errors, which means subprocess return codes are reliable.
- Each stage has built-in resume/skip logic via cached files on disk.

### Key Discoveries
- `encode.py:698-723` — `run_stage1` takes args with `.video`, `.output`, `.detector`, `.threshold`
- `encode.py:726-769` — `run_stage2` takes args with `.output_dir`, `.limit`, `.provider`
- `decode.py:1033-1063` — main() takes `output_dir`, `--start-index`, `--limit`, `--strategy`, `--stitch`
- All four strategies are available: replicate-wan, fal-seedance, fal-seedance-pro, runpod-wan

## Desired End State

A single `python pipeline.py` invocation runs all four stages sequentially, with:
- Clear per-stage status output
- Timing per stage and total
- `--skip` to skip named stages
- `--dry-run` to preview commands
- Non-zero exit on any stage failure

## What We're NOT Doing

- Importing functions from encode.py/decode.py (subprocess keeps independence)
- Adding new dependencies
- Changing encode.py or decode.py internals
- Parallel execution

---

## Phase 1: Create pipeline.py

### Overview
Single file, no changes to existing code.

### Tasks

#### 1. Create `pipeline.py`
- [x] Create `pipeline.py`

```python
#!/usr/bin/env python3
"""Pipeline orchestrator — chains encode → decode → stitch in one command."""

import argparse
import subprocess
import sys
import time

STAGES = ["encode1", "encode2", "decode", "stitch"]

STRATEGIES = ["replicate-wan", "fal-seedance", "fal-seedance-pro", "runpod-wan"]


def build_commands(args):
    """Build the subprocess command list for each stage."""
    commands = {}

    commands["encode1"] = [
        sys.executable, "encode.py", "stage1", args.video,
        "-o", args.output,
    ]
    if args.detector:
        commands["encode1"] += ["-d", args.detector]
    if args.threshold is not None:
        commands["encode1"] += ["-t", str(args.threshold)]

    commands["encode2"] = [
        sys.executable, "encode.py", "stage2", args.output,
    ]
    if args.limit:
        commands["encode2"] += ["--limit", str(args.limit)]

    commands["decode"] = [
        sys.executable, "decode.py", args.output,
        "--strategy", args.strategy,
    ]
    if args.start_index is not None:
        commands["decode"] += ["--start-index", str(args.start_index)]
    if args.limit:
        commands["decode"] += ["--limit", str(args.limit)]

    commands["stitch"] = [
        sys.executable, "decode.py", args.output,
        "--strategy", args.strategy,
        "--stitch",
    ]

    return commands


def run_pipeline(args):
    """Run each stage in sequence, stopping on first failure."""
    skip = set(args.skip) if args.skip else set()
    commands = build_commands(args)
    timings = []

    for stage in STAGES:
        if stage in skip:
            print(f"\n{'='*60}")
            print(f"SKIP: {stage}")
            print(f"{'='*60}")
            timings.append((stage, 0, "skipped"))
            continue

        cmd = commands[stage]

        if args.dry_run:
            print(f"[dry-run] {' '.join(cmd)}")
            timings.append((stage, 0, "dry-run"))
            continue

        print(f"\n{'='*60}")
        print(f"STAGE: {stage}")
        print(f"{'='*60}")
        print(f"Running: {' '.join(cmd)}\n")

        start = time.time()
        result = subprocess.run(cmd)
        elapsed = time.time() - start

        if result.returncode != 0:
            timings.append((stage, elapsed, "FAILED"))
            print(f"\n{'='*60}")
            print(f"PIPELINE FAILED at stage: {stage}")
            print(f"Exit code: {result.returncode}")
            print(f"{'='*60}")
            print_summary(timings, args)
            sys.exit(1)

        timings.append((stage, elapsed, "ok"))

    print_summary(timings, args)


def print_summary(timings, args):
    """Print a summary of all stages."""
    print(f"\n{'='*60}")
    print("PIPELINE SUMMARY")
    print(f"{'='*60}")

    total = 0
    for stage, elapsed, status in timings:
        if status == "skipped":
            print(f"  {stage:10s}  skipped")
        elif status == "dry-run":
            print(f"  {stage:10s}  dry-run")
        elif status == "FAILED":
            print(f"  {stage:10s}  FAILED  ({elapsed:.1f}s)")
            total += elapsed
        else:
            print(f"  {stage:10s}  ok      ({elapsed:.1f}s)")
            total += elapsed

    print(f"  {'':10s}  -------")
    print(f"  {'total':10s}  {total:.1f}s ({total / 60:.1f}min)")

    # Print output paths
    print(f"\nOutput directory: {args.output}")
    print(f"Strategy: {args.strategy}")
    reconstructed = f"{args.output}/reconstructed_{args.strategy.replace('-', '_')}.mp4"
    print(f"Reconstructed video: {reconstructed}")


def main():
    parser = argparse.ArgumentParser(
        description="Run the full lossy pipeline: encode → decode → stitch"
    )
    parser.add_argument("video", help="Path to source video file")
    parser.add_argument("--output", "-o", default="output",
                        help="Output directory (default: output)")
    parser.add_argument("--strategy", choices=STRATEGIES,
                        default="replicate-wan",
                        help="Video generation backend (default: replicate-wan)")

    # Encode options
    parser.add_argument("--detector", "-d", choices=["adaptive", "content"],
                        default=None, help="Shot detection method")
    parser.add_argument("--threshold", "-t", type=float, default=None,
                        help="Shot detection threshold")

    # Decode options
    parser.add_argument("--start-index", type=int, default=None,
                        help="Skip shots before this index")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N shots")

    # Pipeline control
    parser.add_argument("--skip", nargs="+", choices=STAGES,
                        help="Stages to skip (e.g., --skip encode1 encode2)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
```

#### 2. Create tests
- [x] Create `test_pipeline.py`

```python
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
```

### Success Criteria

#### Automated Verification:
- [x] Run: `python -m pytest test_pipeline.py -v` — all tests pass
- [x] Run: `python pipeline.py --help` — shows usage with all options
- [x] Run: `python pipeline.py media/test.mp4 -o output/test --dry-run` — prints 4 commands without executing

#### Manual Verification:
- [x] Manual: Verify existing `python encode.py stage1 --help` and `python decode.py --help` still work unchanged

---

## Final Checklist

- [x] All phases complete
- [x] All tests passing
- [x] Existing encode.py and decode.py unchanged and functional

## Documentation Updates

- [x] Update `docs/design/adr/002-stateless-cli-pipeline.md` — add a note that `pipeline.py` provides convenience orchestration while preserving the stateless model

## References

- Task: `docs/tasks/0003-pipeline-orchestration/task.md`
- ADR: `docs/design/adr/002-stateless-cli-pipeline.md`
