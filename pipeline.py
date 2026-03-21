#!/usr/bin/env python3
"""Pipeline orchestrator — chains encode → decode → stitch in one command."""

import argparse
import subprocess
import sys
import time

STAGES = ["encode1", "encode2", "decode", "audio", "stitch"]

STRATEGIES = ["replicate-wan", "fal-seedance", "fal-seedance-pro", "runpod-wan"]

AUDIO_STRATEGIES = ["elevenlabs", "mmaudio"]


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

    commands["audio"] = [
        sys.executable, "decode.py", args.output,
        "--audio", "--audio-strategy", args.audio_strategy or "elevenlabs",
    ]
    if args.start_index is not None:
        commands["audio"] += ["--start-index", str(args.start_index)]
    if args.limit:
        commands["audio"] += ["--limit", str(args.limit)]

    commands["stitch"] = [
        sys.executable, "decode.py", args.output,
        "--strategy", args.strategy,
        "--stitch",
    ]
    if args.audio_strategy:
        commands["stitch"] += ["--audio-strategy", args.audio_strategy]

    return commands


def run_pipeline(args):
    """Run each stage in sequence, stopping on first failure."""
    skip = set(args.skip) if args.skip else set()
    if not args.audio_strategy:
        skip.add("audio")
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
    parser.add_argument("--audio-strategy", choices=AUDIO_STRATEGIES,
                        default=None,
                        help="Audio generation backend (default: none, skip audio)")

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
