#!/usr/bin/env python3
"""Pipeline orchestrator — chains encode → decode → stitch in one command."""

import argparse
import json
import os
import subprocess
import sys
import time

STAGES = ["encode1", "encode2", "encode3", "decode", "audio", "speech", "stitch"]

STRATEGIES = ["replicate-wan", "fal-seedance", "fal-seedance-pro", "runpod-wan", "runpod-wan-enriched", "runpod-vace"]

AUDIO_STRATEGIES = ["elevenlabs", "mmaudio", "runpod-mmaudio"]


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

    commands["encode3"] = [
        sys.executable, "encode.py", "stage3", args.output,
    ]

    commands["decode"] = [
        sys.executable, "decode.py", args.output,
        "--strategy", args.strategy,
    ]
    # When both video and audio are on RunPod, run audio concurrently on the same pod
    if args.strategy in ("runpod-wan", "runpod-vace") and args.audio_strategy == "runpod-mmaudio":
        commands["decode"].append("--concurrent-audio")
        commands["decode"].append("--keep-pod")
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

    commands["speech"] = [
        sys.executable, "decode.py", args.output,
        "--speech", "--speech-voice", args.speech_voice or "Roger",
    ]
    if args.start_index is not None:
        commands["speech"] += ["--start-index", str(args.start_index)]
    if args.limit:
        commands["speech"] += ["--limit", str(args.limit)]

    commands["stitch"] = [
        sys.executable, "decode.py", args.output,
        "--strategy", args.strategy,
        "--stitch",
    ]
    if args.audio_strategy:
        commands["stitch"] += ["--audio-strategy", args.audio_strategy]
    if args.speech_voice:
        commands["stitch"] += ["--speech-voice", args.speech_voice]

    return commands


def run_pipeline(args):
    """Run each stage in sequence, stopping on first failure."""
    skip = set(args.skip) if args.skip else set()
    if not args.audio_strategy:
        skip.add("audio")
    # Skip separate audio stage when decode handles it concurrently
    if args.strategy in ("runpod-wan", "runpod-vace") and args.audio_strategy == "runpod-mmaudio":
        skip.add("audio")
    if not args.speech_voice:
        skip.add("speech")
    # encode3 (character registry) is only needed for VACE
    if args.strategy != "runpod-vace":
        skip.add("encode3")
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

    # Clean up RunPod pod state file (pod is terminated by the last stage that uses it)
    pod_state = os.path.join(args.output, "runpod_pod.json")
    if os.path.exists(pod_state):
        # Pod wasn't cleaned up — terminate it as safety net
        try:
            with open(pod_state) as f:
                state = json.load(f)
            import runpod
            runpod.api_key = os.environ.get("RUNPOD_API_KEY")
            if runpod.api_key and state.get("pod_id"):
                runpod.terminate_pod(state["pod_id"])
                print(f"  Safety net: terminated pod {state['pod_id']}")
            os.remove(pod_state)
        except Exception as e:
            print(f"  Warning: Could not clean up pod: {e}")
            print(f"  Check https://www.runpod.io/console/pods")

    print_summary(timings, args)


def aggregate_costs(output_dir, strategy, audio_strategy, timings):
    """Read per-stage cost files and write a unified costs.json."""
    timing_map = {stage: elapsed for stage, elapsed, status in timings if status not in ("skipped", "dry-run")}
    costs = {"stages": {}, "total_cost_estimate": 0.0, "total_wall_clock_s": 0.0}

    # Encode (Gemini) costs
    encode_costs_path = os.path.join(output_dir, "encode_costs.json")
    if os.path.exists(encode_costs_path):
        with open(encode_costs_path) as f:
            enc = json.load(f)
        costs["stages"]["encode2"] = {
            "provider": "gemini",
            "model": enc.get("model", "unknown"),
            "cost_estimate": enc.get("cost_estimate", 0.0),
            "wall_clock_s": round(timing_map.get("encode2", 0.0), 1),
            "shots": len(enc.get("per_shot", [])),
            "total_input_tokens": enc.get("total_input_tokens", 0),
            "total_output_tokens": enc.get("total_output_tokens", 0),
        }
        costs["total_cost_estimate"] += enc.get("cost_estimate", 0.0)

    # Decode (video) costs
    strategy_name = strategy.replace("-", "_")
    decode_progress_path = os.path.join(output_dir, f"decode_progress_{strategy_name}.json")
    if os.path.exists(decode_progress_path):
        with open(decode_progress_path) as f:
            dec = json.load(f)
        costs["stages"]["decode"] = {
            "provider": strategy,
            "cost_estimate": dec.get("total_cost_estimate", 0.0),
            "wall_clock_s": round(timing_map.get("decode", 0.0), 1),
            "shots_completed": len(dec.get("completed", [])),
            "shots_failed": len(dec.get("failed", [])),
        }
        costs["total_cost_estimate"] += dec.get("total_cost_estimate", 0.0)

    # Audio costs
    if audio_strategy:
        audio_progress_path = os.path.join(output_dir, f"audio_progress_{audio_strategy}.json")
        if os.path.exists(audio_progress_path):
            with open(audio_progress_path) as f:
                aud = json.load(f)
            costs["stages"]["audio"] = {
                "provider": audio_strategy,
                "cost_estimate": aud.get("total_cost_estimate", 0.0),
                "wall_clock_s": round(timing_map.get("audio", 0.0), 1),
                "shots_completed": len(aud.get("completed", [])),
                "shots_skipped": len(aud.get("skipped", [])),
                "shots_failed": len(aud.get("failed", [])),
            }
            costs["total_cost_estimate"] += aud.get("total_cost_estimate", 0.0)

    costs["total_cost_estimate"] = round(costs["total_cost_estimate"], 4)
    costs["total_wall_clock_s"] = round(sum(timing_map.values()), 1)

    costs_path = os.path.join(output_dir, "costs.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(costs_path, "w") as f:
        json.dump(costs, f, indent=2)

    return costs


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

    # Aggregate and print costs
    if not args.dry_run:
        costs = aggregate_costs(args.output, args.strategy, args.audio_strategy, timings)
        if costs["total_cost_estimate"] > 0:
            print(f"\n  Estimated total cost: ${costs['total_cost_estimate']:.2f}")
            for stage_name, stage_data in costs["stages"].items():
                print(f"    {stage_name}: ${stage_data['cost_estimate']:.4f} ({stage_data['provider']})")
            print(f"\n  Cost breakdown: {args.output}/costs.json")

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
    parser.add_argument("--speech-voice", default=None,
                        help="ElevenLabs voice for speech generation (default: none, skip speech)")

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
