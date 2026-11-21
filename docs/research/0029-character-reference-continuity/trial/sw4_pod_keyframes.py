"""Validate self-hosted klein keyframes on a real pod (task 0017, research 0029 "Self-hosted klein").

Runs exactly the path `decode.py --strategy runpod-ltx25-keyframe --keyframe-generator pod` runs:
the strategy's `prepare()` boots the pod, downloads LTX-2.5 and klein, draws the sheets and keyframes
on the pod and unloads klein; then ONE clip is rendered for the first keyframe shot, which proves LTX
loads after the unload. Only one clip, to keep the check cheap: the clip-level K-vs-T result is
research 0029's. The pod is terminated on exit (the session's atexit cleanup).

Spends RunPod money: run it under tools/capped_run.py with --output-dir set to OUTPUT_DIR.

    op run --env-file .env.tpl -- uv run python tools/capped_run.py --cap 0.30 --margin 0.02 \\
        --output-dir RUN --no-project --retries 0 -- uv run python \\
        docs/research/0029-character-reference-continuity/trial/sw4_pod_keyframes.py RUN \\
        --start-index 684 --limit 27
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types

sys.path.insert(0, os.getcwd())
import config  # noqa: E402
import decode  # noqa: E402
import manifest  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir")
    ap.add_argument("--start-index", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-clip", action="store_true", help="stop after the keyframes")
    args = ap.parse_args()
    config.load_env()

    prompts, _ = manifest.load_shots(args.output_dir)
    if args.start_index:
        prompts = [p for p in prompts if p["index"] >= args.start_index]
    if args.limit:
        prompts = prompts[:args.limit]
    ns = types.SimpleNamespace(output_dir=args.output_dir, keep_pod=False, style_prefix=None,
                               keyframe_fal_cap=0.0, keyframe_generator="pod")
    s = decode._create_keyframe_strategy(ns)
    t0 = time.time()
    s.prepare(prompts)
    report = {"prepare_s": round(time.time() - t0, 1), "keyframes": sorted(s._keyframes),
              "fallbacks": {k: v.value for k, v in s.fallbacks.items()}}
    print(json.dumps(report))
    if s._keyframes and not args.no_clip:
        idx = min(s._keyframes)
        entry = next(p for p in prompts if p["index"] == idx)
        t1 = time.time()
        clips_dir = manifest.clips_dir(args.output_dir, s.name)
        os.makedirs(clips_dir, exist_ok=True)   # run_decode does this; the first pod run lacked it
        res = s.generate(s.format_prompt(entry), clips_dir, idx,
                         entry["duration_s"], seed=idx, entry=entry)
        report.update(clip_shot=idx, clip_s=round(time.time() - t1, 1),
                      clip_paths=[r.path for r in res or []])
    with open(os.path.join(args.output_dir, "pod_validation.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report))
    if not report.get("clip_paths") and not args.no_clip:
        sys.exit(1)


if __name__ == "__main__":
    main()
