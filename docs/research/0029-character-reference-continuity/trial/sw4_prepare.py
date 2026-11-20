"""Draw the K arm's sheets and keyframes for a window of shots, with no pod (research 0029, SW4 paid check).

Runs exactly what `decode.py --strategy runpod-ltx25-keyframe` does in `prepare()`, before
any pod exists, so the sheets and first frames can be looked at BEFORE RunPod is paid for.
The files land in the decode output dir's `characters/` and `keyframes/` caches (keyed on
prompt, generator and sheets), so the real run reuses them instead of paying fal twice;
`keyframes/cost.jsonl` carries the spend into `--keyframe-fal-cap`.

Spends fal money (sheets <= ~$0.06, keyframes <= ~$0.03 each; capped by --keyframe-fal-cap).
Needs FAL_KEY and LOSSY_FACE_MODELS (default models/face).

    op run --env-file .env.tpl -- uv run python \
        docs/research/0029-character-reference-continuity/trial/sw4_prepare.py \
        output/sw4_v2 --start-index 684 --limit 30 --style-prefix "..." --keyframe-fal-cap 0.9
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.getcwd())
import config  # noqa: E402
import manifest  # noqa: E402
from decode import build_character_shot_map  # noqa: E402
from strategies_video import RunPodLtx25KeyframeStrategy  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir")
    ap.add_argument("--start-index", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--style-prefix", default=None)
    ap.add_argument("--keyframe-fal-cap", type=float, default=1.0)
    args = ap.parse_args()
    config.load_env()

    prompts, _ = manifest.load_shots(args.output_dir)
    if args.start_index:
        prompts = [p for p in prompts if p["index"] >= args.start_index]
    if args.limit:
        prompts = prompts[:args.limit]
    with open(manifest.characters_path(args.output_dir)) as f:
        characters_data = json.load(f)

    # Same construction as decode.dry_run_keyframes: skip __init__, which creates a RunPod session.
    s = object.__new__(RunPodLtx25KeyframeStrategy)
    s._init_character_identity(build_character_shot_map(characters_data), characters_data,
                               args.style_prefix or manifest.load_style_prefix(args.output_dir))
    s._output_dir = args.output_dir
    s._generator = None
    s._embedder = None
    s._fal_cap_usd = args.keyframe_fal_cap
    s._keyframes = {}
    s.fallbacks = {}
    s.prepare(prompts)
    print(json.dumps({"keyframes": len(s._keyframes), "fallbacks": {k: v.value for k, v in s.fallbacks.items()}}))


if __name__ == "__main__":
    main()
