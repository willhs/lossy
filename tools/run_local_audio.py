"""Run local audio generation (MMAudio SFX and/or MusicGen music) across clips.

Loads each model once and reuses it across all clips. Output dirs + progress
files match what stitch expects (--audio-strategy mmaudio-local /
--music-strategy musicgen-local).
"""
import argparse
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from decode import load_env, run_audio  # noqa: E402
from local_audio import LocalMMAudioStrategy, LocalMusicGenStrategy  # noqa: E402


def ns(clip):
    return types.SimpleNamespace(output_dir=f"output/{clip}", start_index=None, limit=None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--kind", choices=["sfx", "music"], required=True)
    ap.add_argument("--mmaudio-model", default="medium_44k")
    args = ap.parse_args()
    load_env()

    if args.kind == "sfx":
        strat = LocalMMAudioStrategy(model_name=args.mmaudio_model)
    else:
        strat = LocalMusicGenStrategy()

    for clip in args.clips:
        print(f"\n===== {args.kind.upper()} :: {clip} =====", flush=True)
        run_audio(ns(clip), strat)


if __name__ == "__main__":
    main()
