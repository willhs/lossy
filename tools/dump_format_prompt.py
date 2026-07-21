#!/usr/bin/env python3
"""Dump RunPodWan22Strategy.format_prompt() output for chosen shots without touching
a GPU/pod. Used to verify the CharacterIdentityMixin locked-identity prefix is
prepended verbatim and to preview what a misattributed character string would read
like, ahead of spending any RunPod time.

Usage: python tools/dump_format_prompt.py <output_dir> [shot_index ...]
"""
import json
import sys

import manifest
from decode import build_character_shot_map
from strategies_video import RunPodWan22Strategy


def load_strategy(output_dir: str) -> tuple[RunPodWan22Strategy, dict]:
    shots, _dialog = manifest.load_shots(output_dir)
    with open(manifest.characters_path(output_dir)) as f:
        characters_data = json.load(f)
    character_shot_map = build_character_shot_map(characters_data)
    strategy = RunPodWan22Strategy(
        output_dir=output_dir,
        character_shot_map=character_shot_map,
        characters_data=characters_data,
    )
    by_idx = {s["index"]: s for s in shots}
    return strategy, by_idx


def main():
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "output/sw_r2_leia_after"
    indices = [int(a) for a in sys.argv[2:]] or [2, 8, 19, 26]

    strategy, by_idx = load_strategy(output_dir)

    print(f"=== format_prompt() dump for {output_dir} ===\n")
    for idx in indices:
        entry = by_idx.get(idx)
        if entry is None:
            print(f"--- shot {idx}: not found ---\n")
            continue
        print(f"--- shot {idx} ---")
        print(strategy.format_prompt(entry))
        print()

    # Synthetic check: what would shots 6-7 read like if misattributed to
    # obiwan_kenobi instead of princess_leia_organa (the known 98a3154 regression)?
    with open(manifest.characters_path(output_dir)) as f:
        characters_data = json.load(f)
    shots, _dialog = manifest.load_shots(output_dir)
    by_idx_full = {s["index"]: s for s in shots}
    obiwan = next(
        (c for c in characters_data.get("characters", []) if c["name"] == "obiwan_kenobi"),
        None,
    )
    if obiwan:
        print("=== misattribution check: obiwan_kenobi mapped to shots 6-7 ===\n")
        synthetic_map = {6: ["obiwan_kenobi"], 7: ["obiwan_kenobi"]}
        synthetic_strategy = RunPodWan22Strategy(
            output_dir=output_dir,
            character_shot_map=synthetic_map,
            characters_data=characters_data,
        )
        for idx in (6, 7):
            entry = by_idx_full.get(idx)
            if entry is None:
                continue
            print(f"--- shot {idx} (as currently attributed in {output_dir}/characters.json) ---")
            print(synthetic_strategy.format_prompt(entry))
            print()


if __name__ == "__main__":
    main()
