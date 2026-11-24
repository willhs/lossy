"""Regenerate the speech lines whose voice changed with the recast (research 0030, change 1).

Works on a COPY of the run dir (RUN_DIR), then the caller syncs it back. Targets are:
  * every line whose (new) speaker is a recast character, and
  * every line that already has a clip whose voice now differs (old speaker+map vs new).
Lines of non-recast characters that have no clip yet are left for the full run's speech pass.

    op run --env-file .env.tpl -- uv run python docs/research/0030-sw4-full-run-preflight/regen_recast_speech.py \
        RUN_DIR OLD_SPEAKERS.json OLD_VOICE_MAP.json [--go]

Without --go it only prints the plan and the estimated cost. Replaced clips are moved to
RUN_DIR/speech_replaced/ (not deleted).
"""
import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import decode  # noqa: E402
import manifest  # noqa: E402
from strategies_audio import SpeechStrategy  # noqa: E402

FALLBACK = "Roger"  # decode.py --speech-voice default
COST_PER_CHAR = SpeechStrategy.COST_PER_1K_CHARS / 1000


def voice(speakers, vmap, i):
    ch = speakers.get(i)
    return (vmap.get(ch) or FALLBACK) if ch else FALLBACK


def plan(run_dir, old_speakers_path, old_map_path):
    _, dialog = manifest.load_shots(run_dir)
    new_sp, new_vm = manifest.load_speakers(run_dir), manifest.load_voice_map(run_dir)
    # the run's registry renamed Obi-Wan (obiwan_kenobi -> obi_wan_ben_kenobi): a rename, not a recast
    alias = {"obiwan_kenobi": "obi_wan_ben_kenobi"}
    old_sp = {int(k): alias.get(v["character"], v["character"])
              for k, v in json.load(open(old_speakers_path))["assignments"].items()}
    old_vm = {alias.get(k, k): v for k, v in json.load(open(old_map_path)).items()}
    recast = {c for c in set(old_vm) | set(new_vm) if old_vm.get(c) != new_vm.get(c)}
    progress = json.load(open(manifest.speech_progress_path(run_dir)))
    have = set(progress["completed"])
    targets = sorted(
        i for i in range(len(dialog))
        if new_sp.get(i) in recast or (i in have and voice(old_sp, old_vm, i) != voice(new_sp, new_vm, i))
    )
    return dialog, recast, have, targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("old_speakers")
    ap.add_argument("old_voice_map")
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args()

    dialog, recast, have, targets = plan(a.run_dir, a.old_speakers, a.old_voice_map)
    chars = sum(len(dialog[i]["text"]) for i in targets)
    print(f"recast characters: {sorted(recast)}")
    print(f"{len(targets)} target lines ({len([i for i in targets if i in have])} replace an existing clip), "
          f"{chars} chars, estimated ${chars * COST_PER_CHAR:.2f}")
    if not a.go:
        return

    progress_path = manifest.speech_progress_path(a.run_dir)
    progress = json.load(open(progress_path))
    speech_dir = manifest.speech_dir(a.run_dir)
    stale = os.path.join(a.run_dir, "speech_replaced")
    os.makedirs(stale, exist_ok=True)
    tset = set(targets)
    for i in targets:
        p = manifest.speech_clip_path(speech_dir, i)
        if os.path.exists(p):
            shutil.move(p, os.path.join(stale, os.path.basename(p)))
        progress["clips"].pop(str(i), None)
    progress["completed"] = [i for i in progress["completed"] if i not in tset]
    progress["failed"] = [i for i in progress["failed"] if i not in tset]

    # run_speech walks every line; mark the non-targets done for the duration of the run so it
    # skips them, then restore. `fake` lines never get a clip entry.
    real_done = list(progress["completed"])
    fake = [i for i in range(len(dialog)) if i not in tset and i not in set(real_done)]
    progress["completed"] = real_done + fake
    manifest.write_json_atomic(progress_path, progress)
    spent_before = progress["total_cost_estimate"]
    try:
        decode.run_speech(argparse.Namespace(output_dir=a.run_dir, start_index=None, limit=None),
                          SpeechStrategy(voice=FALLBACK))
    finally:
        progress = json.load(open(progress_path))
        fakeset = set(fake)
        progress["completed"] = [i for i in progress["completed"] if i not in fakeset]
        manifest.write_json_atomic(progress_path, progress)
    done = [i for i in targets if i in set(progress["completed"])]
    print(f"regenerated {len(done)}/{len(targets)}; failed: {[i for i in targets if i not in set(done)]}")
    print(f"spend this run: ${progress['total_cost_estimate'] - spent_before:.2f}")


if __name__ == "__main__":
    main()
