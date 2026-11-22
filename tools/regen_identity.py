#!/usr/bin/env python3
"""Regenerate only the identity `description` of each character in characters.json.

Stage 3 rewrites the whole registry (roster, names, shot lists), so re-running
it to pick up a new IDENTITY_DESCRIPTION_SPEC would also reshuffle who appears
in which shot. This tool keeps everything but `description`: same roster, same
names, same shot lists, same hand-set fields (keep_name). It reuses the exact
prompt shape stage 3 uses for text-match-only stubs, so the spec is the one
definition in encode.py.

    uv run python tools/regen_identity.py OUTPUT_DIR [--backup-dir DIR] [--dry-run]

Before overwriting, characters.json is copied into the backup dir (default
OUTPUT_DIR/characters.pre-photo-YYYYMMDD/). Nothing else in OUTPUT_DIR is
touched, and shots.json is only read, so manifest.encode_fingerprint does not
change.
"""
import argparse
import datetime
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import manifest  # noqa: E402

MAX_SUBJECTS = 20


def sample_shots(shots: list[int], limit: int = MAX_SUBJECTS) -> list[int]:
    """Up to `limit` shots spread evenly across the list (not just the first N)."""
    shots = sorted(shots)
    if len(shots) <= limit:
        return shots
    step = (len(shots) - 1) / (limit - 1)
    return [shots[round(i * step)] for i in range(limit)]


def build_prompt(char: dict, subjects_by_shot: dict[int, str], identity_spec: str) -> str | None:
    """The description prompt for one character, or None if no shot has subjects."""
    lines = [
        f"Shot {s}: {subjects_by_shot[s]}"
        for s in sample_shots(char.get("shots", []))
        if subjects_by_shot.get(s)
    ]
    if not lines:
        return None
    return (
        f"Character: {char['display_name']}\n"
        "Appears in these shots:\n" + "\n".join(lines) + "\n\n"
        f"Write {identity_spec}"
    )


def regenerate(client, types, model: str, characters_data: dict,
               subjects_by_shot: dict[int, str], identity_spec: str) -> dict:
    """Return a copy of characters_data with every description regenerated."""
    out = json.loads(json.dumps(characters_data))
    for char in out["characters"]:
        prompt = build_prompt(char, subjects_by_shot, identity_spec)
        if prompt is None:
            print(f"  {char['name']}: no subjects, description kept")
            continue
        resp = client.models.generate_content(
            model=model,
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
            config=types.GenerateContentConfig(temperature=0.3),
        )
        new = (resp.text or "").strip()
        if not new:
            print(f"  {char['name']}: empty response, description kept")
            continue
        char["description"] = new
        print(f"  {char['name']}: done")
    return out


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("output_dir")
    ap.add_argument("--backup-dir", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the prompt for each character; no Gemini calls, no writes")
    args = ap.parse_args()

    import config
    config.load_env()
    import encode

    chars_path = manifest.characters_path(args.output_dir)
    with open(chars_path) as f:
        data = json.load(f)
    with open(manifest.shots_path(args.output_dir)) as f:
        raw = json.load(f)
    shots = raw["shots"] if isinstance(raw, dict) else raw
    subjects = {e["index"]: e.get("description", {}).get("subjects", "") for e in shots}

    if args.dry_run:
        for c in data["characters"]:
            p = build_prompt(c, subjects, encode.IDENTITY_DESCRIPTION_SPEC)
            print(f"--- {c['name']} ({len(c['shots'])} shots)\n{p}\n")
        return

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key and encode.GEMINI_BACKEND != "vertex":
        sys.exit("GEMINI_API_KEY not set")
    from google.genai import types
    client = encode._gemini_client(api_key or "")

    backup = args.backup_dir or os.path.join(
        args.output_dir, "characters.pre-photo-" + datetime.date.today().strftime("%Y%m%d"))
    os.makedirs(backup, exist_ok=True)
    shutil.copy2(chars_path, os.path.join(backup, os.path.basename(chars_path)))
    print(f"Backed up {chars_path} -> {backup}")

    print(f"Regenerating {len(data['characters'])} descriptions with {encode.ENCODE_MODEL}")
    new = regenerate(client, types, encode.ENCODE_MODEL, data, subjects,
                     encode.IDENTITY_DESCRIPTION_SPEC)
    with open(chars_path, "w") as f:
        json.dump(new, f, indent=2)
    print(f"Wrote {chars_path}")


if __name__ == "__main__":
    main()
