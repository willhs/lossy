"""Anonymised judging pack for a blind vision-model pass (research 0025).

Fallback for score.py when the Gemini project is rate-limited. Per shot, one
JPEG: the ORIGINAL frame, then one row per candidate with 4 frames (at 10, 35,
60 and 85% of the clip) so motion and stability are visible. Candidates are
labelled only A/B/C in a per-shot seeded shuffle, and every tile is scaled to
the same size so output resolution does not identify the model.

The letter -> model key is written OUTSIDE the pack dir, so a judge given the
pack never sees it. score_blind.py maps the judge's JSON back afterwards.

    uv run python tools/model_pilot/blind_pack.py output/pilot0025 ltx23 ltx25-base k6
"""

from __future__ import annotations

import argparse
import json
import os
import random

from PIL import Image, ImageDraw, ImageFont

from sheets import _fit, _frame, _probe_duration  # same dir

TW, TH = 384, 216
FRACS = (0.10, 0.35, 0.60, 0.85)


def tile(img):
    import sheets
    sheets.TILE_W, sheets.TILE_H = TW, TH
    return _fit(img)


def build(root: str, models: list[str], seed: int = 1000) -> tuple[str, str]:
    bench = json.load(open(os.path.join(root, "benchmark.json")))
    pack = os.path.join(root, "blind_pack" if seed == 1000 else f"blind_pack_s{seed}")
    os.makedirs(pack, exist_ok=True)
    font = ImageFont.load_default()
    key, brief = {}, []
    for shot in bench["shots"]:
        idx = shot["index"]
        order = list(models)
        random.Random(seed + idx).shuffle(order)
        letters = {chr(65 + i): m for i, m in enumerate(order)}
        key[str(idx)] = letters
        rows = 1 + len(order)
        sheet = Image.new("RGB", (40 + 4 * TW, rows * (TH + 4)), (12, 12, 12))
        d = ImageDraw.Draw(sheet)
        orig = Image.open(os.path.join(root, shot["keyframe"])).convert("RGB")
        sheet.paste(tile(orig), (40, 0))
        d.text((4, TH // 2), "ORIG", fill=(255, 220, 120), font=font)
        for r, (letter, m) in enumerate(letters.items(), start=1):
            clip = os.path.join(root, "clips", m, f"{idx:04d}.mp4")
            dur = _probe_duration(clip)
            y = r * (TH + 4)
            d.text((14, y + TH // 2), letter, fill=(255, 255, 255), font=font)
            for c, f in enumerate(FRACS):
                sheet.paste(tile(_frame(clip, dur * f) if dur else None), (40 + c * TW, y))
        sheet.save(os.path.join(pack, f"shot_{idx:04d}.jpg"), quality=85)
        d_ = shot["description"]
        brief.append({"shot": idx, "image": f"shot_{idx:04d}.jpg", "candidates": sorted(letters),
                      "description": {k: d_.get(k) for k in ("shot_type", "camera_movement", "subjects",
                                                             "action", "setting", "lighting")}})
    with open(os.path.join(pack, "shots.json"), "w") as f:
        json.dump(brief, f, indent=1)
    # The key lives outside the pack on purpose: the judge only ever sees the pack.
    key_path = os.path.join(root, "blind_key.json" if seed == 1000 else f"blind_key_s{seed}.json")
    with open(key_path, "w") as f:
        json.dump(key, f, indent=1)
    return pack, key_path


def main():
    ap = argparse.ArgumentParser(description="Build an anonymised judging pack")
    ap.add_argument("root")
    ap.add_argument("models", nargs="+")
    ap.add_argument("--seed", type=int, default=1000)
    a = ap.parse_args()
    pack, key = build(a.root, a.models, a.seed)
    print("pack:", pack, "key (do not show judge):", key)


if __name__ == "__main__":
    main()
