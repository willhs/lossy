# ruff: noqa: E501  -- the rubric is prose sent verbatim to the judge
"""Blind-ish vision-model scoring of the pilot clips (research 0025).

Per shot, one Gemini call sees: the original film keyframe, the shot's text
description, and every model's clip as inline video under a shuffled letter
(seeded per shot, so reruns reproduce the same blinding). The judge never sees
model names. It scores each clip 1-5 on a fixed rubric; letters are mapped
back to models only after the response is parsed.

    uv run python tools/model_pilot/score.py output/pilot0025 ltx23 ltx25-base k6
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import config  # noqa: E402
from encode import _gemini_client  # noqa: E402

JUDGE_MODEL = "gemini-2.5-flash"
# Paid-tier list price per 1M tokens (input incl. video, output) for the spend log.
PRICE_IN, PRICE_OUT = 0.30, 2.50

RUBRIC = """You are judging AI-generated video clips that try to reconstruct one shot of a film from a text description.
You get: (1) a still frame from the ORIGINAL shot, (2) the text description the generators were given, (3) several candidate clips labelled with letters.
Score EVERY candidate independently, integers 1-5 (5 best):
- identity: are the characters/creatures/vehicles the right kind and look consistent with the original frame and description? A robot must look mechanical, not a person in a costume; a furry creature must not be a human in a hood; the right number of figures; no two figures blended into one; identity stable through the clip.
- prompt_adherence: does the clip show the described setting, action, shot type, camera move and lighting?
- artefacts: 5 = clean; deduct for morphing, melting or extra limbs, garbled faces, flicker, baked-in letterbox bars, on-screen text or watermarks, frozen frames.
- motion: natural, physically plausible and appropriately energetic for the description. 1 = frozen slideshow or chaotic warping.
- overall: how good is this clip as a stand-in for the original shot in a full-length reconstruction?
Judge only what you see. Do not reward a clip for resolution alone. Be strict and use the full range.
Return JSON: {"scores": {"<letter>": {"identity": n, "prompt_adherence": n, "artefacts": n, "motion": n, "overall": n, "note": "<=20 words"}}, "best": "<letter>"}"""

DIMS = ("identity", "prompt_adherence", "artefacts", "motion", "overall")


def _desc(shot: dict) -> str:
    d = shot["description"]
    return "\n".join(f"{k}: {d[k]}" for k in ("shot_type", "camera_movement", "subjects", "action",
                                               "setting", "lighting") if d.get(k))


def score_shot(client, root: str, shot: dict, models: list[str]) -> dict | None:
    from google.genai import types

    idx = shot["index"]
    clips = {m: os.path.join(root, "clips", m, f"{idx:04d}.mp4") for m in models}
    clips = {m: p for m, p in clips.items() if os.path.exists(p)}
    if len(clips) < 2:
        return None
    order = sorted(clips)
    random.Random(idx).shuffle(order)
    letters = {chr(65 + i): m for i, m in enumerate(order)}

    parts = [types.Part.from_text(text=RUBRIC)]
    if shot.get("keyframe"):
        with open(os.path.join(root, shot["keyframe"]), "rb") as f:
            parts += [types.Part.from_text(text="ORIGINAL FRAME:"),
                      types.Part.from_bytes(data=f.read(), mime_type="image/jpeg")]
    parts.append(types.Part.from_text(text=f"DESCRIPTION:\n{_desc(shot)}"))
    for letter, m in letters.items():
        with open(clips[m], "rb") as f:
            parts += [types.Part.from_text(text=f"CANDIDATE {letter}:"),
                      types.Part.from_bytes(data=f.read(), mime_type="video/mp4")]

    for attempt in range(6):
        try:
            resp = client.models.generate_content(
                model=JUDGE_MODEL, contents=parts,
                config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.0),
            )
            break
        except Exception as e:  # 429 per-minute quota: back off and retry
            if "429" not in str(e) or attempt == 5:
                raise
            time.sleep(20 * (attempt + 1))
    data = json.loads(resp.text)
    u = resp.usage_metadata
    cost = ((u.prompt_token_count or 0) * PRICE_IN + (u.candidates_token_count or 0) * PRICE_OUT
            + (getattr(u, "thoughts_token_count", 0) or 0) * PRICE_OUT) / 1e6
    by_model = {letters[k]: v for k, v in data.get("scores", {}).items() if k in letters}
    return {"index": idx, "category": shot["category"], "letters": letters,
            "scores": by_model, "best": letters.get(data.get("best", ""), None), "cost_usd": cost}


def summarise(results: list[dict], models: list[str]) -> dict:
    summary = {}
    for m in models:
        rows = [r["scores"][m] for r in results if m in r["scores"]]
        if not rows:
            continue
        summary[m] = {d: round(sum(float(x.get(d, 0)) for x in rows) / len(rows), 2) for d in DIMS}
        summary[m]["n"] = len(rows)
        summary[m]["wins"] = sum(1 for r in results if r["best"] == m)
    return summary


def main():
    global JUDGE_MODEL
    ap = argparse.ArgumentParser(description="Blind vision-model scoring of pilot clips")
    ap.add_argument("root")
    ap.add_argument("models", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--judge", default=JUDGE_MODEL)
    ap.add_argument("--only", type=int, nargs="*")
    a = ap.parse_args()
    JUDGE_MODEL = a.judge
    config.load_env()
    client = _gemini_client(os.environ["GEMINI_API_KEY"])
    bench = json.load(open(os.path.join(a.root, "benchmark.json")))
    out = a.out or os.path.join(a.root, "scores.json")
    results = []
    for shot in bench["shots"]:
        if a.only and shot["index"] not in a.only:
            continue
        try:
            r = score_shot(client, a.root, shot, a.models)
        except Exception as e:
            print(f"  {shot['index']}: judge failed: {e!r:.200}")
            continue
        if r:
            results.append(r)
            print(f"  {r['index']:5d} best={r['best']:14s} "
                  + " ".join(f"{m}={r['scores'].get(m, {}).get('overall', '-')}" for m in a.models))
    summary = summarise(results, a.models)
    total = sum(r["cost_usd"] for r in results)
    json.dump({"judge": JUDGE_MODEL, "summary": summary, "judge_cost_usd": round(total, 4),
               "results": results}, open(out, "w"), indent=1)
    print(json.dumps(summary, indent=1))
    print(f"judge cost ${total:.4f} -> {out}")


if __name__ == "__main__":
    main()
