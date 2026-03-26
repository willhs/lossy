#!/usr/bin/env python3
"""Evaluate reconstruction quality by re-encoding output and comparing to original."""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import google.genai as genai
from google.genai import types as genai_types

from encode import (
    SYSTEM_PROMPT,
    aggregate_shot_audio,
    extract_audio,
    frames_for_duration,
    run_yamnet,
)

# Pricing for cost tracking (gemini-3.1-flash-lite-preview)
GEMINI_INPUT_COST = 0.075 / 1_000_000
GEMINI_OUTPUT_COST = 0.30 / 1_000_000

# Description fields and their comparison types
CATEGORICAL_FIELDS = ["shot_type", "camera_movement"]
TEXT_FIELDS = ["subjects", "action", "lighting", "color_palette", "mood", "setting", "sound"]

SIMILARITY_PROMPT = """You are evaluating how well a reconstructed video matches the original.
Compare these two descriptions of the same field and rate their similarity from 0.0 to 1.0.

- 1.0 = semantically identical (same meaning, possibly different wording)
- 0.7-0.9 = mostly similar (same core content, minor differences)
- 0.4-0.6 = partially similar (some overlap but notable differences)
- 0.1-0.3 = mostly different (different content or meaning)
- 0.0 = completely different or unrelated

Respond with ONLY a JSON object: {"score": <float>, "reason": "<brief explanation>"}"""


def load_env():
    """Load .env file if present."""
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


def load_json(path: str) -> dict | list:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Phase 1: Keyframe extraction
# ---------------------------------------------------------------------------


def extract_eval_keyframes(video_path: str, scenes: list[dict], output_dir: str):
    """Extract keyframes from reconstructed video at original shot timecodes.

    Uses the same frame count and spacing logic as encode stage1, but takes
    plain scene dicts (with start_s/end_s) instead of PySceneDetect objects.
    """
    os.makedirs(output_dir, exist_ok=True)

    for scene in scenes:
        idx = scene["index"]
        start_s = scene["start_s"]
        duration_s = scene["duration_s"]
        n_frames = frames_for_duration(duration_s)
        step = duration_s / (n_frames + 1)

        for j in range(n_frames):
            timestamp = start_s + step * (j + 1)
            out_path = os.path.join(output_dir, f"{idx + 1:04d}-{j + 1:02d}.jpg")

            if os.path.exists(out_path):
                continue

            subprocess.run(
                [
                    "ffmpeg", "-ss", f"{timestamp:.3f}",
                    "-i", video_path,
                    "-vframes", "1",
                    "-vf", "scale='if(gt(iw,ih),512,-2)':'if(gt(ih,iw),512,-2)'",
                    "-q:v", "2", "-y", out_path,
                ],
                capture_output=True,
            )


# ---------------------------------------------------------------------------
# Phase 2: Re-encoding
# ---------------------------------------------------------------------------


def reencode_descriptions(
    scenes: list[dict], keyframes_dir: str
) -> tuple[list[dict], dict]:
    """Run Gemini vision on eval keyframes to produce re-encoded descriptions.

    Returns (eval_prompts, cost_info) where eval_prompts is a list of dicts
    matching the prompts.json structure, and cost_info tracks token usage.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not set")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    eval_prompts = []
    cost = {
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "cost_estimate": 0.0,
    }

    for scene in scenes:
        idx = scene["index"]
        duration_s = scene["duration_s"]
        n_frames = frames_for_duration(duration_s)

        # Load keyframe images
        parts = []
        for j in range(n_frames):
            img_path = os.path.join(keyframes_dir, f"{idx + 1:04d}-{j + 1:02d}.jpg")
            if not os.path.exists(img_path):
                continue
            with open(img_path, "rb") as f:
                parts.append(genai_types.Part.from_bytes(data=f.read(), mime_type="image/jpeg"))

        if not parts:
            print(f"  Warning: no keyframes for shot {idx}, skipping")
            eval_prompts.append({"index": idx, "description": None})
            continue

        # Context text (minimal -- no camera/audio hints for unbiased eval)
        context = f"Shot {idx + 1} of a film. Duration: {duration_s:.1f}s. {n_frames} frames shown."
        parts.append(genai_types.Part.from_text(text=context))

        try:
            response = client.models.generate_content(
                model="gemini-3.1-flash-lite-preview",
                contents=[genai_types.Content(role="user", parts=parts)],
                config=genai_types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.3,
                    response_mime_type="application/json",
                ),
            )
            description = json.loads(response.text.strip())

            # Track cost
            usage = response.usage_metadata
            in_tok = usage.prompt_token_count or 0
            out_tok = usage.candidates_token_count or 0
            cost["total_input_tokens"] += in_tok
            cost["total_output_tokens"] += out_tok
            cost["cost_estimate"] += in_tok * GEMINI_INPUT_COST + out_tok * GEMINI_OUTPUT_COST

            eval_prompts.append({
                "index": idx,
                "start_s": scene["start_s"],
                "end_s": scene["end_s"],
                "duration_s": duration_s,
                "description": description,
            })
            print(f"  Shot {idx}: described (${cost['cost_estimate']:.4f} total)")

        except Exception as e:
            print(f"  Shot {idx}: Gemini error: {e}")
            eval_prompts.append({"index": idx, "description": None})

    return eval_prompts, cost


def reencode_audio(
    video_path: str, scenes: list[dict], eval_dir: str
) -> dict[str, dict] | None:
    """Run YAMNet on reconstructed audio to produce audio classifications.

    Returns dict mapping shot index (as string) to {bucket, labels}, or None
    if audio extraction fails.
    """
    try:
        wav_path = extract_audio(video_path, eval_dir)
    except Exception as e:
        print(f"  Audio extraction failed: {e}")
        return None

    try:
        scores, class_names = run_yamnet(wav_path, eval_dir)
    except Exception as e:
        print(f"  YAMNet classification failed: {e}")
        return None

    audio_labels = {}
    for scene in scenes:
        idx = scene["index"]
        label = aggregate_shot_audio(scores, class_names, scene["start_s"], scene["end_s"])
        audio_labels[str(idx)] = label

    return audio_labels


# ---------------------------------------------------------------------------
# Phase 3: Comparison engine
# ---------------------------------------------------------------------------


def score_text_similarity(
    original: str, reconstructed: str, field_name: str, client
) -> tuple[float, str]:
    """Use Gemini to score similarity between two text descriptions.

    Returns (score, reason) where score is 0.0-1.0.
    """
    if not original and not reconstructed:
        return 1.0, "both empty"
    if not original or not reconstructed:
        return 0.0, "one is empty"

    prompt = f"Field: {field_name}\nOriginal: {original}\nReconstructed: {reconstructed}"

    try:
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=[genai_types.Content(role="user", parts=[genai_types.Part.from_text(text=prompt)])],
            config=genai_types.GenerateContentConfig(
                system_instruction=SIMILARITY_PROMPT,
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )
        result = json.loads(response.text.strip())
        return float(result["score"]), result.get("reason", "")
    except Exception as e:
        return 0.0, f"error: {e}"


def compare_categorical(original: str, reconstructed: str) -> float:
    """Exact match comparison for categorical fields. Returns 0.0 or 1.0."""
    if not original and not reconstructed:
        return 1.0
    return 1.0 if (original or "").lower().strip() == (reconstructed or "").lower().strip() else 0.0


def jaccard_similarity(a: list[str], b: list[str]) -> float:
    """Jaccard similarity between two label lists."""
    set_a = set(s.lower() for s in a)
    set_b = set(s.lower() for s in b)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def compare_audio(original: dict | None, reconstructed: dict | None) -> dict:
    """Compare audio classifications for a shot.

    Returns {"bucket_match": float, "label_similarity": float}.
    """
    if original is None and reconstructed is None:
        return {"bucket_match": 1.0, "label_similarity": 1.0}
    if original is None or reconstructed is None:
        return {"bucket_match": 0.0, "label_similarity": 0.0}

    bucket_match = 1.0 if original.get("bucket") == reconstructed.get("bucket") else 0.0
    label_sim = jaccard_similarity(original.get("labels", []), reconstructed.get("labels", []))
    return {"bucket_match": bucket_match, "label_similarity": label_sim}


def compare_shot(
    original_prompt: dict,
    eval_prompt: dict,
    original_audio: dict | None,
    eval_audio: dict | None,
    client,
) -> dict:
    """Compare a single shot across all dimensions.

    Returns dict with per-field scores and an overall shot score.
    """
    orig_desc = original_prompt.get("description", {}) or {}
    eval_desc = eval_prompt.get("description", {}) or {}

    scores = {}

    # Categorical fields
    for field in CATEGORICAL_FIELDS:
        scores[field] = compare_categorical(
            orig_desc.get(field, ""), eval_desc.get(field, "")
        )

    # Text fields -- Gemini similarity
    for field in TEXT_FIELDS:
        score, reason = score_text_similarity(
            orig_desc.get(field, ""), eval_desc.get(field, ""), field, client
        )
        scores[field] = score

    # Audio
    audio_scores = compare_audio(original_audio, eval_audio)
    scores["audio_bucket"] = audio_scores["bucket_match"]
    scores["audio_labels"] = audio_scores["label_similarity"]

    # Overall: weighted average
    video_fields = CATEGORICAL_FIELDS + TEXT_FIELDS
    video_avg = sum(scores[f] for f in video_fields) / len(video_fields)
    audio_avg = (scores["audio_bucket"] + scores["audio_labels"]) / 2

    # 70% video, 30% audio (video is the primary output)
    scores["video_avg"] = video_avg
    scores["audio_avg"] = audio_avg
    scores["overall"] = 0.7 * video_avg + 0.3 * audio_avg

    return scores


def compare_all(
    original_prompts: list[dict],
    eval_prompts: list[dict],
    original_audio_labels: dict | None,
    eval_audio_labels: dict | None,
) -> dict:
    """Compare all shots and compute aggregate scores.

    Returns {"shots": [...], "aggregate": {...}}.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key) if api_key else None

    shot_results = []
    for orig, evl in zip(original_prompts, eval_prompts):
        idx = orig["index"]
        if evl.get("description") is None:
            shot_results.append({"index": idx, "skipped": True})
            continue

        orig_audio = original_audio_labels.get(str(idx)) if original_audio_labels else None
        eval_audio = eval_audio_labels.get(str(idx)) if eval_audio_labels else None

        scores = compare_shot(orig, evl, orig_audio, eval_audio, client)
        shot_results.append({"index": idx, **scores})

    # Aggregate -- average across non-skipped shots
    scored_shots = [s for s in shot_results if not s.get("skipped")]
    all_fields = CATEGORICAL_FIELDS + TEXT_FIELDS + ["audio_bucket", "audio_labels", "video_avg", "audio_avg", "overall"]
    aggregate = {}
    for field in all_fields:
        values = [s[field] for s in scored_shots if field in s]
        aggregate[field] = sum(values) / len(values) if values else 0.0

    return {"shots": shot_results, "aggregate": aggregate}


# ---------------------------------------------------------------------------
# Phase 4: Report
# ---------------------------------------------------------------------------


def print_summary(results: dict):
    """Print a compact score summary to stdout."""
    agg = results["aggregate"]
    print("\n=== Eval Summary ===")
    print(f"  Overall:  {agg.get('overall', 0):.2f}")
    print(f"  Video:    {agg.get('video_avg', 0):.2f}")
    print(f"  Audio:    {agg.get('audio_avg', 0):.2f}")
    print(f"  Shots:    {len(results['shots'])} ({sum(1 for s in results['shots'] if not s.get('skipped'))} scored)")

    # Worst shots
    scored = [s for s in results["shots"] if not s.get("skipped")]
    worst = sorted(scored, key=lambda s: s.get("overall", 0))[:3]
    if worst:
        print("\n  Worst shots:")
        for s in worst:
            print(f"    Shot {s['index']}: {s.get('overall', 0):.2f}")


def generate_report(
    output_dir: str,
    strategy: str,
    results: dict,
    encode_cost: dict,
    report_dir: str,
) -> str:
    """Write a markdown evaluation report.

    Returns the path to the written report file.
    """
    agg = results["aggregate"]
    shots = results["shots"]
    scored_shots = [s for s in shots if not s.get("skipped")]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    date_slug = datetime.now().strftime("%Y%m%d-%H%M")
    source_name = Path(output_dir).name

    report_base = Path(report_dir)
    report_base.mkdir(parents=True, exist_ok=True)

    report_name = f"eval-{source_name}-{strategy}-{date_slug}.md"
    report_path = report_base / report_name

    # Sort shots by overall score for worst-performing list
    worst = sorted(scored_shots, key=lambda s: s.get("overall", 0))

    lines = []
    lines.append(f"# Eval: {source_name} / {strategy}")
    lines.append("")
    lines.append(f"**Date**: {timestamp}")
    lines.append(f"**Overall Score**: {agg.get('overall', 0):.3f}")
    lines.append(f"**Video Score**: {agg.get('video_avg', 0):.3f}")
    lines.append(f"**Audio Score**: {agg.get('audio_avg', 0):.3f}")
    lines.append(f"**Shots**: {len(scored_shots)} scored, {len(shots) - len(scored_shots)} skipped")
    lines.append(f"**Eval Cost**: ${encode_cost.get('cost_estimate', 0):.4f} ({encode_cost.get('total_input_tokens', 0)} in / {encode_cost.get('total_output_tokens', 0)} out tokens)")
    lines.append("")

    # Per-dimension scores
    lines.append("## Scores by Dimension")
    lines.append("")
    lines.append("| Dimension | Score |")
    lines.append("|-----------|-------|")
    for field in CATEGORICAL_FIELDS + TEXT_FIELDS:
        lines.append(f"| {field} | {agg.get(field, 0):.3f} |")
    lines.append(f"| audio_bucket | {agg.get('audio_bucket', 0):.3f} |")
    lines.append(f"| audio_labels | {agg.get('audio_labels', 0):.3f} |")
    lines.append("")

    # Worst performing shots
    lines.append("## Worst Performing Shots")
    lines.append("")
    n_worst = min(5, len(worst))
    for s in worst[:n_worst]:
        idx = s["index"]
        lines.append(f"### Shot {idx} (overall: {s.get('overall', 0):.3f})")
        lines.append("")
        lines.append("| Field | Score |")
        lines.append("|-------|-------|")
        for field in CATEGORICAL_FIELDS + TEXT_FIELDS + ["audio_bucket", "audio_labels"]:
            if field in s:
                lines.append(f"| {field} | {s[field]:.3f} |")
        lines.append("")

    # Full per-shot table
    lines.append("## All Shots")
    lines.append("")
    lines.append("| Shot | Overall | Video | Audio |")
    lines.append("|------|---------|-------|-------|")
    for s in shots:
        if s.get("skipped"):
            lines.append(f"| {s['index']} | skipped | - | - |")
        else:
            lines.append(f"| {s['index']} | {s.get('overall', 0):.3f} | {s.get('video_avg', 0):.3f} | {s.get('audio_avg', 0):.3f} |")
    lines.append("")

    report_path.write_text("\n".join(lines))
    return str(report_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Evaluate reconstruction quality")
    parser.add_argument("output_dir", help="Path to the pipeline output directory")
    parser.add_argument("--strategy", required=True, help="Strategy name (e.g. fal-seedance)")
    parser.add_argument("--report-dir", default="docs/research", help="Directory for markdown report")
    args = parser.parse_args()

    load_env()

    output_dir = Path(args.output_dir)
    manifest = load_json(output_dir / "manifest.json")
    prompts = load_json(output_dir / "prompts.json")
    reconstructed_path = output_dir / f"reconstructed_{args.strategy}.mp4"

    if not reconstructed_path.exists():
        print(f"Error: reconstructed video not found: {reconstructed_path}")
        sys.exit(1)

    # Load original audio labels if available
    audio_labels_path = output_dir / "audio_labels.json"
    original_audio_labels = load_json(audio_labels_path) if audio_labels_path.exists() else None

    eval_dir = output_dir / "eval" / args.strategy
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Scope scenes to only those that have prompts (manifest may cover more)
    prompt_indices = {p["index"] for p in prompts}
    scenes = [s for s in manifest["scenes"] if s["index"] in prompt_indices]
    print(f"Evaluating {len(scenes)} shots (of {len(manifest['scenes'])} in manifest)")

    # Phase 1: Extract keyframes
    print("Extracting keyframes from reconstructed video...")
    eval_keyframes_dir = eval_dir / "keyframes"
    extract_eval_keyframes(str(reconstructed_path), scenes, str(eval_keyframes_dir))

    # Phase 2: Re-encode
    print("Re-encoding with Gemini vision...")
    eval_prompts, encode_cost = reencode_descriptions(scenes, str(eval_keyframes_dir))

    print("Re-classifying audio with YAMNet...")
    eval_audio_labels = reencode_audio(str(reconstructed_path), scenes, str(eval_dir))

    # Phase 3: Compare
    print("Comparing original vs re-encoded...")
    results = compare_all(prompts, eval_prompts, original_audio_labels, eval_audio_labels)

    # Save raw eval data
    eval_results = {
        "strategy": args.strategy,
        "timestamp": datetime.now().isoformat(),
        "encode_cost": encode_cost,
        "eval_prompts": eval_prompts,
        "eval_audio_labels": eval_audio_labels,
        "results": results,
    }
    with open(eval_dir / "eval_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)

    # Phase 4: Report
    report_path = generate_report(
        output_dir=str(output_dir),
        strategy=args.strategy,
        results=results,
        encode_cost=encode_cost,
        report_dir=args.report_dir,
    )
    print_summary(results)
    print(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    main()
