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
# Audio clip judge
# ---------------------------------------------------------------------------


AUDIO_JUDGE_PROMPT = """You are evaluating AI-generated audio for a film reconstruction.
You will hear an audio clip and be given the intended sound description.

Rate the audio on two dimensions:
1. **relevance** (0.0-1.0): Does the audio match the description? Are the described sounds present?
2. **quality** (0.0-1.0): Is the audio clean and recognizable, or does it contain glitches, artifacts, distortion, or noise that makes it sound broken/synthetic?

Quality guide:
- 1.0 = Clean, natural, could pass as real audio
- 0.7-0.9 = Mostly clean, minor imperfections
- 0.4-0.6 = Noticeably synthetic but recognizable sounds
- 0.1-0.3 = Heavy artifacts, glitchy, mostly unrecognizable
- 0.0 = Pure noise/garbage

Respond with ONLY a JSON object: {"relevance": <float>, "quality": <float>, "notes": "<brief observation>"}"""


def judge_audio_clip(client, clip_path: str, sound_description: str) -> dict:
    """Send an audio clip to Gemini for quality/relevance judgment.

    Returns {"relevance": float, "quality": float, "notes": str} or
    {"error": str} on failure.
    """
    with open(clip_path, "rb") as f:
        audio_data = f.read()

    mime = "audio/flac" if clip_path.endswith(".flac") else "audio/mpeg"
    parts = [
        genai_types.Part.from_bytes(data=audio_data, mime_type=mime),
        genai_types.Part.from_text(text=f"Intended sound: {sound_description}"),
    ]

    try:
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=[genai_types.Content(role="user", parts=parts)],
            config=genai_types.GenerateContentConfig(
                system_instruction=AUDIO_JUDGE_PROMPT,
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )
        result = json.loads(response.text.strip())
        return {
            "relevance": float(result.get("relevance", 0)),
            "quality": float(result.get("quality", 0)),
            "notes": result.get("notes", ""),
        }
    except Exception as e:
        return {"error": str(e)}


def run_audio_clip_eval(
    output_dir: Path,
    audio_strategy: str,
    prompts: list[dict],
    sample_every: int | None,
    report_dir: str,
):
    """Evaluate individual audio clips for quality and relevance."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not set")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    audio_dir = output_dir / "audio" / audio_strategy

    if not audio_dir.exists():
        print(f"Error: audio directory not found: {audio_dir}")
        sys.exit(1)

    # Build list of clips to evaluate
    clips_to_eval = []
    for prompt in prompts:
        idx = prompt["index"]
        sound_desc = (prompt.get("description") or {}).get("sound", "")
        if not sound_desc:
            continue

        # Find the clip file (flac or mp3)
        for ext in [".flac", ".mp3"]:
            clip_path = audio_dir / f"{idx:04d}{ext}"
            if clip_path.exists():
                clips_to_eval.append((idx, str(clip_path), sound_desc))
                break

    if sample_every and sample_every > 1:
        clips_to_eval = clips_to_eval[::sample_every]

    print(f"Evaluating {len(clips_to_eval)} audio clips from {audio_strategy}...")

    results = []
    cost_tokens = {"input": 0, "output": 0}

    for i, (idx, clip_path, sound_desc) in enumerate(clips_to_eval):
        result = judge_audio_clip(client, clip_path, sound_desc)
        result["index"] = idx
        result["sound_description"] = sound_desc

        if "error" in result:
            print(f"  Shot {idx}: error -- {result['error']}")
        else:
            print(f"  Shot {idx}: relevance={result['relevance']:.1f} quality={result['quality']:.1f} -- {result['notes'][:80]}")

        results.append(result)

        # Progress every 50 clips
        if (i + 1) % 50 == 0:
            scored = [r for r in results if "error" not in r]
            if scored:
                avg_q = sum(r["quality"] for r in scored) / len(scored)
                avg_r = sum(r["relevance"] for r in scored) / len(scored)
                print(f"  ... {i + 1}/{len(clips_to_eval)} done (avg quality={avg_q:.2f}, relevance={avg_r:.2f})")

    # Compute aggregates
    scored = [r for r in results if "error" not in r]
    if not scored:
        print("No clips were successfully evaluated.")
        return

    avg_quality = sum(r["quality"] for r in scored) / len(scored)
    avg_relevance = sum(r["relevance"] for r in scored) / len(scored)

    # Quality distribution buckets
    buckets = {"good (>=0.7)": 0, "ok (0.4-0.7)": 0, "poor (<0.4)": 0}
    for r in scored:
        q = r["quality"]
        if q >= 0.7:
            buckets["good (>=0.7)"] += 1
        elif q >= 0.4:
            buckets["ok (0.4-0.7)"] += 1
        else:
            buckets["poor (<0.4)"] += 1

    # Find quality transition point (where does quality drop?)
    if len(scored) >= 10:
        window = max(5, len(scored) // 20)
        rolling_quality = []
        for i in range(len(scored) - window + 1):
            avg = sum(r["quality"] for r in scored[i:i + window]) / window
            rolling_quality.append((scored[i]["index"], avg))

    # Worst clips
    worst = sorted(scored, key=lambda r: r["quality"])[:10]
    # Best clips
    best = sorted(scored, key=lambda r: r["quality"], reverse=True)[:5]

    # Print summary
    print(f"\n=== Audio Clip Eval: {audio_strategy} ===")
    print(f"  Clips evaluated: {len(scored)} ({len(results) - len(scored)} errors)")
    print(f"  Avg quality:   {avg_quality:.2f}")
    print(f"  Avg relevance: {avg_relevance:.2f}")
    print(f"  Distribution:  {buckets['good (>=0.7)']} good, {buckets['ok (0.4-0.7)']} ok, {buckets['poor (<0.4)']} poor")
    print(f"\n  Best clips:")
    for r in best:
        print(f"    Shot {r['index']}: quality={r['quality']:.1f} relevance={r['relevance']:.1f} -- {r['notes'][:60]}")
    print(f"\n  Worst clips:")
    for r in worst[:5]:
        print(f"    Shot {r['index']}: quality={r['quality']:.1f} relevance={r['relevance']:.1f} -- {r['notes'][:60]}")

    # Write report
    report_path = _write_audio_clip_report(
        output_dir=str(output_dir),
        audio_strategy=audio_strategy,
        scored=scored,
        avg_quality=avg_quality,
        avg_relevance=avg_relevance,
        buckets=buckets,
        worst=worst,
        best=best,
        report_dir=report_dir,
    )

    # Save JSON results
    eval_dir = output_dir / "eval" / audio_strategy
    eval_dir.mkdir(parents=True, exist_ok=True)
    with open(eval_dir / "audio_clip_eval.json", "w") as f:
        json.dump({
            "audio_strategy": audio_strategy,
            "timestamp": datetime.now().isoformat(),
            "clips_evaluated": len(scored),
            "avg_quality": avg_quality,
            "avg_relevance": avg_relevance,
            "buckets": buckets,
            "results": results,
        }, f, indent=2)

    print(f"\nReport written to: {report_path}")


def _write_audio_clip_report(
    output_dir: str,
    audio_strategy: str,
    scored: list[dict],
    avg_quality: float,
    avg_relevance: float,
    buckets: dict,
    worst: list[dict],
    best: list[dict],
    report_dir: str,
) -> str:
    """Write markdown report for audio clip evaluation."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    date_slug = datetime.now().strftime("%Y%m%d-%H%M")
    source_name = Path(output_dir).name

    report_base = Path(report_dir)
    report_base.mkdir(parents=True, exist_ok=True)
    report_path = report_base / f"eval-audio-{source_name}-{audio_strategy}-{date_slug}.md"

    lines = []
    lines.append(f"# Audio Clip Eval: {source_name} / {audio_strategy}")
    lines.append("")
    lines.append(f"**Date**: {timestamp}")
    lines.append(f"**Clips evaluated**: {len(scored)}")
    lines.append(f"**Avg Quality**: {avg_quality:.3f}")
    lines.append(f"**Avg Relevance**: {avg_relevance:.3f}")
    lines.append("")

    # Distribution
    lines.append("## Quality Distribution")
    lines.append("")
    lines.append("| Bucket | Count | % |")
    lines.append("|--------|-------|---|")
    for label, count in buckets.items():
        pct = count / len(scored) * 100 if scored else 0
        lines.append(f"| {label} | {count} | {pct:.0f}% |")
    lines.append("")

    # Quality over time (by shot index)
    lines.append("## Quality Over Shot Index")
    lines.append("")
    lines.append("Shows how quality varies across the film (by shot index).")
    lines.append("")
    # Group into ranges
    if scored:
        max_idx = max(r["index"] for r in scored)
        range_size = max(1, (max_idx + 1) // 10)
        ranges = {}
        for r in scored:
            bucket_start = (r["index"] // range_size) * range_size
            bucket_label = f"{bucket_start}-{bucket_start + range_size - 1}"
            if bucket_label not in ranges:
                ranges[bucket_label] = []
            ranges[bucket_label].append(r)

        lines.append("| Shot Range | Clips | Avg Quality | Avg Relevance |")
        lines.append("|------------|-------|-------------|---------------|")
        for label, clips in sorted(ranges.items(), key=lambda x: int(x[0].split("-")[0])):
            avg_q = sum(r["quality"] for r in clips) / len(clips)
            avg_r = sum(r["relevance"] for r in clips) / len(clips)
            lines.append(f"| {label} | {len(clips)} | {avg_q:.2f} | {avg_r:.2f} |")
        lines.append("")

    # Best clips
    lines.append("## Best Clips")
    lines.append("")
    lines.append("| Shot | Quality | Relevance | Notes |")
    lines.append("|------|---------|-----------|-------|")
    for r in best:
        lines.append(f"| {r['index']} | {r['quality']:.1f} | {r['relevance']:.1f} | {r['notes'][:80]} |")
    lines.append("")

    # Worst clips
    lines.append("## Worst Clips")
    lines.append("")
    lines.append("| Shot | Quality | Relevance | Notes |")
    lines.append("|------|---------|-----------|-------|")
    for r in worst:
        lines.append(f"| {r['index']} | {r['quality']:.1f} | {r['relevance']:.1f} | {r['notes'][:80]} |")
    lines.append("")

    # Common issues in worst clips
    lines.append("## Common Issues")
    lines.append("")
    low_quality = [r for r in scored if r["quality"] < 0.4]
    low_relevance = [r for r in scored if r["relevance"] < 0.4]
    lines.append(f"- **Low quality (<0.4)**: {len(low_quality)} clips ({len(low_quality)/len(scored)*100:.0f}%)")
    lines.append(f"- **Low relevance (<0.4)**: {len(low_relevance)} clips ({len(low_relevance)/len(scored)*100:.0f}%)")
    lines.append("")

    report_path.write_text("\n".join(lines))
    return str(report_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Evaluate reconstruction quality")
    subparsers = parser.add_subparsers(dest="command")

    # Video eval (default / original mode)
    video_parser = subparsers.add_parser("video", help="Evaluate reconstructed video quality")
    video_parser.add_argument("output_dir", help="Path to the pipeline output directory")
    video_parser.add_argument("--strategy", required=True, help="Video strategy name (e.g. fal-seedance)")
    video_parser.add_argument("--report-dir", default="docs/research", help="Directory for markdown report")

    # Audio clip eval
    audio_parser = subparsers.add_parser("audio", help="Evaluate individual audio clips")
    audio_parser.add_argument("output_dir", help="Path to the pipeline output directory")
    audio_parser.add_argument("--strategy", required=True, help="Audio strategy name (e.g. runpod-mmaudio)")
    audio_parser.add_argument("--sample", type=int, default=None, help="Evaluate every Nth clip (e.g. --sample 10)")
    audio_parser.add_argument("--report-dir", default="docs/research", help="Directory for markdown report")

    args = parser.parse_args()
    load_env()

    # Default to video if no subcommand (backwards compat)
    if args.command is None:
        # Re-parse as video for backwards compat with old CLI
        parser.print_help()
        sys.exit(1)

    if args.command == "audio":
        output_dir = Path(args.output_dir)
        prompts = load_json(output_dir / "prompts.json")
        run_audio_clip_eval(output_dir, args.strategy, prompts, args.sample, args.report_dir)
        return

    # Video eval
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
