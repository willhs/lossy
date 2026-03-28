#!/usr/bin/env python3
"""Evaluate audio captioning models against the existing Gemini pipeline.

Extracts audio clips for selected shots, runs a captioning model (CLAPCap by
default), and produces a side-by-side comparison against YAMNet labels and
Gemini sound descriptions from prompts.json.

Usage:
    python tools/eval_audio_captioning.py output/star_wars_iv_v2/
    python tools/eval_audio_captioning.py output/star_wars_iv_v2/ --shots 5,63,133
    python tools/eval_audio_captioning.py output/star_wars_iv_v2/ --model clapcap
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time

# 19 representative shots selected for bucket coverage, duration range, and
# content variety across the film.
DEFAULT_SHOT_INDICES = [
    5, 63, 133, 188, 208, 210, 373, 432, 484, 753,
    787, 813, 1039, 1098, 1138, 1178, 1411, 1485, 1585,
]

# Speech-leak keywords (mirrors strategies_audio.py _SPEECH_RE).
_SPEECH_RE = re.compile(
    r"\bvoice(?:s)?\b"
    r"|\bspeaking\b|\bspeaks\b|\bspoken\b|\bspeech\b"
    r"|\bdialogue\b"
    r"|\btalking\b|\btalks\b"
    r"|\bconversation\b"
    r"|\bsays\b|\bsaying\b"
    r"|\bshout(?:s|ing)?\b"
    r"|\bwhisper(?:s|ing)?\b"
    r"|\bscream(?:s|ing)?\b"
    r"|\byell(?:s|ing)?\b"
    r"|\bmurmur(?:s|ing)?\b"
    r"|\bnarrat(?:es?|ing|ion)\b"
    r"|\bvocal\b",
    re.IGNORECASE,
)


def load_data(output_dir):
    """Load prompts.json and audio_labels.json from the output directory."""
    prompts_path = os.path.join(output_dir, "prompts.json")
    labels_path = os.path.join(output_dir, "audio_labels.json")

    with open(prompts_path) as f:
        prompts = json.load(f)

    audio_labels = {}
    if os.path.exists(labels_path):
        with open(labels_path) as f:
            audio_labels = json.load(f)

    return prompts, audio_labels


def extract_clips(prompts, shot_indices, audio_path, clip_dir):
    """Extract audio clips for selected shots using ffmpeg."""
    clip_paths = {}
    for idx in shot_indices:
        shot = prompts[idx]
        start_s = shot["start_s"]
        duration_s = shot["duration_s"]
        clip_path = os.path.join(clip_dir, f"clip_{idx}.wav")
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", audio_path,
                "-ss", str(start_s), "-t", str(duration_s),
                "-ac", "1", "-ar", "16000",
                clip_path,
            ],
            capture_output=True, check=True,
        )
        clip_paths[idx] = clip_path
    return clip_paths


def run_clapcap(clip_paths, shot_indices):
    """Run CLAPCap on extracted clips. Returns {index: (caption, time_s)}."""
    from msclap import CLAP

    print("Loading CLAPCap model...")
    model = CLAP(version="clapcap", use_cuda=False)

    results = {}
    for idx in shot_indices:
        path = clip_paths[idx]
        t0 = time.perf_counter()
        captions = model.generate_caption(audio_files=[path])
        elapsed = time.perf_counter() - t0
        caption = captions[0] if captions else ""
        results[idx] = (caption, elapsed)
        print(f"  Shot {idx:>5d}: {elapsed:.2f}s — {caption}")
    return results


def run_whisper(clip_paths, shot_indices):
    """Run Whisper Audio Captioning on extracted clips."""
    try:
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor
        import librosa
    except ImportError:
        print("Whisper Audio Captioning requires transformers and librosa. Skipping.")
        return None

    print("Loading Whisper Audio Captioning model...")
    model_id = "MU-NLPC/whisper-small-audio-captioning"
    model = WhisperForConditionalGeneration.from_pretrained(model_id, trust_remote_code=True)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)
    model.eval()

    results = {}
    for idx in shot_indices:
        path = clip_paths[idx]
        audio, sr = librosa.load(path, sr=16000)
        features = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
        t0 = time.perf_counter()
        with torch.no_grad():
            generated = model.generate(
                features.input_features,
                max_length=100,
                forced_decoder_ids=feature_extractor.get_decoder_prompt_ids(
                    language="en", task="transcribe", no_timestamps=True,
                ),
            )
        elapsed = time.perf_counter() - t0
        # Decode and strip the "clotho > caption:" prefix if present
        from transformers import WhisperTokenizer
        tokenizer = WhisperTokenizer.from_pretrained(model_id)
        caption = tokenizer.decode(generated[0], skip_special_tokens=True)
        caption = re.sub(r"^.*?caption:\s*", "", caption)
        results[idx] = (caption, elapsed)
        print(f"  Shot {idx:>5d}: {elapsed:.2f}s — {caption}")
    return results


def build_results(prompts, audio_labels, shot_indices, model_results):
    """Build comparison results for each shot."""
    results = []
    for idx in shot_indices:
        shot = prompts[idx]
        labels_entry = audio_labels.get(str(idx), {})
        yamnet_labels = labels_entry.get("labels", [])
        bucket = labels_entry.get("bucket", shot.get("audio_detected", {}).get("bucket", ""))
        gemini_sound = shot.get("description", {}).get("sound", "")
        has_dialogue = shot.get("dialogue") is not None

        caption, inference_time = model_results.get(idx, ("", 0.0))
        speech_leak = bool(_SPEECH_RE.search(caption))

        results.append({
            "index": idx,
            "bucket": bucket,
            "duration_s": shot["duration_s"],
            "has_dialogue": has_dialogue,
            "yamnet_labels": yamnet_labels,
            "gemini_sound": gemini_sound,
            "model_caption": caption,
            "inference_time_s": round(inference_time, 3),
            "speech_leak": speech_leak,
            "caption_length": len(caption),
        })
    return results


def write_report(results, output_path, model_name):
    """Write a markdown comparison report."""
    total = len(results)
    avg_time = sum(r["inference_time_s"] for r in results) / total if total else 0
    speech_leaks = sum(1 for r in results if r["speech_leak"])
    avg_len = sum(r["caption_length"] for r in results) / total if total else 0
    descriptive = sum(1 for r in results if r["caption_length"] > 20)

    lines = [
        f"# Audio Captioning Eval: {model_name}",
        "",
        "## Summary",
        "",
        f"- **Model**: {model_name}",
        f"- **Shots evaluated**: {total}",
        f"- **Avg inference time**: {avg_time:.2f}s per clip",
        f"- **Speech leak rate**: {speech_leaks}/{total} ({100*speech_leaks/total:.0f}%)",
        f"- **Avg caption length**: {avg_len:.0f} chars",
        f"- **Descriptive captions** (>20 chars): {descriptive}/{total} ({100*descriptive/total:.0f}%)",
        "",
        "## Per-Shot Comparison",
        "",
        "| Idx | Bucket | Dur | DLG | YAMNet Labels | Gemini Sound | Model Caption | Time | Leak |",
        "|-----|--------|-----|-----|---------------|--------------|---------------|------|------|",
    ]

    for r in results:
        yamnet = ", ".join(r["yamnet_labels"][:3]) if r["yamnet_labels"] else "-"
        dlg = "Y" if r["has_dialogue"] else ""
        leak = "**YES**" if r["speech_leak"] else ""
        lines.append(
            f"| {r['index']} | {r['bucket']} | {r['duration_s']:.1f}s | {dlg} "
            f"| {yamnet} | {r['gemini_sound'][:60]} | {r['model_caption'][:60]} "
            f"| {r['inference_time_s']:.2f}s | {leak} |"
        )

    lines.extend([
        "",
        "## Detailed Captions",
        "",
    ])

    for r in results:
        yamnet = ", ".join(r["yamnet_labels"]) if r["yamnet_labels"] else "-"
        lines.extend([
            f"### Shot {r['index']} ({r['bucket']}, {r['duration_s']:.1f}s{', dialogue' if r['has_dialogue'] else ''})",
            "",
            f"- **YAMNet**: {yamnet}",
            f"- **Gemini**: {r['gemini_sound']}",
            f"- **{model_name}**: {r['model_caption']}",
            f"- Inference: {r['inference_time_s']:.2f}s | Speech leak: {'YES' if r['speech_leak'] else 'no'}",
            "",
        ])

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate audio captioning models against Gemini pipeline"
    )
    parser.add_argument("output_dir", help="Path to encode output directory")
    parser.add_argument(
        "--shots", type=str, default=None,
        help="Comma-separated shot indices to evaluate (default: 19 representative shots)",
    )
    parser.add_argument(
        "--model", choices=["clapcap", "whisper"], default="clapcap",
        help="Captioning model to use (default: clapcap)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory for results (default: eval_audio_captioning/ in output dir)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.rstrip("/")
    audio_path = os.path.join(output_dir, "audio.wav")
    if not os.path.exists(audio_path):
        print(f"Error: {audio_path} not found", file=sys.stderr)
        sys.exit(1)

    shot_indices = DEFAULT_SHOT_INDICES
    if args.shots:
        shot_indices = [int(x.strip()) for x in args.shots.split(",")]

    results_dir = args.output or os.path.join(output_dir, "eval_audio_captioning")
    os.makedirs(results_dir, exist_ok=True)

    print(f"Loading data from {output_dir}...")
    prompts, audio_labels = load_data(output_dir)

    # Validate shot indices
    valid_indices = []
    for idx in shot_indices:
        if idx < len(prompts):
            valid_indices.append(idx)
        else:
            print(f"Warning: shot index {idx} out of range (max {len(prompts)-1}), skipping")
    shot_indices = valid_indices

    with tempfile.TemporaryDirectory() as clip_dir:
        print(f"Extracting {len(shot_indices)} audio clips...")
        clip_paths = extract_clips(prompts, shot_indices, audio_path, clip_dir)

        print(f"Running {args.model} on clips...")
        if args.model == "clapcap":
            model_results = run_clapcap(clip_paths, shot_indices)
        elif args.model == "whisper":
            model_results = run_whisper(clip_paths, shot_indices)
            if model_results is None:
                sys.exit(1)

        print("Building comparison...")
        results = build_results(prompts, audio_labels, shot_indices, model_results)

    # Write outputs
    results_path = os.path.join(results_dir, "results.json")
    report_path = os.path.join(results_dir, "report.md")

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {results_path}")

    write_report(results, report_path, args.model)
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
