---
id: task-0002
type: spec
purpose: "Classify a film's soundtrack into structured text and enrich the prompt manifest with audio descriptions."
tags: ["encoder", "audio", "yamnet"]
related: ["docs/research/0002-shot-to-prompt/research.md", "docs/work/roadmap.md"]
created: 2026-03-15
updated: 2026-03-15
---

# Audio Encoding v1 — Classify Soundtrack into Text Manifest

## Goal

Extract audio information from the source film and feed it into the encoding pipeline so that each shot's description in `prompts.json` is informed by what's happening sonically — not just visually. This closes the gap where audio (38% of the source file) is completely ignored by the encoder.

## Context

### What the pipeline does today

The lossy pipeline compresses films into text manifests and reconstructs them via AI video generation. The encoder (`encode.py`) runs in two stages:

1. **Stage 1** — PySceneDetect detects shot boundaries, FFmpeg extracts 2-8 keyframes per shot → `manifest.json`
2. **Stage 2** — Extracts embedded subtitles (SRT via FFmpeg), detects camera motion (Farneback optical flow), calls Gemini Flash-Lite with keyframes + metadata to produce structured descriptions → `prompts.json`

The decoder (`decode.py`) reads `prompts.json`, generates video clips via Replicate Wan or fal.ai Seedance, speed-adjusts and stitches them into a reconstructed film. The decoder's `format_prompt()` converts structured descriptions into flat text prompts for video generation models.

### What's missing

Audio is completely ignored. The encoder extracts no audio information — no transcription, no sound classification, no music detection. The only dialogue data comes from embedded SRT subtitles (which not all files have, and which lack speaker identity). The decoder generates silent video.

This means the text manifest — the project's core "compressed" representation — is missing ~38% of the source information. Scenes with no dialogue but critical audio (the Star Wars opening fanfare, lightsaber hums, explosion sequences, tense silence) are indistinguishable from each other in the current manifest.

### Why it matters for reconstruction

Even though v1 of the decoder produces silent video, audio metadata improves visual reconstruction. Video generation models respond to mood and atmosphere cues. "Tense silence, distant machinery hum" produces different video than "triumphant orchestral music, crowd cheering." Sound categories in the prompt will make generated clips more faithful even without a soundtrack.

### What this task covers

Audio classification via YAMNet feeds into the existing Gemini vision API call as context — the same pattern camera motion already uses. Optical flow produces a label ("pan right") that Gemini incorporates into its description. YAMNet produces audio labels ("Orchestra, Explosion, Rumble") that Gemini incorporates into a natural-language `sound` field. The result is richer descriptions in `prompts.json` that capture both what the shot looks like and what it sounds like.

This task does not cover:
- Synthesizing or reconstructing audio during decode (future task)
- Generating a soundtrack from the manifest (future task)
- Speaker diarization via WhisperX (future task — v1 reuses existing SRT subtitle extraction for dialogue)

### Tools

v1 uses **YAMNet** only — a TensorFlow Hub model that classifies audio into 521 AudioSet categories (speech, music, silence, gunshot, explosion, laughter, rain, etc.) per ~0.48s frame. Runs on CPU. Lightweight.

**WhisperX** (Whisper + wav2vec2 alignment + pyannote.audio diarization) is a natural follow-on for replacing SRT-based dialogue with transcription + speaker IDs, but is out of scope for v1.

### Where it fits in the pipeline

Audio classification slots into stage 2 as a new preprocessing step alongside subtitles and camera motion. The flow becomes:

1. Subtitle extraction (existing)
2. Camera motion detection (existing)
3. **Audio extraction + YAMNet classification (new)**
4. Gemini vision API (existing, now with audio context)

YAMNet runs once on the full audio track, produces per-shot labels, and caches results to `audio_labels.json` (same pattern as `camera_motion.json`). Those labels are then passed as context to the Gemini call for each shot.

### How audio enriches prompts.json

The Gemini system prompt gains a new output field (`sound`) and each shot's user prompt gains audio context. The result in `prompts.json`:

```json
{
  "index": 0,
  "start_s": 0.0,
  "end_s": 28.779,
  "duration_s": 28.779,
  "camera_motion_detected": "static",
  "audio_detected": {"bucket": "music", "labels": ["Soundtrack music", "Orchestra", "Brass instrument"]},
  "dialogue": null,
  "description": {
    "shot_type": "...",
    "camera_movement": "...",
    "subjects": "...",
    "action": "...",
    "lighting": "...",
    "color_palette": "...",
    "mood": "...",
    "setting": "...",
    "sound": "Triumphant orchestral fanfare with prominent brass section and timpani rolls, building in intensity"
  }
}
```

The `audio_detected` field stores the raw YAMNet classification (like `camera_motion_detected`). The `sound` field in `description` is Gemini's natural-language audio description, informed by both the YAMNet labels and the visual keyframes.

## Requirements

- Extract the audio track from the source video (FFmpeg → WAV, mono, 16kHz)
- Run YAMNet on the full audio to classify sound events per ~0.48s frame
- Build a category grouping map: 521 YAMNet classes → 6 buckets (music, speech, effects, ambient, silence, other), preserving top-N raw labels per shot
- Align YAMNet classification windows to shot boundaries from `manifest.json`, picking the dominant bucket and top labels per shot
- Cache per-shot audio labels to `audio_labels.json` (same pattern as `camera_motion.json`)
- Feed audio labels into the Gemini vision API prompt as context for each shot
- Add a `sound` field to the Gemini system prompt's structured output schema
- Store raw audio classification alongside camera motion in `prompts.json` entries
- Resume support: cache extracted WAV, raw YAMNet scores, and per-shot audio labels
- Works on the Star Wars EP IV run (~1,150 shots, ~2h audio)

## Success Criteria

- [ ] Stage 2 produces `audio_labels.json` with per-shot YAMNet classifications
- [ ] YAMNet classifications are grouped into meaningful buckets (not just "speech" for every shot with dialogue)
- [ ] Audio labels are passed as context to Gemini, which produces a natural-language `sound` field per shot
- [ ] `prompts.json` entries include both `audio_detected` and a `sound` description
- [ ] Existing tests pass; new tests cover category grouping, per-shot aggregation, and manifest format
- [ ] Full Star Wars EP IV audio encodes in a reasonable time (target: under 30 minutes on a Mac)
- [ ] Architecture docs updated to cover the audio encoding stage
- [ ] ADR written for audio classification approach (YAMNet, bucket grouping, per-shot granularity)
