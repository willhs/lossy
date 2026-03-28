---
id: task-0013
type: spec
purpose: "Evaluate lightweight local audio captioning models as a replacement for the YAMNet + Gemini pipeline that produces the sound field in encode stage2."
tags: ["audio", "encoder", "eval", "research"]
related:
  - "docs/design/adr/005-yamnet-audio-classification.md"
  - "docs/research/0007-speech-filter/research.md"
  - "docs/tasks/0005-audio-generation/task.md"
  - "docs/tasks/0002-audio-encoding/task.md"
created: 2026-03-28
updated: 2026-03-28
---

# Evaluate Lightweight Audio Captioning Models

## Goal

Determine whether a local audio captioning model (CoNeTTE, CLAPCap, or similar) can replace the current YAMNet labels + Gemini imagination pipeline for producing the `sound` field in encode stage2. Produce a research doc with findings and a recommendation.

## Context

The current pipeline for generating sound descriptions works in two steps:

1. **YAMNet** classifies each shot's audio into coarse labels (521 AudioSet classes grouped into 6 buckets) and caches them to `audio_labels.json`.
2. **Gemini** receives the labels alongside visual keyframes and "imagines" a natural-language sound description for the `sound` field in `prompts.json`.

This has three known problems:

- **Hallucination**: Gemini invents sounds based on visual context rather than actual audio content. A shot showing an explosion may get "massive detonation" even if the audio is quiet dialogue.
- **Speech leakage**: Gemini descriptions often include character voices and dialogue references, which cause MMAudio (an SFX/ambiance model) to produce garbled output. The speech filter (0007) is a workaround, not a fix.
- **Heavy dependency**: TensorFlow Hub is pulled in solely for YAMNet classification (~500MB install).

Local audio captioning models generate natural-language descriptions directly from audio waveforms, bypassing both problems: descriptions are grounded in what's actually heard, and models trained on AudioCaps/Clotho data don't hallucinate speech descriptions for non-speech audio.

**Candidates:**
- **CoNeTTE** (41M params, MIT, CPU-friendly, designed for 1-30s clips)
- **CLAPCap** (Microsoft CLAP backbone, ~200M params, MIT, CPU-friendly)
- **Whisper Audio Captioning** (244M small variant, CC BY-NC license -- note restriction)

**Test data**: `output/star_wars_iv_v2/` contains `prompts.json` (1460 shots), `audio.wav` (239MB), `scenes.json`, and `audio_labels.json`.

## Scope

### Must Do

- Select ~20 representative shots spanning music, SFX, ambient, dialogue, and silence buckets
- Build `tools/eval_audio_captioning.py` to extract audio clips and run each model
- Install and run at least CoNeTTE and CLAPCap on the selected clips
- Measure per-clip inference time on M-series Mac (CPU)
- Produce side-by-side comparison: YAMNet labels vs Gemini sound description vs each model's caption
- Assess each model on: audio grounding, speech-freedom, descriptive richness for MMAudio, throughput
- Write research doc at `docs/research/0008-audio-captioning/`

### Might Do

- Test Whisper Audio Captioning if license is acceptable
- Run MMAudio on a few clips with model captions vs Gemini descriptions to compare generated audio quality
- Benchmark memory usage alongside throughput

### Won't Do

- Modify `encode.py` to use a new model (separate task if eval succeeds)
- Remove YAMNet or TensorFlow from the current pipeline
- Build a production-ready integration
- Evaluate paid/API-based captioning services

## Constraints

- Must run on CPU (M-series Mac) -- no GPU requirement for eval
- No new paid API calls -- local models only
- Eval script goes in `tools/`, not in the main pipeline
- Research-first: findings inform a follow-up integration task

## References

- [ADR-005: YAMNet Audio Classification](../../../docs/design/adr/005-yamnet-audio-classification.md) -- current pipeline design and trade-offs
- [Research 0007: Speech Filter](../../../docs/research/0007-speech-filter/research.md) -- speech leakage problem and workaround results
- [Task 0005: Audio Generation](../../../docs/tasks/0005-audio-generation/task.md) -- MMAudio integration that consumes the `sound` field
- [Task 0002: Audio Encoding](../../../docs/tasks/0002-audio-encoding/task.md) -- original YAMNet integration

## Success Criteria

- [ ] Eval script runs all candidate models on selected clips and produces a comparison report
- [ ] Research doc answers: can a local model produce descriptions that are (1) grounded in actual audio, (2) speech-free without filtering, (3) descriptive enough for MMAudio, and (4) fast enough for full-film encoding?
- [ ] Clear recommendation: adopt a specific model, investigate further, or keep current pipeline
