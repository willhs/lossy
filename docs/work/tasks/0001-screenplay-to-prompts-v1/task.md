---
id: task-0001
type: spec
purpose: "Build a two-stage encoding pipeline: shot detection via PySceneDetect, then shot-to-prompt generation using a vision API with metadata enrichment."
tags: ["encoder", "pipeline", "vision-api", "pyscenedetect"]
related: ["research/0002-shot-to-prompt/research.md", "research/0001-shot-detection/research.md", "design/adr/003-pyscenedetect-for-shot-detection.md"]
created: 2026-03-14
updated: 2026-03-14
---

# Screenplay-to-Prompts v1

## Goal

Build a two-stage encoding pipeline that takes a film video file as input and produces a prompt manifest (one descriptive text prompt per shot) as output.

**Stage 1 — Shot Detection**: Run PySceneDetect against the input file to detect shot boundaries, extract keyframes, and save intermediate outputs (shot list, keyframe images) to disk.

**Stage 2 — Shots to Prompts**: For each detected shot, generate a descriptive text prompt capturing visual content, motion, camera movement, lighting, and mood. Use a vision API (Gemini 2.5 Flash-Lite recommended) with metadata enrichment from subtitles and optical flow.

## Context

We already have a working PySceneDetect integration (`encode.py`) that detects ~2,070 shots from Star Wars Episode IV. The next step is turning those shots into text prompts that can drive video generation in the decode stage.

Research in `docs/research/0002-shot-to-prompt/research.md` identified the cheapest viable approach: extract 4-8 frames per shot at 512px, enrich with subtitle dialogue and optical flow camera motion labels, then send to Gemini 2.5 Flash-Lite (~$0.60 for 3,000 shots, or free on the free tier over 3 days).

The current `encode.py` script is a standalone tool. This task re-engineers it into a pipeline architecture where each stage reads the previous stage's output, making it possible to re-run individual stages and inspect intermediate results.

## Requirements

- Pipeline architecture: each stage reads input from disk and writes output to disk (no in-memory coupling between stages)
- Stage 1 outputs: shot boundary list (JSON or CSV with start/end timestamps), keyframe images (3-6 per shot at 512px, uniform sampling)
- Stage 2 inputs: stage 1 outputs + original video file (for subtitle extraction)
- Stage 2 outputs: prompt manifest (JSON, one entry per shot with structured fields)
- Prompt fields per shot: shot type, camera movement, subjects/action, lighting, color palette, mood, dialogue (if any)
- Subtitle extraction and alignment to shots (ffmpeg SRT extraction, timestamp overlap)
- Camera motion detection via optical flow (OpenCV Farneback or antiboredom/camera-motion-detector)
- Vision API integration (Gemini 2.5 Flash-Lite as default, with ability to swap providers)
- CLI interface: `python encode.py stage1 input.mp4` and `python encode.py stage2 output_dir/`
- Intermediate outputs stored in a structured directory (e.g., `output/{film}/shots/`, `output/{film}/frames/`, `output/{film}/prompts/`)

## Success Criteria

- [ ] Stage 1 runs PySceneDetect on Star Wars EP IV and produces a shot list + keyframe images in a structured output directory
- [ ] Stage 2 reads stage 1 output, extracts subtitles, runs optical flow, and generates prompts via Gemini Flash-Lite
- [ ] Prompt manifest contains structured descriptions for all ~2,070 shots
- [ ] Each stage can be run independently (re-running stage 2 doesn't require re-running stage 1)
- [ ] Total API cost for stage 2 is under $5 for the full film
- [ ] Pipeline handles the edge cases identified in research: shots over 30s (credits/crawl), very short shots (<1s)
