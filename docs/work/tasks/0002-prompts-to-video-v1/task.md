---
id: task-0002
type: spec
purpose: "Build the decoder: read a prompt manifest, generate video clips via Replicate Wan 2.2, and stitch them into a reconstructed film."
tags: ["decoder", "pipeline", "video-generation", "replicate"]
related: ["research/prompt-to-video-landscape.md", "design/architecture.md", "work/tasks/0001-screenplay-to-prompts-v1/task.md"]
created: 2026-03-14
updated: 2026-03-14
---

# Prompts-to-Video v1

## Goal

Build a decoder that reads the prompt manifest produced by the encoder (prompts.json), generates a video clip for each shot via Replicate's Wan 2.2 text-to-video API, and stitches the clips into a single reconstructed film using FFmpeg.

## Context

The encoder pipeline is complete and tested. Star Wars EP IV has been fully encoded: 1,161 shots with structured descriptions (shot type, camera movement, subjects, action, lighting, color palette, mood, setting) plus metadata (duration, dialogue, camera motion). The output lives in `output/star_wars_iv_v2/prompts.json`.

The decoder is the second stage of the lossy pipeline (architecture.md). It reads the "compressed" text representation and reconstructs video from it. Quality will be low -- that's the point. The interesting output is the comparison between original and reconstruction.

Research in `docs/research/prompt-to-video-landscape.md` identified Replicate + Wan 2.2 Fast as the cheapest managed API option at ~$0.05/clip (~$58 for all 1,161 shots). The model generates fixed-length clips (~5s at 480p), so duration mismatch with original shots must be handled in post-processing.

## Requirements

- Read prompts.json and convert structured description objects into flat text prompts suitable for video generation
- Call Replicate Wan 2.2 Fast API (wan-video/wan-2.2-t2v-fast) for each shot
- Download generated clips to a structured output directory
- Resume support: track which shots have been generated, skip completed ones on restart, save progress incrementally
- Start-offset flag: ability to skip N shots from the beginning (e.g., skip credits/crawl for Star Wars)
- FFmpeg stitcher: concatenate all clips into a single video, with optional speed-adjustment to match original shot durations
- CLI interface: `python decode.py --prompts output/star_wars_iv_v2/prompts.json --start-index 7`
- Output structure: `output/{film}/clips/` for individual clips, `output/{film}/reconstructed.mp4` for final output
- Rate limit handling and error retry (Replicate API)
- Cost tracking: log actual API costs or at minimum count of clips generated

## Success Criteria

- [ ] Prompt formatter produces coherent video-gen prompts from the structured description format
- [ ] Test batch of 10-20 clips generates successfully via Replicate
- [ ] Actual API cost for test batch matches estimates (~$0.50-1.00)
- [ ] Generated clips are recognizable as attempts at the described scenes (low bar -- they don't need to be good)
- [ ] Resume works: can stop and restart without re-generating completed clips
- [ ] FFmpeg stitcher concatenates clips into a single playable video
- [ ] Full pipeline can reconstruct Star Wars EP IV (minus credits) for under $60
