---
id: vision
type: note
purpose: "Capture the north star for lossy."
scope: ["philosophy"]
non_goals: []
tags: ["vision"]
related: ["philosophy/principles.md"]
---

## Purpose

A comically lossy film codec that compresses movies/TV into text prompts and reconstructs them via AI video generation.

## Concept

Treat a film as data to be compressed. The "encoder" breaks a film into scenes and produces a text description of each — visual content, camera work, dialogue, mood, etc. The "decoder" takes those text prompts and feeds them into an AI video generation model to reconstruct the film scene by scene. The result is a heavily degraded, AI-hallucinated version of the original — a round-trip through language that reveals what information survives natural language description and what gets lost or mangled.

## Pipeline

1. **Encode** — Process source film into a sequence of timestamped scene descriptions (the "compressed" format). Method TBD — could be a vision model watching the film, manual description, or a hybrid.
2. **Decode** — Feed each scene description as a prompt into a video generation model. Generate clips at ~480p to minimize cost, using cloud GPU services with time-based pricing (e.g. RunPod, Replicate). Stitch clips together into a final reconstructed film.
3. **Compare** — Present original vs reconstructed side-by-side to examine what was preserved and what was lost.

## Constraints & Goals

- Minimize cost throughout — cheap inference, low resolution, shortest viable clip lengths.
- The project is intentionally absurd. The "codec" framing is the joke — this is obviously a terrible way to compress video. The interest is in what the process reveals about the limits of language as a representation of visual media, and the current state of AI video generation.
- Output quality being bad is a feature, not a bug.

## Output

A Weird Software blog post walking through the process, showing results, and reflecting on what was learned. The repo contains the encoding/decoding pipeline and tooling.
