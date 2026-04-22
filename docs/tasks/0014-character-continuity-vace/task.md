---
id: task-0014
type: spec
purpose: "Test character continuity by adding an encode stage 3 that resolves characters into a registry, generating canonical character portraits, and using them as VACE reference images during video generation."
tags: ["character-continuity", "encode", "vace", "decode", "experiment"]
related: ["research/0011-character-continuity/research.md"]
created: 2026-03-31
updated: 2026-03-31
---

# Character Continuity: Encode Stage 3 + VACE Reference Conditioning

## Goal

Test whether a fully generative character continuity pipeline — no frames from source media — produces visually consistent characters across shots. Measure quality and consistency against the current T2V-1.3B baseline.

## Context

Each clip is currently generated in isolation with no cross-shot identity anchoring. Characters look like a different actor in almost every clip (see research/0011). The approach here avoids using source media frames in the output by:

1. Adding an encode stage 3 that makes a single Gemini text pass over all `subjects` fields in `prompts.json`, clustering them into named characters with canonical appearance descriptions, and writing a `characters.json` sidecar.
2. At decode time, generating a canonical portrait image per character using an image model (driven by the character description).
3. Using those generated portraits as VACE reference images when generating each clip, so character appearance is anchored without coupling the output to the source material.

VACE is used over I2V because VACE's reference images are appearance anchors, not literal starting frames — the clip composition and motion come entirely from the text prompt. VACE-1.3B fits on the current 24 GB RunPod pod with no infrastructure changes.

The key unknown is whether VACE-1.3B produces acceptable quality compared to the current T2V-1.3B output.

## Scope

### Must Do
- Implement encode stage 3: Gemini text pass over `prompts.json` that produces `characters.json` with character names, canonical descriptions, and the shot indices they appear in
- Implement character portrait generation: one image per character from their canonical description, stored as `characters/{name}.png` in the output directory
- Implement VACE strategy (RunPod): new `RunPodVaceStrategy` that loads VACE-1.3B and passes the relevant character portrait(s) as reference images alongside the text prompt
- Run a test decode on an existing encoded film and visually compare character consistency and generation quality against T2V-1.3B output

### Might Do
- Expose stage 3 via `encode.py stage3` CLI so it can be run standalone or skipped
- Fall back to T2V when no character reference is available for a shot

### Won't Do
- Use frames from source media as reference images
- Support VACE-14B (requires pod upgrade, out of scope for this experiment)
- Automate the quality comparison — visual inspection is sufficient for the test

## Constraints

- Must fit within the existing 24 GB RunPod pod (VACE-1.3B only)
- No new dependencies beyond what's already available in the pipeline
- Character portrait generation should use an existing API (Gemini image gen, fal.ai, or similar already in use)

## References

- [Character continuity research](../../research/0011-character-continuity/research.md) — root cause analysis and option assessment

## Success Criteria

- [ ] `encode.py stage3` produces a valid `characters.json` for a test film with at least 2 named characters
- [ ] Portrait images are generated for each character and stored in the output directory
- [ ] VACE strategy generates clips using character reference images without OOM on the 24 GB pod
- [ ] Visual inspection shows reduced character drift compared to T2V-1.3B baseline
- [ ] Generation quality (ignoring character consistency) is acceptable relative to T2V-1.3B
