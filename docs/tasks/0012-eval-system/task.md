---
id: task-0012
type: spec
purpose: "Add an automated evaluation system that measures reconstruction quality by re-encoding the output and comparing to the original encoding."
tags: ["eval", "quality", "gemini", "yamnet"]
related: ["design/adr/002-stateless-cli-pipeline.md", "design/adr/005-yamnet-audio-classification.md"]
created: 2026-03-26
updated: 2026-03-26
---

# Eval System

## Goal

Create `eval.py` -- a standalone CLI that measures how well a decoded video reproduces the original film, producing per-shot and aggregate quality scores across video, audio, and speech dimensions. Reports are written as markdown to `docs/research/` for tracking over time.

## Context

There is currently no automated way to measure output quality. The only comparison tool is a browser-based side-by-side viewer (`tools/compare.html`) for manual human review. As the pipeline evolves (prompt formatting, new strategies, stitching changes), there's no way to detect regressions or quantify improvements without watching every clip.

The core insight: re-encode the reconstructed video using the same stage2 process (Gemini vision, YAMNet, subtitle extraction) and compare the resulting descriptions field-by-field against the original `prompts.json`. This is "lossy-native" -- it evaluates through the same lens the pipeline uses to perceive video.

**Key design decision:** Don't re-run shot detection on the reconstructed video. Use the original manifest's timecodes to extract keyframes from the reconstructed video at the same positions. This keeps shots aligned by construction and avoids the shot-boundary-mismatch problem.

## Scope

### Must Do

- CLI (`eval.py`) that takes an output directory and strategy name
- Extract keyframes from reconstructed video at original manifest timecodes
- Run Gemini vision on extracted keyframes to produce re-encoded descriptions
- Run YAMNet on reconstructed audio at original shot boundaries
- Re-extract subtitles from reconstructed video
- Per-shot comparison across all dimensions:
  - Categorical fields (shot_type, camera_movement): exact match (0 or 1)
  - Text fields (subjects, action, lighting, color_palette, mood, setting, sound): Gemini similarity scoring (0-1)
  - Audio bucket: exact match; audio labels: Jaccard similarity
  - Dialogue: word error rate vs original subtitles
- Aggregate scores: per-dimension averages and overall score
- Markdown report written to `docs/research/` with per-shot breakdown, worst-performing shots highlighted, and eval cost

### Might Do

- Integration with `pipeline.py` (optional `--eval` flag to auto-run after stitch)
- Score history tracking (append to a running log for trend analysis)

### Won't Do

- Visual/perceptual metrics (SSIM, LPIPS, FID) -- these require heavy dependencies and don't align with the text-description approach
- Modifications to `tools/compare.html` -- the browser tool stays manual
- Real-time or continuous evaluation -- this is a batch process run after decode+stitch

## Constraints

- Follows the stateless CLI stage pattern (ADR-002) -- reads/writes files on disk, no shared state
- Reuses existing stage2 logic (Gemini vision, YAMNet, subtitle extraction) rather than duplicating
- Gemini API cost should stay low: re-encode is ~stage2 cost, similarity prompts add ~$0.001/field/shot
- No new dependencies beyond what's already in the project

## References

- `design/adr/002-stateless-cli-pipeline.md` -- stateless CLI pattern this must follow
- `design/adr/005-yamnet-audio-classification.md` -- YAMNet classification approach to reuse for audio eval

## Success Criteria

- [ ] `python eval.py output/film --strategy fal-seedance` produces a markdown report
- [ ] Report contains per-shot scores for video, audio, and speech dimensions
- [ ] Report contains aggregate scores (per-dimension averages + overall)
- [ ] Worst-performing shots are identified in the report
- [ ] Eval cost is tracked and reported
- [ ] Tests cover comparison logic (field matchers, score aggregation)
