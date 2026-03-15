---
id: task-0003
type: spec
purpose: "Refactor the decoder to support multiple video generation backends via a strategy pattern, then trial Seedance 1.0 Pro Fast as an alternative to Replicate Wan 2.2."
tags: ["decoder", "video-generation", "strategy-pattern", "seedance", "fal-ai"]
related: ["research/0003-prompt-to-video/research.md", "design/adr/004-replicate-wan22-for-video-generation.md", "research/experiments/0001-first-e2e-decode-test.md"]
created: 2026-03-15
updated: 2026-03-15
---

# Seedance Strategy Trial

## Goal

Refactor `decode.py` so video generation backends are swappable (strategy pattern), then run a 15-shot test batch with Seedance 1.0 Pro Fast on fal.ai to evaluate whether integer-second duration control (2-12s) eliminates the speed-adjustment artifacts that were the biggest problem in experiment 0001.

## Context

Experiment 0001 revealed that 53% of test shots (8 of 15) needed 2-5x speed adjustment because Wan 2.2 generates fixed ~5.06s clips. Sub-1.5s shots became freeze frames, and the 19s establishing shot played at nearly 4x slowdown. This is the pipeline's biggest quality problem.

Seedance 1.0 Pro on fal.ai supports integer-second duration control (2-12s), which would eliminate speed adjustment for most shots. Duration analysis of shots 10-24 shows 11 of 15 shots would have <0.5s delta from the original duration with simple rounding.

The decoder also needs to support multiple backends going forward (Replicate/Wan, fal.ai/Seedance, self-hosted RunPod), so a strategy pattern refactor is the right foundation.

## Requirements

- Strategy pattern in decode.py with a common interface for video generation backends
- `generate_clip` returns a list of `ClipResult` (not a single result) because long shots may be split into multiple clips
- `ClipResult` includes `path`, `actual_duration_s`, and `cost` so the stitcher and cost tracking work generically
- Existing Replicate/Wan behaviour preserved as `ReplicateWanStrategy` with no functional changes
- New `FalSeedanceStrategy` that rounds target duration to nearest integer (clamped 2-12s) and splits shots >12s into multiple clips
- Stitcher uses `actual_duration_s` from results instead of hardcoded `81/16`
- CLI `--strategy` flag to select backend (default: `replicate-wan`)
- Trial uses Seedance Fast variant at 480p for cost efficiency (~$0.10/clip)
- Test batch covers shots 10-24 (same as experiment 0001, skipping the opening text crawl)
- Results documented in `docs/research/experiments/0002-seedance-duration-test.md`

## Success Criteria

- [ ] `decode.py --strategy replicate-wan` produces identical behaviour to current code
- [ ] `decode.py --strategy fal-seedance` generates clips with durations matching original shots (within 1s for shots in the 2-12s range)
- [ ] Shot 10 (19s) is split into two clips (12s + 7s) and stitched seamlessly
- [ ] Stitched output from Seedance has noticeably fewer speed-adjustment artifacts than the Wan 2.2 output
- [ ] Experiment 0002 documents quality, duration accuracy, and cost comparison
