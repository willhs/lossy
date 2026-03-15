---
id: adr-004-replicate-wan22
type: decision
purpose: "Record the decision to use Replicate's Wan 2.2 Fast for text-to-video generation in the decode stage."
scope: ["design", "decoder"]
non_goals: []
tags: ["adr", "decoder", "video-generation"]
related: ["design/architecture.md", "research/0003-prompt-to-video/research.md"]
---

# Context

The decoder needs to generate a video clip for each shot description in the prompt manifest (~1,161 shots for Star Wars EP IV). Cost is the primary constraint -- the project's aesthetic embraces low quality, so we optimized for cheapest-per-clip rather than best output.

We evaluated commercial APIs (Replicate, fal.ai, Runway, Google Veo, Kling), open-source self-hosted options (Wan 2.2, CogVideoX, LTX-Video on RunPod), and various models at different price points. Full analysis in `docs/research/0003-prompt-to-video/research.md`.

# Decision

Use **Replicate** with **wan-video/wan-2.2-t2v-fast** for video generation.

- 480p (832x480), 16:9
- 81 frames at 16fps = ~5.06s per clip
- ~$0.05 per clip, ~$58 for the full film
- Simple Python SDK, no infrastructure to manage

# Consequences

**Good:**
- Cheapest managed API option by a significant margin ($58 vs $230+ for alternatives)
- Dead-simple integration (5 lines of code)
- Reproducible via seed parameter

**Bad:**
- Fixed clip duration (~5s). Original shots range from <1s to 28s, requiring FFmpeg speed-adjustment in post. Very short original shots produce near-still frames after speed-up. Very long shots look slow-motion.
- 480p only on the fast variant. Sufficient for the project but limits comparison quality.
- No arbitrary duration control. All commercial APIs use discrete enums, not continuous values.

**Tested:**
- 15-clip test batch on Star Wars EP IV shots 10-24 (opening space battle + droids)
- 0 failures, $0.75 total cost, matched estimate exactly
- Character/object shots (C-3PO, R2-D2, Star Destroyers) produce recognizable results
- Abstract space/planet shots less convincing but still identifiable
- SSL issue with macOS Python urllib -- switched to httpx for downloads

# Alternatives Considered

- **fal.ai + Wan 2.5**: Better quality but 5x cost ($0.25/clip, ~$290 total)
- **Self-hosted Wan 2.2 on RunPod**: Cheapest possible (~$5-15) but requires GPU instance management
- **Seedance 1.0 Pro on fal.ai**: Integer-second duration control (2-12s) which would reduce speed-adjustment artifacts, but 4x cost
- **Google Veo 3.1 Fast**: Best quality but 10x cost ($0.50/clip, ~$580 total)

# Status Update (2026-03-15)

The decoder now supports multiple video generation backends via a strategy pattern (`--strategy` CLI flag). Replicate Wan 2.2 Fast remains the default and cheapest option. Seedance 1.0 Pro Fast on fal.ai was added as an alternative that supports 2-12s integer duration control, addressing the speed-adjustment artifacts documented in the "Bad" consequences above.

See experiment 0002 (`docs/research/experiments/0002-seedance-duration-test.md`) for the comparison results.
