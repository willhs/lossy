---
id: adr-007-temporal-prompt-variation-for-split-shots
type: decision
purpose: "Record the decision to generate per-segment descriptions for long shots so that split clips get distinct prompts during decode."
scope: ["design", "encoder", "decoder"]
non_goals: []
tags: ["adr", "encoder", "decoder", "video-generation", "prompt-engineering"]
related: ["design/adr/005-runpod-self-hosted-wan-strategy.md", "research/0003-prompt-to-video/research.md"]
---

# Context

When a shot exceeds a generation strategy's max clip duration (e.g., ~4s for `runpod-wan22`, ~6s for `runpod-wan`, ~12s for `fal-seedance`), `_target_durations()` splits it into multiple parts. Before this change, each part was generated with the **same prompt** — only the seed varied. This produced near-identical clips that, when concatenated, created a visually looped or repetitive result for any shot longer than 5-10 seconds.

The root cause: a single whole-shot description cannot capture temporal variation within a long shot, so all split parts converge on the same visual.

# Decision

Generate **per-segment descriptions** for long shots (>= 8s) during encode stage 2. For each such shot, split its keyframes into two halves and make one additional Gemini call per half, providing the whole-shot description as context. Store the resulting descriptions as `temporal_segments` in `prompts.json`.

During decode, when a shot is split into N parts, each part selects its description by proportional mapping to the available segments:

```
seg_idx = round(part_idx * (len(temporal_segs) - 1) / max(N - 1, 1))
```

The strategy's own `format_prompt()` is then called on the selected segment description, so model-specific formatting (Wan vs Seedance vs generic) applies transparently.

A **fallback** applies when `temporal_segments` is absent (e.g., prompts generated before this change): generic temporal progression cues are appended to the prompt ("Beginning of the action.", "The action continues.", "The action concludes.").

# Consequences

**Good:**
- Each split clip gets a meaningfully different prompt derived from what actually happens in that time window
- Whole-shot context is preserved — segment descriptions remain coherent with the overall shot
- Backward compatible: old prompts.json files work via the generic-cue fallback
- Strategy-agnostic: works for any strategy that uses `entry` in its generate() call
- Adds 2 Gemini Flash Lite calls per long shot (~$0.001-0.003 each) — negligible extra encode cost

**Bad:**
- Encode stage 2 now makes more API calls for long shots (2 extra per long shot)
- `temporal_segments` are always 2 halves regardless of how many decode parts result; very long shots split into 3+ parts still get good variation, but the two outer halves share segment descriptions for mid parts

**Operational:**
- Existing encoded projects can be re-encoded from stage 2 to get `temporal_segments` (resume skips already-processed shots)
- Shots < 8s are unaffected

# Alternatives Considered

- **Generic temporal cues only** (original task spec): Append "Beginning of the action.", "The action continues.", "The action concludes." without looking at actual footage. Simpler, but signals rather than encodes temporal content — less effective for shots with real visual change.
- **Segment descriptions during decode**: Run vision analysis in the decode loop instead of encode. Rejected: violates the stateless CLI stage principle (ADR-002); decode should be generation-only and avoid vision API calls.
- **More than 2 segments**: Generate 3 thirds for very long shots. Deferred — 2 halves cover the main use case (2-part splits) and mapping works for more parts.
- **Re-extracting keyframes per segment**: Extract new keyframes at finer time intervals for each segment during encode stage 1. More accurate but requires changes to stage 1 and significantly more disk space. Deferred.
