---
type: spec
spec_id: SPEC-300
status: Active
purpose: Per-segment prompt variation for long shots split across multiple generated clips.
---

# Temporal Prompt Variation for Split Shots

**ID**: SPEC-300

## Overview

When a shot exceeds a video generation strategy's max clip duration, it is split into multiple parts. Without intervention, all parts would be generated from the same prompt, producing near-identical clips. This spec defines the behavior for varying prompts across split parts so that each generated clip is visually distinct.

There are two mechanisms:
1. **Encoded temporal segments** (`temporal_segments` in shots.json): Per-segment descriptions generated from the actual keyframes of each half of the shot during encode stage 2.
2. **Generic fallback cues**: When encoded segments are absent, a simple text cue is appended based on temporal position ("Beginning of the action.", etc.).

## Encode: Temporal Segment Generation

<!-- REQ-001 to REQ-009 -->
**REQ-001**: For any shot with `duration_s >= 8.0`, the encoder shall attempt to generate `temporal_segments` alongside the main shot description.

**REQ-002**: `temporal_segments` shall be a list of `n_segments` description dicts, each in the same JSON schema as the main shot `description`, where `n_segments = max(2, ceil(duration_s / 12.0))`, capped at the number of available keyframes.

**REQ-003**: Keyframes shall be divided into `n_segments` groups as evenly as possible using the slice `frame_files[i*N//n_segments : (i+1)*N//n_segments]` for each segment index `i`, where `N` is the total keyframe count.

**REQ-004**: Each segment shall be described from its own keyframe group so that each covers a distinct temporal window of the shot.

**REQ-005**: The whole-shot description shall be provided as context when generating each segment, so the segment description stays coherent with the full shot.

**REQ-006**: If fewer than 2 keyframes are available for the shot, `temporal_segments` shall not be generated and the field shall be absent from the shots.json entry.

**REQ-007**: If a Gemini call fails for any segment, `temporal_segments` shall not be added (partial segments shall not be stored).

**REQ-008**: Shots with `duration_s < 8.0` shall not have `temporal_segments` generated; the encode cost for short shots is unchanged.

## Decode: Per-Part Prompt Selection

<!-- REQ-010 to REQ-019 -->
**REQ-010**: When a shot is split into N > 1 parts, each part shall use a distinct prompt rather than the whole-shot prompt.

**REQ-011**: If `temporal_segments` is present in the shot entry, the per-part prompt shall be derived from the segment at index `round(part_idx * (len(segments) - 1) / (N - 1))`, formatted through the strategy's `format_prompt()`.

**REQ-012**: If `temporal_segments` is absent, the per-part prompt shall be the whole-shot prompt with a generic temporal cue appended (see REQ-020 to REQ-022).

**REQ-013**: For a single-part shot (N = 1), the prompt shall be returned unchanged — no variation applied.

**REQ-014**: The segment selection mapping shall cover the full range: part 0 always maps to segment 0, part N-1 always maps to the last segment.

## Decode: Generic Fallback Cues

<!-- REQ-020 to REQ-029 -->
**REQ-020**: The first part of a split shot (part_index == 0) shall have "Beginning of the action." appended.

**REQ-021**: The last part of a split shot (part_index == total_parts - 1, total_parts >= 2) shall have "The action concludes." appended.

**REQ-022**: Any middle part (0 < part_index < total_parts - 1) shall have "The action continues." appended.

**REQ-023**: The cue shall be separated from the base prompt by a single space.

**REQ-024**: When total_parts <= 1, the prompt shall be returned unchanged.
