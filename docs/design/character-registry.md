---
type: spec
spec_id: SPEC-200
status: Active
purpose: Requirements for encode stage 3 — character registry extraction from shot subjects.
---

# Character Registry (Encode Stage 3)

**ID**: SPEC-200

## Overview

Stage 3 reads all `description.subjects` fields from `prompts.json`, sends them to Gemini in a single call, and produces a `characters.json` sidecar with named characters, canonical descriptions, and shot appearances. This registry drives portrait generation and VACE reference conditioning downstream.

## Output Structure

<!-- REQ-001 to REQ-009 -->
**REQ-001**: Stage 3 shall write a `characters.json` file in the output directory.
**REQ-002**: `characters.json` shall contain a `"characters"` array at the top level.
**REQ-003**: Each character entry shall have a `"name"` field: lowercase, underscores, no spaces.
**REQ-004**: Each character entry shall have a `"display_name"` field: human-readable name.
**REQ-005**: Each character entry shall have a `"description"` field: canonical appearance text.
**REQ-006**: Each character entry shall have a `"shots"` field: a list of integer shot indices.
**REQ-007**: Shot indices in the `"shots"` field shall correspond to indices present in `prompts.json`.

## Character Selection

<!-- REQ-010 to REQ-019 -->
**REQ-010**: Only characters appearing in at least 2 shots shall be included.
**REQ-011**: The registry shall favor the most prominent characters by shot count. The unsupervised path instructs Gemini to return ~5 characters; the TMDB-seeded path has no fixed cap and typically returns 10-20 characters drawn from the TMDB cast list.
**REQ-012**: Characters shall be the most prominent by shot count.
**REQ-013**: Different descriptions of the same character across shots shall be merged into one entry.

## Shot Assignment Refinement

<!-- REQ-030 to REQ-039 -->
**REQ-030**: After initial character extraction, stage 3 shall refine shot assignments by matching unassigned shots against the character registry.
**REQ-031**: An unassigned shot whose subjects description matches a character shall be added to that character's shots list.
**REQ-032**: Refinement shall not remove any existing shot assignments.

## Input Handling

<!-- REQ-020 to REQ-029 -->
**REQ-020**: Stage 3 shall support v1 format (flat array) `prompts.json`.
**REQ-021**: Stage 3 shall support v2 format (`{"format": "v2", "shots": [...]}`) `prompts.json`.
**REQ-022**: Stage 3 shall exit with an error if `prompts.json` does not exist.
**REQ-023**: Stage 3 shall exit with an error if no subjects are found in any shot.
**REQ-024**: Stage 3 shall exit with an error if `GEMINI_API_KEY` is not set.
