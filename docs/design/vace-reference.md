---
type: spec
spec_id: SPEC-220
status: Active
purpose: Requirements for VACE reference conditioning in video generation.
---

# VACE Reference Conditioning

**ID**: SPEC-220

## Overview

The `runpod-vace` strategy uses VACE-1.3B to generate video clips with character reference images. When a character is identified in a shot, their portrait is passed as a VACE reference image to anchor appearance. Shots without characters fall back to the T2V-1.3B workflow.

## Workflow Selection

<!-- REQ-001 to REQ-009 -->
**REQ-001**: Shots with an uploaded character portrait shall use the VACE workflow.
**REQ-002**: Shots without any character portrait shall fall back to the T2V workflow.
**REQ-003**: The VACE workflow shall use the `wan2.1_vace_1.3B_fp16.safetensors` diffusion model.
**REQ-004**: The VACE workflow shall include `WanVaceToVideo` and `TrimVideoLatent` nodes.
**REQ-005**: The VACE `WanVaceToVideo` strength shall be 1.0.

## Reference Image Handling

<!-- REQ-010 to REQ-019 -->
**REQ-010**: The primary character (first in the shot's character list) shall be used as the reference image.
**REQ-011**: The `LoadImage` node shall reference the portrait filename as uploaded to the pod.
**REQ-012**: Portraits shall be uploaded to the pod's ComfyUI input directory before clip generation.

## Prompt Enrichment

<!-- REQ-020 to REQ-029 -->
**REQ-020**: When a character is identified in a shot, the prompt sent to the workflow shall include the character's canonical name from `characters.json`.
**REQ-021**: When a character is identified in a shot, the prompt shall include the character's canonical description from `characters.json`.

## Strategy Integration

<!-- REQ-030 to REQ-039 -->
**REQ-030**: `runpod-vace` shall be a valid strategy choice in `decode.py` and `pipeline.py`.
**REQ-031**: `pipeline.py` shall include `encode3` in the stage list when strategy is `runpod-vace`.
**REQ-032**: `pipeline.py` shall skip `encode3` for non-VACE strategies.
