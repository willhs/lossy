---
id: task-0008
type: spec
purpose: "Split decode.py into focused modules to make adding new strategies easier."
tags: ["refactor", "decode", "architecture"]
related:
  - "docs/design/adr/002-stateless-cli-pipeline.md"
created: 2026-03-23
updated: 2026-03-23
---

# Refactor decode.py

## Goal

Break decode.py (2022 lines) into focused modules so adding a new video or audio backend means working in one ~450-line file instead of a 2000-line monolith.

## Context

Every new feature task (7 so far) adds strategies or capabilities to decode.py, requiring reading the entire file to find the right insertion point. The file mixes four distinct concerns: prompt formatting, video strategies, audio strategies, and FFmpeg stitching.

## Result

Split into 5 files following the existing flat module convention (`runpod_pod.py` as precedent):

| File | Lines | Contents |
|------|-------|----------|
| `decode.py` | 512 | CLI, dataclasses, run loops, re-exports |
| `prompt_format.py` | 170 | CAMERA_TERMS, format_prompt, _format_prompt_wan, _format_prompt_seedance |
| `strategies_video.py` | 458 | GenerationStrategy + 4 video backends |
| `strategies_audio.py` | 450 | AudioStrategy + 3 audio backends + SpeechStrategy |
| `stitch.py` | 456 | stitch_clips, _stitch_audio, _stitch_speech, _probe_duration |

decode.py re-exports all public symbols so `from decode import X` continues to work unchanged. test_decode.py required zero modifications.

Additionally deduplicated `_target_durations()` across audio strategies via a shared `_split_duration()` helper.
