---
id: task-0006
type: spec
purpose: "Generate a voice/dialogue track from subtitle data and mix it with the SFX/ambient audio track."
tags: ["speech", "dialogue", "decode", "fal", "elevenlabs"]
related:
  - "docs/design/adr/002-stateless-cli-pipeline.md"
  - "docs/tasks/0005-audio-generation/task.md"
  - "docs/tasks/0002-audio-encoding/task.md"
created: 2026-03-21
updated: 2026-03-21
---

# Speech Generation from Subtitles

## Goal

Generate a time-aligned voice/dialogue track from subtitle data already extracted by the encoder, and mix it with the existing SFX/ambient audio track so the reconstructed film has both sound effects and spoken dialogue.

## Context

The encoder (stage 2) already extracts subtitles via SRT parsing, aligns them to shots via `align_subtitles_to_shots()`, and stores a `dialogue` field (currently `string[]`) in each prompts.json entry. The audio pipeline already generates SFX/ambient audio per shot via `AudioStrategy`. Speech synthesis is the missing piece for a complete audio reconstruction.

The existing audio infrastructure provides a clear pattern to follow: `AudioStrategy` base class with `ElevenLabsStrategy` and `MMAudioStrategy` implementations, progress tracking via JSON, and FFmpeg-based stitching/muxing.

fal.ai offers ElevenLabs TTS Turbo v2.5 at ~$0.05/1k chars, which provides fast, natural-sounding speech synthesis.

## Scope

### Must Do

- Enrich `dialogue` field in prompts.json from `string[]` to `object[]` with `{text, start_s, end_s}` (offsets relative to shot start) -- requires changes to `align_subtitles_to_shots()` to preserve SRT timestamps
- New `SpeechStrategy` class in decode.py parallel to `AudioStrategy`, calling `fal-ai/elevenlabs/tts/turbo-v2.5`
- `run_speech()` function that iterates prompts, extracts dialogue + timestamps, calls SpeechStrategy per line, tracks progress in `speech_progress.json`
- `_stitch_speech()` that uses FFmpeg `adelay` to place TTS clips at SRT timestamp offsets within each shot, pads to shot duration, and concatenates into `speech_track.wav`
- Mix speech + SFX during mux: extend `stitch_clips()` to combine `audio_track.wav` + `speech_track.wav` via FFmpeg `amix`, then mux into final MP4
- Pipeline.py integration: new stage between audio and stitch with `--speech-voice` CLI arg (default: "Roger")
- Speech toggle in comparator: 4th audio state cycle (generated -> generated+speech -> original -> off)
- Mock-based tests for SpeechStrategy, run_speech(), timestamp alignment enrichment, and stitch mixing

### Might Do

- Multi-voice support: map different subtitle speakers to different ElevenLabs voices
- Subtitle speaker diarization in the encoder (who said what)
- Speech rate/pacing adjustment to match original timing more closely

### Won't Do

- Lip-sync or mouth animation in the generated video
- Emotion/tone detection from subtitle context
- Real-time speech preview
- Non-English language support (first pass)

## Constraints

- No new dependencies beyond what fal_client already provides
- Single narrator voice to start ("Roger"), architecture ready for multi-voice later
- Speech and SFX tracks kept separate until final mix (independently toggleable in comparator)
- Must preserve backward compatibility with existing prompts.json (entries without enriched dialogue should still work)
- Cost target: ~$0.05/1k chars via fal-ai/elevenlabs TTS

## Subtasks

1. **Enrich dialogue field with SRT timestamps** -- change `align_subtitles_to_shots()` to return `{text, start_s, end_s}` (relative to shot start) instead of plain strings
2. **Create SpeechStrategy class** in decode.py -- parallel to AudioStrategy, calls `fal-ai/elevenlabs/tts/turbo-v2.5`, returns result with path, duration, and shot-relative offset
3. **Add run_speech() function** -- iterates prompts, extracts dialogue + timestamps, calls SpeechStrategy per line, tracks progress in `speech_progress.json`
4. **Add speech stage to pipeline.py** -- new stage between audio and stitch, CLI args: `--speech-voice` (default Roger)
5. **Time-align speech clips in stitch** -- new `_stitch_speech()` using FFmpeg `adelay` to place clips at SRT offsets, pad to shot duration, concatenate into `speech_track.wav`
6. **Mix speech + SFX during mux** -- extend `stitch_clips()` to combine `audio_track.wav` + `speech_track.wav` via FFmpeg `amix`, then mux into final MP4
7. **Add speech toggle to comparator** -- 4th audio state: generated -> generated+speech -> original -> off, serve speech track as separate audio element
8. **Tests** -- mock-based tests for SpeechStrategy, run_speech(), timestamp alignment enrichment, and stitch mixing

## References

- [ADR-002: Stateless CLI Pipeline](docs/design/adr/002-stateless-cli-pipeline.md) -- stage pattern to follow
- [Task 0005: Audio Generation](docs/tasks/0005-audio-generation/task.md) -- SFX audio pipeline this builds on
- [Task 0002: Audio Encoding](docs/tasks/0002-audio-encoding/task.md) -- subtitle extraction in encoder

## Success Criteria

- [ ] `align_subtitles_to_shots()` returns enriched dialogue objects with `{text, start_s, end_s}`
- [ ] `python decode.py output/film --strategy fal-seedance --speech --speech-voice Roger` generates speech clips for all shots with dialogue
- [ ] Speech clips are time-aligned to original subtitle positions within each shot
- [ ] `python decode.py output/film --strategy fal-seedance --stitch` produces a final video with mixed SFX + speech audio
- [ ] Comparator supports 4-state audio toggle including speech
- [ ] Tests pass for speech strategy, run_speech, timestamp enrichment, and stitch mixing
