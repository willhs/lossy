---
id: task-0005
type: spec
purpose: "Generate per-shot audio from sound descriptions during decode and mix into the reconstructed film."
tags: ["audio", "decode", "fal"]
related:
  - "docs/design/adr/002-stateless-cli-pipeline.md"
  - "docs/design/adr/005-yamnet-audio-classification.md"
  - "docs/tasks/0002-audio-encoding/task.md"
  - "docs/design/architecture.md"
created: 2026-03-21
updated: 2026-03-21
---

# Audio Generation v1

## Goal

Generate audio clips from the `sound` descriptions in `prompts.json` and mix them into the reconstructed film, so the decoder output is no longer silent.

## Context

The encoding pipeline already classifies audio via YAMNet and produces rich `sound` descriptions per shot (e.g., "Triumphant orchestral fanfare with brass and timpani"). The decoder generates video clips but produces silent output. The `sound` field is captured but unused during decode.

fal.ai (already used for Seedance video generation) offers several text-to-audio models. ElevenLabs Sound Effects via fal.ai ($0.002/sec) gives the best quality for mixed SFX/ambient content with duration control. MMAudio V2 ($0.001/sec) is cheaper and also supports video-to-audio. Stable Audio 2.5 ($0.20 flat) is strongest for music but expensive for short shots.

## Scope

### Must Do

- New `audio.py` CLI following the stateless stage pattern (ADR-002)
- Strategy pattern for audio generation backends (mirrors decode.py)
- At least one working strategy (ElevenLabs via fal.ai)
- Read `prompts.json`, generate one audio clip per shot from the `sound` field
- Duration-match audio to shot duration
- Save audio clips to `audio/<strategy>/` directory
- Progress tracking and resume support (like decode.py)
- Stitch step: mix all per-shot audio into a single track and mux with the reconstructed video
- Pipeline.py integration (new stage between decode and stitch, or alongside stitch)

### Might Do

- MMAudio V2 as a second/cheaper strategy
- Video-to-audio mode (feed generated video clips to MMAudio V2 instead of text prompts)
- Crossfade between adjacent audio clips for smoother transitions
- Volume normalization across clips

### Won't Do

- Speech/dialogue synthesis (future task, depends on WhisperX encoding)
- Music vs SFX routing to different models per shot
- Audio effects processing (reverb, EQ, etc.)
- Real-time audio preview

## Constraints

- No new dependencies beyond what fal_client already provides (audio files handled via ffmpeg)
- Audio output format: WAV or FLAC intermediate, final mux via FFmpeg
- Must work with existing prompts.json format -- no encoder changes needed
- Cost target: < $0.01/shot for a typical 5-second clip

## References

- [ADR-002: Stateless CLI Pipeline](docs/design/adr/002-stateless-cli-pipeline.md) -- stage pattern to follow
- [ADR-005: YAMNet Audio Classification](docs/design/adr/005-yamnet-audio-classification.md) -- how sound descriptions are produced
- [Task 0002: Audio Encoding](docs/tasks/0002-audio-encoding/task.md) -- the encoding side that feeds this task

## Success Criteria

- [ ] `python audio.py output/film --strategy elevenlabs` generates audio clips for all shots
- [ ] Audio clips are duration-matched to shot durations (within 1 second)
- [ ] `python audio.py output/film --strategy elevenlabs --stitch` produces a muxed video with audio
- [ ] Pipeline.py can chain audio generation as a stage
- [ ] Full film audio generation costs < $15 (~1,150 shots)
