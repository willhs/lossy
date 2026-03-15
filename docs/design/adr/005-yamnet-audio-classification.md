---
id: adr-005-yamnet-audio-classification
type: decision
purpose: "Record the decision to use YAMNet for audio classification as preprocessing context for Gemini descriptions."
scope: ["design", "encoder"]
non_goals: []
tags: ["adr", "encoder", "audio", "yamnet"]
related: ["design/architecture.md", "research/0002-shot-to-prompt/research.md"]
---

# Context

Audio is 38% of the source file but was completely ignored by the encoder. Scenes with critical audio (opening fanfare, lightsaber hums, explosions, tense silence) were indistinguishable in the text manifest. Even though the decoder produces silent video, audio metadata improves visual reconstruction -- video generation models respond to mood and atmosphere cues.

We needed to decide: (A) how to classify the audio, and (B) how to represent it in the manifest.

# Decision

Use **YAMNet** (TensorFlow Hub) for audio classification, feeding the labels as text context into the existing **Gemini** vision API call. Gemini produces a natural-language `sound` description field per shot.

This mirrors the camera motion pattern: optical flow produces a label ("pan right") that feeds into the Gemini prompt as context. YAMNet produces audio labels ("Orchestra, Explosion, Rumble") that feed in the same way.

YAMNet's 521 AudioSet classes are grouped into 6 high-level buckets:
- **speech** (0-23, 33-66): human vocal and body sounds
- **music** (24-32, 132-276): singing, instruments, genres
- **ambient** (277-293): wind, water, thunder, fire
- **effects** (294-493): vehicles, mechanical, weapons, impacts
- **silence** (494): silence
- **other** (67-131, 495-520): animals, signal artifacts

Per-shot output: dominant bucket + top-N class labels, cached to `audio_labels.json`.

# Consequences

**Good:**
- No extra API cost -- audio labels are a few text tokens added to the existing Gemini call
- Gemini synthesizes labels + visual keyframes into rich natural-language descriptions
- Matches the established preprocessing pattern (subtitles, camera motion, now audio)
- YAMNet is lightweight, runs on CPU in ~1-2 minutes for a 2-hour film

**Bad:**
- Adds TensorFlow as a dependency (~500MB install)
- Audio description quality depends on Gemini interpreting YAMNet labels correctly
- YAMNet classifies in ~0.48s windows -- very short shots may only get 1-2 classification frames
- No speaker diarization (requires WhisperX, deferred to v2)

# Alternatives Considered

- **(A) Standalone audio manifest with bucket/labels only**: Raw YAMNet labels are too thin -- "effects: Explosion, Rumble" is tagging, not describing. Would need a follow-on task to wire into prompts anyway.
- **(B) Send audio clips to Gemini directly**: Gemini supports audio input natively. Richest option but adds a separate API call per shot. Unnecessary when labels-as-context produces good results.
- **(C) WhisperX for transcription + speaker diarization**: High value for dialogue-heavy films but heavy dependency (Whisper, wav2vec2, pyannote.audio). Deferred -- existing SRT subtitle extraction covers dialogue for v1.
