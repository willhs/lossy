---
id: experiment-0006
type: experiment
purpose: "Compare Kokoro (self-hosted, open-weight) against ElevenLabs Turbo v2.5 (paid API) for per-character dialogue TTS, to see whether the paid API can be dropped."
tags: ["experiment", "tts", "speech", "kokoro", "elevenlabs", "voices", "cost"]
related:
  - "../../tasks/0006-speech-generation/plan.md"
---

# Experiment 0006: Kokoro vs ElevenLabs TTS

## Hypothesis

Kokoro (open-weight, self-hosted, MIT-licensed) can produce dialogue TTS good
enough to replace ElevenLabs Turbo v2.5 for at least some characters, removing
a paid API dependency and making a good r/LocalLLaMA talking point alongside
the reconstruction itself.

## Metrics & Method

`strategies_audio.py` now has two interchangeable speech backends behind the
same interface (`generate(text, speech_dir, shot_index, line_index, offset_s)`):

- `SpeechStrategy` — existing ElevenLabs Turbo v2.5 via fal.ai, $0.05/1k chars.
- `KokoroSpeechStrategy` — new, wraps the local `kokoro` pipeline, $0 marginal
  cost, requires `uv add kokoro kokoro-onnx soundfile` manually -- not a declared
  project dependency, since its transitive deps are heavy and this trial is
  optional/untried per the task notes.

Both are selectable via `decode.py --speech --speech-strategy {elevenlabs,kokoro}`.

Planned comparison, once the extra is installed on a machine with the model
weights available:

1. Generate the same set of sample lines with both backends — the per-character
   samples produced by `voice_casting.py --sample` are the natural shared set.
2. Compare: wall-clock time per line, subjective naturalness/prosody, and
   whether Kokoro's built-in voice bank (`af_*`/`am_*` names) has plausible
   fits for the proposed ElevenLabs casting.
3. Compare cost at full-film scale: ElevenLabs at ~$0.05/1k chars vs Kokoro's
   $0 marginal cost (compute-only, already paid for via the RunPod pod used
   for video generation).

## Results

Not yet run — Kokoro requires downloading model weights, which wasn't done in
this session (the mandatory ElevenLabs casting/attribution work stayed inside
the ~$5 budget without it, and installing a new heavy dependency needs a
deliberate `uv add kokoro kokoro-onnx soundfile` before it can be exercised for real). The
interface, CLI wiring, and cost-tracking are in place and unit-tested; the
open item is running an actual side-by-side listen.

## Interpretation

Mechanism-first: the goal for this task was to make swapping speech backends
trivial, not to commit to one. That's done — `--speech-strategy kokoro` is a
one-flag opt-in with no changes to `run_speech`, `speakers.json`, or
`voice_map.json` needed (voice names are just backend-specific strings in the
same map).

## Next Steps

- `uv add kokoro kokoro-onnx soundfile` on a machine with spare disk/compute, then run the
  planned comparison above on the `sw_r2_leia` test clip's attributed lines.
- If Kokoro quality holds up for at least the non-lead/background characters,
  consider defaulting new films to a mixed cast (Kokoro for minor characters,
  ElevenLabs for leads) to cut cost on the full Star Wars IV run without
  touching quality where it matters most.
