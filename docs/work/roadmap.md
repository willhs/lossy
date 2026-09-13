---
id: roadmap
type: note
purpose: "Outline the Now/Next/Later priorities for the project."
scope: ["work"]
non_goals: []
tags: ["roadmap", "planning"]
related: []
---

## Now

- **Resume the full run on `runpod-ltx2` for the remaining ~2000 shots.** Defects 5 and 6
  landed 2026-09-12 (`aac8ee2`), the last gate the 2026-09-06 GO decision set — nothing
  code-side blocks the run anymore. Defects 7 and 11/12 remain accepted as known v1
  limits (7 and 11/12 get revisited before mastering, not before generating).

## Done

- ~~**Fix defects 5 + 6**~~ — `aac8ee2` (2026-09-12): `CharacterIdentityMixin._prepend_identity`
  now caps identity blocks to the two most prominent characters per shot (defect 5,
  identity bleed on 196/2069 shots); `IDENTITY_DESCRIPTION_SPEC` now describes
  droids/robots as mechanical rather than "humanoid" (defect 6). Both spot-checked
  against `star_wars_iv_v2/characters.json`.

- ~~**Pick a source film**~~ — Star Wars Episode IV (1080p, ~2h, 1,161 shots at time of pick; 2,069 shots after the July re-encode).
- ~~**Build the encoder**~~ — PySceneDetect shot detection + Gemini Flash-Lite prompt generation. Full pipeline tested.
- ~~**Build the decoder**~~ — Replicate Wan 2.2 Fast (480p, ~$0.05/clip), FFmpeg stitcher with speed-adjustment. 15-clip test batch successful.
- ~~**Fix speed-adjustment issues**~~ — added Seedance strategy with 2-12s duration control. Strategy pattern in decoder supports swappable backends.
- ~~**Audio generation**~~ — per-shot audio from sound descriptions via ElevenLabs/MMAudio (fal.ai). Audio muxed into reconstructed film during stitch.
- ~~**Cost tracking**~~ — per-stage API cost tracking (Gemini tokens, video/audio generation). Pipeline writes unified `costs.json` to output dir.
- ~~**Build the comparator**~~ — side-by-side view via `tools/compare.html` + dev server.
- ~~**Trial Seedance strategy**~~ — Seedance Fast duration control validated; A/B test on shots 10–24 (see experiments/0002, 0003). Wan-optimized prompt formatter confirmed better for subject accuracy.
- ~~**Character continuity pipeline**~~ — encode stage 3 (character registry via Gemini), portrait generation (fal.ai Flux Schnell), VACE strategy (RunPodVaceStrategy with reference conditioning + T2V fallback + prompt identity enrichment). Shot assignment refinement pass added. Superseded as the active continuity path by `CharacterIdentityMixin` + I2V chaining on `runpod-ltx2` (below); the VACE continuity rerun was never picked back up.
- ~~**LTX-2 vs Wan22 trial + dress rehearsal**~~ — `runpod-ltx2` strategy built and live-verified (research 0022); end-to-end dress rehearsal on an 88-shot segment (research 0023) found and fixed real defects (resume dropping split shots, no identity conditioning, no I2V chaining causing internal jump-cuts, narrow GPU fallback chain) and produced the go/no-go numbers above. LTX-2 at matched resolution runs ~2.3x faster than Wan22, not the ~3x an earlier unmatched trial suggested.
- ~~**Pod reliability**~~ — automatic re-provisioning on pod-boot failure (`RunPodSession.with_setup_retry`), cloud fallback (community exhausted before secure), exit-time stray-pod sweep, and a one-retry-per-clip recovery for the "server up but not yet sampleable" failure mode.

## Next

- **Ship the full run** once the "Now" defects are addressed or explicitly accepted — run decoder on all 2,069 shots of Star Wars IV with `runpod-ltx2`. Currently only the 88-shot rehearsal segment has been generated end-to-end.
- **TI2V-5B evaluation** — kept as a fallback path if `runpod-ltx2` quality turns out insufficient at full-run scale (see research/0012). Lower priority now that the rehearsal gave LTX-2 a conditional go.

## Later

- **Audio encoding improvements** — WhisperX for dialogue/speaker diarization. Encode dialogue into the prompt manifest for future speech synthesis.
- **Write the blog post** — walk through the process, show results, reflect on what language preserves and what it loses.
- **Try variations** — different description detail levels, different video gen models, manual vs automated encoding. See how results change.
- **Compression ratio gag** — calculate the "bitrate" of the text manifest vs the original file size. Present it seriously.
- **Seedance 2.0** — fal.ai multimodal reference API (expected mid-2026) will allow character reference images per generation call without VACE or LoRA. Drop-in replacement for continuity once available.
