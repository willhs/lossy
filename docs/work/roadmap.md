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

- **Full-run go/no-go on `runpod-ltx2`** — the dress rehearsal (shots 794-881, research 0023) gave a **conditional go**: ~$15-25 projected for all 2069 shots, all four blocking defects fixed (auto re-provision on pod failure, idle-pod timeout, costume-free canonical descriptions, encode-fingerprint pinning against stale artifacts). Still open before committing to the full run:
  - Defect 5 — identity bleed when >=3 characters are stacked in one shot (196/2069 shots, 9.5%); mitigation is capping identity blocks to the two most prominent characters.
  - Defect 6 — non-humanoid characters (C-3PO) render as a human in costume; reword the registry description to "mechanical robot", not "humanoid".
  - Defect 7 — speaker attribution has a ~1.5% measured error floor (confirmed wrong-scene attributions); mitigation is routing low-confidence lines to a narrator voice.
  - Defect 11/12 — 2.3% ambience-shot failure rate (not retried) and a final mux peaking at 98.8% FS despite the limiter; worth a look before mastering a full film.

## Done

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
