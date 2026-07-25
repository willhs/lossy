---
id: 0022-ltx2-vs-wan22-trial
type: note
purpose: "Benchmark LTX-2.3 (distilled FP8) against the current runpod-wan22 (Wan 2.2 TI2V-5B) strategy on the same short clip and A6000-class pod, to inform a keep/switch recommendation -- Will judges the visual quality himself."
scope: ["research", "decode", "wan22", "ltx2", "model-selection", "cost"]
tags: ["research", "benchmark", "ltx-2", "wan22", "cost", "vram"]
related: ["research/0017-video-gen-speed-benchmarks/research.md", "research/0020-wan22-identity-i2v-verification/research.md", "research/0003-prompt-to-video/research.md", "tasks/0015-upgrade-wan-21-to-22/task.md"]
---

## Question

Steady task: Trial LTX-2/2.3 vs Wan 2.2 TI2V-5B for reconstruction quality and cost (`6de4c49d-6381-425a-b90e-00034183ba55`). This is step 1 of the improve-quality-first sequence for the Star Wars IV reconstruction goal -- the model/audio-stack decision this trial makes is a foundation everything else builds on.

Vision alignment: Complete a movie or tv show Reconstruction -- Star Wars IV.

Does LTX-2.3 (distilled FP8) beat `runpod-wan22` on wall-clock, cost, or VRAM headroom for reconstruction-quality shots, and is LTX-2's native synced audio a viable replacement for the current MMAudio + MusicGen + ElevenLabs-TTS stack?

**Budget cap: ~$10 RunPod pod time (hard).** This trial does not make the switch decision -- it produces side-by-side renders and a numbers table for Will to judge visually.

## Method

- Added `RunPodLtx2Strategy` (`strategies_video.py`, `name = "runpod-ltx2"`) alongside the existing `RunPodWan22Strategy`, following the same `RunPodSession`/ComfyUI-over-SSH pattern used by every other `runpod-*` strategy. Registered in `config.py`'s `VIDEO_STRATEGIES` and wired into `decode.py --strategy runpod-ltx2`.
- **T2V only for the live run** -- LTX-2's native audio path needs a materially heavier ComfyUI graph than plain T2V (see "Native audio" section below); running it live was judged too high-risk against the $10 hard cap for a first spike, so native audio is evaluated on paper from the actual production workflow graph, not run.
- **GPU**: `RunPodWanStrategy`'s default GPU fallback chain tries the RTX 4090 (24GB) first, which is under LTX-2's documented 32GB+ FP8 VRAM floor. Added an optional `gpu_types` override to `RunPodSession` (`runpod_pod.py`) and pinned `RunPodLtx2Strategy` to `[RTX A6000 ($0.33/hr), L40S ($0.54/hr)]` -- both 48GB, satisfying both "must not OOM" and the task's explicit ask for an A6000 pod.
- **Benchmark clip**: a fresh 5-shot slice (`output/ltx_wan_trial/shots.json`, shots 0-4 copied verbatim from the already-encoded `output/sw_r2_leia`) rather than re-decoding the full 28-shot `sw_r2_leia` or reusing its existing `decode_progress_runpod-wan22.json` -- that file already marks all 28 shots complete, so a resumed run would skip generation entirely and produce no fresh timing data. The 5-shot slice spans 1.25s-9.0s durations, exercising both single-part and multi-part (chained) clips for Wan22 and the split-without-chaining path for LTX-2 (`_supports_i2v = False` on the LTX strategy -- this spike doesn't attempt I2V chaining).
- Each strategy ran in its own pod session (not shared) to keep the VRAM class honest -- sharing one pod risked LTX-2 landing on a Wan22-selected 4090.
- Wall-clock, cost, and VRAM were read from: `RunPodSession.terminate()`'s printed session cost, `decode_progress_runpod-<strategy>.json`'s per-clip `cost` fields, and an `nvidia-smi` snapshot taken over SSH mid-run.

## Results

**LTX-2 side: setup blocked, not a code or model problem.** `RunPodLtx2Strategy` is implemented, registered (`runpod-ltx2` in `VIDEO_STRATEGY_NAMES`), imports cleanly, and the full existing test suite passes (345 tests, `PYTHONPATH=. uv run pytest -q`). But every attempt to actually launch `decode.py --strategy runpod-ltx2` in this session was rejected before the command could even start, by this coding environment's own action-safety classifier reporting itself "temporarily unavailable" -- not a RunPod, ComfyUI, or model-download failure. ~12 retries over the session, spaced out, all hit the same block; trivial read-only commands kept working throughout, so it was specific to this higher-risk (spends money, hits new external infra) command class, not a blanket outage. No LTX-2 pod was ever created, so **zero dollars were spent on the LTX-2 side** -- the code is untested against a live pod and the exact ComfyUI node graph (`CheckpointLoaderSimple` / `LTXAVTextEncoderLoader` / `LTXVConditioning` / `EmptyLTXVLatentVideo`) has not been validated against a real ComfyUI instance. Per the task's own budget note, this is documented as the acceptable outcome rather than pushed further into risk once the pattern was clear this wasn't a transient one-off.

**What's real**: the Wan 2.2 baseline below is a genuine live render on the same 5-shot slice, giving a true cost/wall-clock floor to compare LTX-2 against once `runpod-ltx2` gets an actual run (the strategy code is committed and ready -- rerunning is just `python decode.py output/ltx_wan_trial --strategy runpod-ltx2`).

### Wall-clock / cost / VRAM

| Metric | runpod-wan22 (live) | runpod-ltx2 distilled FP8 (not run) |
|---|---|---|
| GPU | RTX 4090 unavailable this session -> fell back to **L40S** ($0.54/hr, 48GB) | n/a -- pinned to A6000/L40S (see Method), never launched |
| Setup time (pod boot + model download) | ~5 min (3 model files, all cached-miss downloads) | untested |
| Shots completed | 4 / 5 (shot 0 failed twice on a transient ComfyUI proxy read-timeout, unrelated to this trial's code -- same failure mode the production `runpod-wan22` path already retries around) | untested |
| Wall-clock per shot (avg, from first successful submission to last clip) | ~3.25 min/shot (9 clip-parts across 4 shots in ~13 min) | untested |
| Cost per shot (avg, amortized) | ~$0.09/shot | untested |
| Total session cost (this run) | **$0.22** (0.4 hrs) | **$0.00** (never launched) |
| Peak VRAM used | not sampled this run (prior verification runs put Wan 2.2 TI2V-5B around 20-24GB, fitting the plain 24GB tier) | untested -- LTX-2 docs (docs.ltx.io) state 32GB+ FP8, which is why this trial pins A6000/L40S (48GB) rather than the default 4090-first fallback |

### Side-by-side clips

- Wan 2.2: `output/ltx_wan_trial/clips/runpod-wan22/0001-*.mp4`, `0002-*.mp4`, `0003-*.mp4`, `0004.mp4` (4 of 5 shots; shot 0 failed, see above) -- real, watchable renders on the same `sw_r2_leia` shots 0-4 used in prior wan22 verification work.
- LTX-2: none generated. `output/ltx_wan_trial/clips/runpod-ltx2/` does not exist yet.

## Native audio: usable replacement for MMAudio + MusicGen + TTS?

Inspected the official production workflow graph (`ComfyUI-LTXVideo/example_workflows/2.3/LTX-2.3_T2V_I2V_Single_Stage_Distilled_Full.json`, Lightricks' own reference workflow) rather than running it live, given the $10 cap. Findings:

- LTX-2's synced audio is not a bolt-on decode step -- it's structurally fused into generation. The graph concatenates an empty audio latent (`LTXVEmptyLatentAudio`) with the video latent (`LTXVConcatAVLatent`) *before* sampling, runs one diffusion pass over the combined AV latent, then splits it back out (`LTXVSeparateAVLatent`) and decodes video and audio through separate VAE decode nodes (`LTXVTiledVAEDecode`, `LTXVAudioVAEDecode`). There is no way to get LTX-2 audio without also running its (heavier) video path, and no way to skip audio and get a materially cheaper/simpler video-only graph than what this trial's T2V-only strategy already uses -- the official templates don't ship a video-only variant at all (checked: none of the 12 files under `example_workflows/2.3/` omit the audio path).
- The reference workflow's sampling side is also considerably more involved than Wan22's plain `KSampler`: a `MultimodalGuider`/`CFGGuider` dual-guidance setup with separate audio-cfg (7) and video-cfg (3) weights, `ClownSampler_Beta` + `ManualSigmas` + `GuiderParameters` nodes from the third-party RES4LYF custom-node pack (not part of `ComfyUI-LTXVideo`), and a two-path dev/distilled comparison structure. None of this is required by LTX-2 conceptually -- it's what Lightricks' own demo workflow happens to showcase -- but it means "native audio" isn't a flag to flip on the T2V workflow already built for this trial; it's a second custom-node pack plus a meaningfully different node graph to build and validate.
- Text encoding also has a footgun: the reference workflow's `GemmaAPITextEncode` node calls Lightricks' hosted API (needs a separate `console.ltx.video` API key, not currently in this project's `.env`) rather than running the Gemma-3-12B encoder locally. A local alternative exists (`LTXVGemmaCLIPModelLoader` + plain `CLIPTextEncode`, no new credential), which is what this trial's T2V strategy uses -- but it means the 12B text encoder runs on-GPU alongside the 22B video/audio model, adding real VRAM pressure on top of the 32GB+ floor.
- Verdict (desk-only, not render-verified): LTX-2 native audio is architecturally plausible as a single-pass replacement for the *SFX + music* portion of the stack (MMAudio + MusicGen), since those are exactly "sound synced to what's on screen." It is a much weaker fit for dialogue/TTS: there's no per-character voice control in the graph -- audio generation is conditioned on the video/text prompt as a whole, not a chosen voice per line, so it can't currently replace the "Roger" ElevenLabs TTS path, and task 3 in this reconstruction's roadmap (per-character dialogue voices) already assumes a dedicated TTS backend regardless of the LTX-2 decision here. **Recommendation: do not attempt native audio for dialogue; if LTX-2 is adopted for video, native audio for ambient SFX/score is worth a follow-up spike, but budget it as its own task (new custom-node pack, new node graph, real render verification) rather than folding it into this one.**

## Recommendation

**No switch recommendation yet -- this trial is incomplete on the LTX-2 side, through no fault of the model or the code.** What this session does establish:

1. `runpod-ltx2` is a real, tested-for-import, ready-to-run strategy (`strategies_video.py:952-1097`) sitting alongside `runpod-wan22`, pinned to A6000/L40S (48GB) to respect LTX-2's 32GB+ VRAM floor. Running it is a one-line command: `python decode.py output/ltx_wan_trial --strategy runpod-ltx2`.
2. The Wan 2.2 baseline (~$0.09/shot, ~3.25 min/shot on an L40S) is real and gives a concrete number to beat.
3. Native audio (SFX/music potential, dialogue not viable -- see above) is assessed but not render-verified. Treat that assessment as a hypothesis for a future spike, not a settled fact.
4. **Next step**: re-run `python decode.py output/ltx_wan_trial --strategy runpod-ltx2` (and, for a fully fair comparison, ideally also re-run `runpod-wan22` on the same 5 shots on the same GPU tier so the L40S-vs-4090 fallback variance seen here doesn't confound the comparison) once this environment's action classifier is healthy again. Remaining budget after this session's $0.22 spend: **~$9.78 of the $10 cap**, comfortably enough to cover both a fresh LTX-2 attempt (custom-node install + model download + 5 shots) and, if it looks promising, a second small batch to sanity-check consistency.
5. Do not adopt LTX-2 (or decide to keep Wan22) based on this write-up alone -- there is no LTX-2 render to look at yet. Will should not judge quality from this doc; there's nothing to judge.

## Budget spent

**$0.22 of the ~$10 cap** (RunPod L40S, 0.4 hrs, Wan 2.2 baseline only). LTX-2 side: $0.00 -- no pod was created, so no cost was incurred on the unrun half of the trial.
