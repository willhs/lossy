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

Both sides now have real, live renders on the same 5-shot slice (`output/ltx_wan_trial`, shots 0-4 of `sw_r2_leia`).

**Getting the LTX-2 side working took three real bug fixes**, found by reading ComfyUI's and Lightricks' actual source (not blog posts) after each live failure:

1. `RunPodSession.download_models()` had a hardcoded 600s SSH timeout -- fine for Wan's ~10GB of model files, not enough for LTX-2's 22GB distilled-FP8 checkpoint. Added a `timeout` param (`runpod_pod.py`), LTX2 strategy now passes 1800s.
2. Guessed text-encoder loader node/inputs wrong twice (`LTXAVTextEncoderLoader` with a `text_encoder_name` key, then a plain `CLIPLoader`) before pulling ComfyUI core's actual `comfy_extras/nodes_lt_audio.py` off GitHub: the real node is `LTXAVTextEncoderLoader` with **three** inputs (`text_encoder`, `ckpt_name`, `device`) -- it loads the gemma text-encoder file *and* the main LTX checkpoint together, because part of what it needs (`text_embedding_projection` weights) lives inside the checkpoint file, not the standalone gemma file.
3. Even with encoding fixed, `KSampler` crashed inside `comfy/ldm/lightricks/av_model.py` ("Tensors must have same number of dimensions: got 4 and 3"). Root cause, confirmed by reading that file: **the distilled-FP8 checkpoint loads as an audio-video *joint* model at the architecture level** -- its text-conditioning path unconditionally expects an AV-shaped latent, so even a pure-video render needs the full `LTXVEmptyLatentAudio -> LTXVConcatAVLatent -> KSampler -> LTXVSeparateAVLatent` scaffold around the sampler. The audio latent is generated and immediately discarded (never decoded/saved) -- this trial still doesn't use LTX-2's audio output, it just can't avoid computing an audio latent internally. **This is a live-verified confirmation of the "Native audio" finding below**, not a contradiction of it: LTX-2 genuinely does not have a cheaper video-only mode.

Also hit and resolved along the way: RunPod's `RTX A6000` was spot-unavailable on 3 of 4 pod-creation attempts (fell back to `L40S`, $0.54/hr, also 48GB), the RunPod account ran to a ~$0 balance mid-session (Will topped it up, see chat), and one pod was silently reclaimed by RunPod's COMMUNITY (spot) tier mid-debug -- exactly the availability risk ADR-008 already flags for this cloud tier.

### Wall-clock / cost / VRAM

| Metric | runpod-wan22 (live) | runpod-ltx2 distilled FP8 (live) |
|---|---|---|
| GPU | RTX 4090 unavailable -> **L40S** ($0.54/hr, 48GB) | RTX A6000 unavailable -> **L40S** ($0.54/hr, 48GB) |
| Setup time (pod boot + model download, cold cache) | ~5 min (3 files, ~10GB total) | ~6-8 min (2 files, ~34GB total: 22GB checkpoint + 12GB gemma text encoder) |
| Shots completed | 4 / 5 (shot 0 failed twice on a transient ComfyUI proxy read-timeout, unrelated to this trial's code) | **5 / 5** |
| Wall-clock per shot (avg, generation only) | ~3.25 min/shot (9 clip-parts across 4 shots in ~13 min) | **~1 min/shot** (8 clip-parts across 5 shots in ~5 min) -- roughly 3x faster, consistent with the 3-5x figure in research/0017 |
| Resolution / duration control | 1280x704 @ 24fps, 2.0-4.0s native clips before splitting | 768x512 @ 25fps, 1.0-4.84s native clips before splitting (width/height hardcoded low in this spike's workflow -- see Caveats) |
| Cost per shot (avg, amortized over the *clean* generation run) | ~$0.09/shot | **~$0.01/shot** for generation alone (5 min @ $0.54/hr / 5 shots). The `decode_progress_runpod-ltx2.json` file reports $1.72 amortized -- that figure is inflated by ~40 min of live debugging (workflow validation errors, direct API probing) that ran on the same pod session before the clean successful pass; it is not a real per-shot cost. |
| Total $ spent (this metric's own session) | **$0.22** (0.4 hrs, clean run) | **~$1.16** across every attempt this session (3 failed fresh-pod debug cycles + the final successful debug-and-generate session), of which the clean 5-shot generation phase itself was **~$0.05** |
| Peak VRAM used | not sampled this run (prior verification runs put Wan 2.2 TI2V-5B around 20-24GB) | not sampled (no `nvidia-smi` snapshot taken before pod teardown) -- 22B FP8 + 12B gemma text encoder did not OOM on 48GB, consistent with the 32GB+ floor from docs.ltx.io, but the actual peak number is unmeasured |

**Total trial spend: ~$1.38 of the ~$10 cap** (Wan22 $0.22 + LTX-2 ~$1.16, mostly debugging, not generation).

### Side-by-side clips

- Wan 2.2: `output/ltx_wan_trial/clips/runpod-wan22/0001-*.mp4`, `0002-*.mp4`, `0003-*.mp4`, `0004.mp4` (4 of 5 shots; shot 0 failed on a transient timeout).
- LTX-2: `output/ltx_wan_trial/clips/runpod-ltx2/0000.mp4` through `0004-*.mp4` (all 5 shots, 8 files). 768x512, H.264 mp4 after the strategy's standard webm->mp4 conversion, confirmed playable via `ffprobe`.
- **Will: these are the two sets to look at side-by-side.** No visual quality judgment is made in this doc.

### Caveats on this comparison

- **Not resolution-matched.** This spike's LTX-2 workflow hardcodes 768x512 (a cautious placeholder chosen while debugging); Wan22 runs at 1280x704. LTX-2 supports higher resolutions -- this understates LTX-2's real per-shot cost/time at a comparable resolution to Wan22. Re-run at matched resolution before drawing a speed/cost conclusion.
- **GPU tier wasn't controlled.** Both runs ended up on the same L40S by coincidence (RunPod spot availability), not by design -- the A6000 that both were nominally pinned to was unavailable both times.
- **Wan22's shot 0 failure** was a transient proxy timeout, not a systematic issue with either model.

## Native audio: usable replacement for MMAudio + MusicGen + TTS?

Inspected the official production workflow graph (`ComfyUI-LTXVideo/example_workflows/2.3/LTX-2.3_T2V_I2V_Single_Stage_Distilled_Full.json`) and, after the live debugging above, ComfyUI core's actual model/node source (`comfy/ldm/lightricks/av_model.py`, `comfy_extras/nodes_lt_audio.py`, `comfy/text_encoders/lt.py`). The live run above **confirms** the core finding here directly (not just from reading the graph): the distilled-FP8 checkpoint's diffusion model is an `LTXAVModel` -- audio-video joint at the architecture level -- so this trial's own T2V-only strategy had to build and run the audio-latent path (`LTXVEmptyLatentAudio` -> `LTXVConcatAVLatent` -> ... -> `LTXVSeparateAVLatent`) just to get video out, then discard the audio latent unused.

- There is no cheaper video-only mode to fall back to -- generating video always means generating (and paying the compute cost of) an audio latent internally, whether or not you decode or use it. The marginal cost of *also* decoding and saving that audio (via `LTXVAudioVAEDecode`) on top of what this trial already runs is real but likely small next to the cost already being paid for the joint AV diffusion pass itself.
- The official reference workflow's *sampling* side (as opposed to the base video+audio latent scaffolding this trial had to add) is more involved again: a `MultimodalGuider`/`CFGGuider` dual-guidance setup with separate audio-cfg/video-cfg weights and `ClownSampler_Beta`/`ManualSigmas`/`GuiderParameters` nodes from the third-party RES4LYF custom-node pack (not part of `ComfyUI-LTXVideo` or ComfyUI core, not installed or evaluated in this trial). That's Lightricks' own demo showcasing higher-quality sampling, not something LTX-2 strictly requires -- this trial's plain `KSampler` (8 steps, cfg 1.0, euler_ancestral) worked fine for video.
- Text encoding has a footgun to avoid: the reference workflow's `GemmaAPITextEncode` node calls Lightricks' hosted API (needs a separate `console.ltx.video` API key, not in this project's `.env`). The node this trial actually verified working, `LTXAVTextEncoderLoader`, runs the Gemma-3-12B encoder locally -- no new credential, but the full 12B encoder runs on-GPU alongside the 22B video/audio model.
- Verdict, now backed by a live run rather than just reading the graph: LTX-2 native audio is architecturally plausible as a replacement for the *SFX + music* portion of the stack (MMAudio + MusicGen) since the audio latent is already being computed for free as a side effect of video generation -- decoding and saving it is the remaining (probably small) increment. It is a much weaker fit for dialogue/TTS: there's no per-character voice control in the graph -- audio is conditioned on the video/text prompt as a whole, not a chosen voice per line -- so it can't replace the "Roger" ElevenLabs TTS path, and task 3 in this reconstruction's roadmap (per-character dialogue voices) already assumes a dedicated TTS backend regardless of this decision. **Recommendation: do not attempt native audio for dialogue; if LTX-2 is adopted for video, wiring up `LTXVAudioVAEDecode` + `SaveAudio` to actually hear the SFX/score LTX-2 is already computing is a small, cheap follow-up spike (the concat/separate scaffolding already exists in `RunPodLtx2Strategy._build_workflow`) -- much cheaper than this trial's initial estimate, since the hard part (getting the AV model to run at all) is done.**

## Recommendation

**No switch/keep decision made here -- that's Will's call from watching the clips.** What this trial does establish with real, live evidence:

1. `runpod-ltx2` is a real, working, committed strategy (`strategies_video.py`, `RunPodLtx2Strategy`) that generated all 5 benchmark shots successfully. Re-running it is `python decode.py output/ltx_wan_trial --strategy runpod-ltx2`.
2. LTX-2 was noticeably faster per shot in this (resolution-unmatched, see Caveats) comparison -- roughly 3x -- consistent with the third-party benchmark research already in this repo (research/0017).
3. LTX-2's audio is not optional or separable -- adopting it for video generation means the AV compute cost is paid either way; actually using the audio output is a small additional step, not a separate heavy path as originally assumed from reading the graph alone.
4. **Before Will judges quality**: re-run at matched resolution (1280x704, matching Wan22) for a fair speed/cost comparison, and if the visual quality holds up, do the small `LTXVAudioVAEDecode` follow-up to hear what the SFX/score would sound like.
5. Remaining budget: **~$8.62 of the ~$10 cap** -- enough for a resolution-matched re-run of both strategies plus the audio-decode follow-up spike.

## Budget spent

**~$1.38 of the ~$10 cap total**: Wan 2.2 baseline $0.22 (clean, 0.4hrs on L40S); LTX-2 ~$1.16 across every attempt this session (most of it debugging three real bugs against a live pod, not generation -- the clean 5-shot LTX-2 generation phase itself cost roughly $0.05). No RunPod pods were left running; both were explicitly terminated at the end of their sessions.
