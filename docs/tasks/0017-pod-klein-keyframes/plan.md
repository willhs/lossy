# Plan 0017: self-hosted klein keyframes

## Current state

- `keyframes.py` owns routing, sheets (`make_sheets`), keyframes (`make_keyframes`), the
  `KeyframeGenerator` protocol and `FalKleinGenerator`. Caches are keyed on `generator.id`.
- `RunPodLtx25KeyframeStrategy.prepare()` (strategies_video.py) runs from `run_decode` before any pod
  exists; `generate()` boots the pod lazily via `_ensure_pod()` -> `with_setup_retry(_setup_pod_once)`.
- `RunPodSession` has `submit_workflow`, `download_output`, `scp_to`, `free_vram` (`/free` with
  `free_memory` only; it does not unload models).
- Research 0029 "Can klein and Qwen run on the LTX RunPod pod?" names the weights and nodes.
- ComfyUI v0.39.2 source (read 22 Nov): `EmptyFlux2LatentImage(width,height,batch_size)`,
  `Flux2Scheduler(steps,width,height)`, `ReferenceLatent(conditioning, latent?)`, `CLIPLoader` type
  `flux2`. The official template `image_flux2_klein_image_edit_4b_distilled.json` uses
  `flux-2-klein-4b-fp8.safetensors` (BFL repo), `qwen_3_4b.safetensors`, `flux2-vae.safetensors`,
  euler, 4 steps, CFG 1, one `VAEEncode` + `ReferenceLatent` per reference image.

## Data shape

- `KleinWeights`: three pinned files (diffusion fp8, text encoder, VAE).
- `klein_graph(prompt, seed, size, refs: list[str]) -> dict`: pure; `refs=[]` is text-to-image (sheet),
  1-2 refs is an edit (keyframe). Same graph shape, so one builder.
- `ComfyRunner` protocol (what the generator needs from a pod): `ensure_ready()`, `upload(png) -> name`,
  `run(graph) -> png bytes | None`. The strategy adapts `RunPodSession` to it; tests fake it.
- `PodKleinGenerator(runner, timing_log)`: `id = "pod-flux2-klein-4b-fp8"`; lazy `ensure_ready()` on
  the first call, so a fully cached film boots no pod in `prepare()`; `used_pod` tells the strategy to
  unload models afterwards. Each image appends `{kind, seed, seconds}` to `keyframes/timing.jsonl`.

## Design options

| Option | Shape | Verdict |
|---|---|---|
| A. Generator owns a `RunPodSession` | generator boots its own pod | No: two pods, two boots, double cost |
| B. Generator takes a runner over the strategy's session | one pod, prepare() boots it lazily | **Chosen** |
| C. Move keyframing into `generate()` per shot | klein and LTX swap per shot | No: model swap per shot, and the look-alike gate needs all sheets first |

## Phases

- [x] Phase 0: spec + plan committed
- [ ] Phase 1: `klein_graph` + `PodKleinGenerator` + tests (keyframes.py, tests/test_keyframes.py)
- [ ] Phase 2: strategy: `--keyframe-generator fal|pod`, klein weights pinned and downloaded on the pod
      when `pod` is chosen, `prepare()` boots the pod for a pod generator, unload models after; tests
- [ ] Phase 3: paid validation under `tools/capped_run.py` (pod) on SW4 684-710: sheets + keyframes on the
      pod, one LTX clip after the unload, timings; then fal on the same window with what budget is left;
      score keyframe cosine-to-sheet with the 0029 scorer's method; contact sheet
- [ ] Phase 4: research 0029 section, research 0030 cost table, AGENTS.md line; flip default to `pod` if pass
- [ ] Wrap: full `uv run python -m pytest`, commit

## Assumptions

1. Weights: BFL's `flux-2-klein-4b-fp8.safetensors` (rev pinned) + Comfy-Org `qwen_3_4b.safetensors`
   (bf16) + `flux2-vae.safetensors`, as the official ComfyUI template. fp4 encoder skipped: no native fp4 on
   Ampere and it is the quality risk.
2. Steps 4, euler, CFG 1 via `BasicGuider` (CFG 1 makes the negative unused), as the distilled template.
3. Unload is `/free {"unload_models": true, "free_memory": true}`; LTX reloads on its first graph.
4. A pod-side image failure is a `KeyframeError` (that shot falls back to text), as with fal. A pod that
   cannot be set up raises out of `prepare()` exactly as it would from `generate()`.
5. Validation does sheets + keyframes for the whole window but renders only ONE LTX clip (proves the
   unload-then-LTX path) to stay under budget; the clip-level K-vs-T result already exists (0029).
6. The fal comparison is redrawn on the same window only with the budget left after the pod run; the
   published SW4 fal keyframe-stage figures (Luke 0.674, Obi-Wan 0.643, Owen 0.768) are the fallback.
7. Budget: $0.50 covers RunPod + fal together.
