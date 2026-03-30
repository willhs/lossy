---
id: 0012-wan-model-variants
type: note
purpose: "Catalogue available Wan 2.1 and 2.2 model variants, their capabilities, VRAM requirements, and suitability for image-conditioned generation in the lossy pipeline."
scope: ["research", "decode", "video-generation", "runpod"]
tags: ["research", "wan", "i2v", "model-variants", "vram", "quantization"]
related: ["research/0003-prompt-to-video/research.md", "research/0011-character-continuity/research.md"]
---

## Question

What open-source Wan model variants are available, how do they differ architecturally, and which are viable candidates for image-conditioned generation on the existing RunPod infrastructure?

## Context

The pipeline currently runs `wan2.1_t2v_1.3B_fp16` via ComfyUI on a 24 GB GPU (RTX 3090/4090 class). This model is text-only. Exploring I2V variants is motivated by the character continuity problem (see research/0011).

---

## Wan 2.1 Variants

All Wan 2.1 models share the same 3D causal VAE (`wan_2.1_vae.safetensors`) and the same UMT5-XXL text encoder (`umt5_xxl_fp8_e4m3fn_scaled.safetensors`). I2V and FLF2V additionally require the CLIP vision encoder (`clip_vision_h.safetensors`).

### T2V-1.3B (current)

- Text-to-video only. 480P.
- Weights: 2.84 GB fp16. No fp8 variant.
- VRAM: ~8 GB.
- The only Wan variant that runs on low-VRAM consumer hardware without quantization.
- Fast inference — primary reason it is used in the pipeline.

### T2V-14B

- Text-to-video only. 480P and 720P.
- Weights: 28.6 GB fp16 / 14.3 GB fp8.
- VRAM: ~16 GB fp8, ~32 GB fp16.
- Same architecture as I2V-14B minus the image conditioning pathways.

### I2V-14B-480P and I2V-14B-720P

- Image-to-video. Separate checkpoint per resolution — not a runtime flag.
- Weights: 32.8 GB fp16 / 16.4 GB fp8.
- VRAM: ~20 GB fp8, ~32 GB+ fp16.
- The 480P and 720P checkpoints are separately trained; you cannot switch resolution by changing a parameter.
- Requires `clip_vision_h.safetensors` in addition to the shared encoders.

**How image conditioning works:**

The I2V-14B shares the same DiT backbone as T2V-14B (same depth, hidden dim, attention config). Image conditioning is added via two mechanisms:

1. **CLIP cross-attention** — the reference image is encoded by a frozen CLIP ViT-H/14 into 257 tokens, projected via an MLP, then prepended to the text context embeddings before cross-attention. I2V uses a separate cross-attention class (`WanI2VCrossAttention`) with dedicated key/value projections for image tokens (`k_img`, `v_img`), computed independently and summed with text-context attention.

2. **VAE first-frame conditioning** — the reference image is VAE-encoded as a single frame, zero-padded to match the clip length, and passed as a conditioning tensor alongside a binary mask (frame 0 = 1.0, remaining frames = 0.0). The model learns to generate motion forward from this known starting frame.

The I2V checkpoint is larger than T2V (~32.8 GB vs ~28.6 GB) because it carries the extra CLIP projection weights. The two models are not interchangeable — you cannot load an I2V checkpoint into a T2V pipeline.

### FLF2V-14B (First-Last-Frame-to-Video)

- Takes both the first and last frame as anchors and generates the interpolation between them. 720P only.
- Weights: 32.8 GB fp16 / 16.4 GB fp8.
- VRAM: ~20 GB fp8.
- Useful for shots where you know the end state (e.g. a character walks through a door).

### VACE-1.3B and VACE-14B

- Video-Aware Conditioning Engine. Unified model for reference-to-video (R2V), video-to-video editing (V2V), and masked video editing.
- Accepts up to 5 reference images, optional source video, masks, and start/end frames via a Video Condition Unit (VCU).
- VACE-1.3B: 4.31 GB fp16, ~8 GB VRAM — fits on the current pod alongside the existing text encoder stack.
- VACE-14B: 34.7 GB fp16, no fp8 variant in Comfy-Org repack, requires ~40 GB+ VRAM.
- VACE-1.3B is notable: same VRAM footprint as the current T2V-1.3B, but accepts reference images for identity anchoring.

### MAGRef-14B (third-party)

- Research model (ICLR 2026) built on the I2V-14B base.
- Designed for multi-subject identity-consistent video generation from reference images.
- Uses pixel-wise masked region-aware channel concatenation of reference image latents with noise latents.
- Not an official Wan-AI release. Available via Comfy-Org repackaged.

---

## Wan 2.2 Variants

Wan 2.2's 14B models use a Mixture-of-Experts (MoE) architecture split across two checkpoints: a `high_noise` expert (handles early denoising / structure) and a `low_noise` expert (handles refinement). Both must be loaded simultaneously, doubling the VRAM requirement vs. Wan 2.1 14B.

All Wan 2.2 14B models use the same `wan_2.1_vae.safetensors` as Wan 2.1. The exception is TI2V-5B, which uses a new `wan2.2_vae.safetensors`.

### T2V-A14B

- Text-to-video, MoE. Two checkpoint files (~28.6 GB fp16 / 14.3 GB fp8_scaled each).
- Recommended: 80 GB VRAM single GPU. Consumer GPU requires model offloading.

### I2V-A14B

- Image-to-video, MoE. Same two-file structure as T2V-A14B.
- Same VRAM profile — 80 GB recommended.

### TI2V-5B

- Unified text-to-video and image-to-video in a single 5B dense model. 720P at 24fps.
- Weights: ~10 GB fp16.
- VRAM: ~24 GB (fits on current pod). ComfyUI docs note ~8 GB with offloading.
- Uses the new `wan2.2_vae.safetensors` — requires downloading the additional VAE.
- The most practical Wan 2.2 upgrade for the current infrastructure: single file, handles both T2V and I2V, 720P.

### S2V-14B

- Audio/speech-driven video generation. MoE, single checkpoint.
- bf16: ~32.6 GB / fp8_scaled: ~16.4 GB.
- Not relevant to the character continuity use case.

### Animate-14B

- Character animation and replacement with "holistic movement and expression replication".
- bf16: ~34.5 GB, no fp8 variant in Comfy-Org repack.
- Potentially relevant for character work but requires high-VRAM hardware.

### Fun Variants (2.1 and 2.2)

Control/inpainting/camera-trajectory variants exist for both 2.1 and 2.2 in 1.3B and 14B sizes. Not directly relevant to character continuity.

---

## fp8 Quality Ordering

Per ComfyUI documentation: `fp16 > bf16 > fp8_scaled > fp8_e4m3fn`

For quality-sensitive use (faces, character fidelity), prefer `fp8_scaled` over `fp8_e4m3fn` if running quantized.

---

## VRAM Summary

| Model | fp8 Size | Approx VRAM (fp8) | Fits 24 GB | Notes |
|---|---|---|---|---|
| T2V-1.3B (current) | 2.84 GB | ~8 GB | Yes + audio | No fp8 variant |
| T2V-14B | 14.3 GB | ~16 GB | Yes | T2V only |
| I2V-14B-480P | 16.4 GB | ~20 GB | Tight (no audio) | Best I2V option for 24 GB |
| I2V-14B-720P | 16.4 GB | ~20 GB | Tight (no audio) | 720P output |
| FLF2V-14B | 16.4 GB | ~20 GB | Tight (no audio) | First+last frame |
| VACE-1.3B | 4.31 GB | ~8 GB | Yes + audio | Up to 5 ref images |
| VACE-14B | 34.7 GB | ~40 GB+ | No | fp16 only |
| TI2V-5B (2.2) | ~10 GB | ~24 GB | Borderline | Unified T2V+I2V, new VAE |
| I2V-A14B (2.2) | ~29 GB | ~60 GB+ | No | MoE, two files |

---

## Weight Sources

- **Official**: `Wan-AI` org on HuggingFace (multi-file sharded)
- **ComfyUI drop-in**: `Comfy-Org/Wan_2.1_ComfyUI_repackaged` and `Comfy-Org/Wan_2.2_ComfyUI_Repackaged` (single safetensor files per variant, fp16/bf16/fp8 variants where available)

---

## Candidates for the Lossy Pipeline

**VACE-1.3B** — lowest-friction option. Same VRAM as current, accepts reference images, no pod changes. Quality ceiling is lower than 14B but worth testing for character anchoring at no infrastructure cost.

**I2V-14B-480P fp8_scaled** — strongest I2V quality that can fit on 24 GB (without concurrent audio). Requires downloading `clip_vision_h.safetensors` and a new ComfyUI workflow. Concurrent audio would need to run sequentially rather than concurrently.

**TI2V-5B (Wan 2.2)** — unified T2V+I2V in one model, 720P, fits on 24 GB. Requires the new 2.2 VAE. The cleanest upgrade path if willing to change the VAE.

**I2V via fal.ai** (`fal-ai/wan-i2v`) — no pod changes, full fp16 quality, pay-per-clip. Keeps concurrent audio on RunPod separate and unaffected.
