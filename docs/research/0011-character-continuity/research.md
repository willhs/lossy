---
id: 0011-character-continuity
type: note
purpose: "Investigate character continuity across independently generated clips and assess options to reduce character drift without sacrificing high compression."
scope: ["research", "decode", "prompts", "video-generation"]
tags: ["research", "character-consistency", "video-generation", "wan", "seedance", "lora"]
related: ["research/0002-shot-to-prompt/research.md", "research/0003-prompt-to-video/research.md", "research/0005-compression-analysis/research.md"]
---

## Question

Characters in the lossy pipeline look like a different actor or variant in almost every clip. How can we mitigate this without sacrificing the high compression / text-only approach that makes the pipeline cheap and fast?

## Root Cause Analysis

The pipeline generates each clip in total isolation. There are five compounding reasons why characters drift:

1. **Independent Gemini descriptions.** Each shot's `subjects` field is written from that shot's keyframes alone. Gemini may describe "a young man with dark hair" in one shot, "a tall figure in a jacket" in the next. No cross-shot context, no character name registry.

2. **Text-only generation.** All active strategies (Wan T2V, Seedance T2V, RunPod Wan) are text-to-video. The model interprets subject descriptions freshly per clip with no visual anchor.

3. **Independent seeds.** Each shot is seeded with its index (`seed=idx`). Seeds across shots have no relationship, so the model is free to sample different regions of character-space each time.

4. **No locked character descriptions.** The `format_prompt_wan()` function places subjects first (good for T5 attention) but the description itself varies — whatever Gemini wrote. There is no canonical "Luke Skywalker = 20-year-old white male, sandy blond hair, white tunic, brown leather belt" that is enforced across shots.

5. **No post-processing reconciliation.** The stitch step concatenates clips; it applies no temporal or identity correction.

## What Is Already Working

The encoding pipeline captures rich per-shot data — subjects, action, lighting, setting, camera motion, audio, and dialogue. This is a good foundation. The problem is purely in the decode/generation phase.

---

## Five Options

### Option 1 — Locked Character Descriptors in Prompts

**Mechanism:** Build a character registry at encode time and inject a canonical locked descriptor for each recognized character into every clip prompt, regardless of what Gemini described that shot.

**How it works in the pipeline:**
- After stage 2 encoding, run a post-processing step that reads all `subjects` fields across `prompts.json` and clusters them into named characters (either via another Gemini call or a simple majority-vote on name mentions).
- For each character, synthesize a single canonical appearance string: age range, hair, eye color, skin tone, clothing down to fabric/accessories.
- In `prompt_format.py`, replace or augment the per-shot `subjects` field with the canonical descriptor for any character present in that shot.
- Add a negative token to each prompt: `different clothing, different hair color, inconsistent appearance`.

**Effectiveness:** Low-to-medium. Reduces gross drift (wrong gender, wrong hair color, wrong clothing) but does not prevent subtle facial geometry variation — the model samples different facial structures even given identical text. Better as a foundation for other options than as a standalone fix.

**Compression compatibility:** Full. The canonical descriptions are small and integrate naturally into existing prompt formatting. No structural pipeline change needed.

**Implementation complexity:** Low. A post-encode script (~100 lines) that reads `prompts.json`, clusters characters, and writes a `characters.json` sidecar. `prompt_format.py` picks up the sidecar and overrides `subjects` per shot.

**Cost:** None.

---

### Option 2 — Image-to-Video (I2V) with Canonical First Frame

**Mechanism:** Switch from text-to-video to image-to-video generation. Pass a canonical reference image of each character as the first frame. The model generates forward from that visual anchor, maintaining appearance for the duration of the clip.

**Supported today:**
- `fal-ai/wan-i2v` — Wan 2.1 I2V endpoint, accepts `image_url` + `prompt`.
- `fal-ai/wan-flf2v` — Wan first-and-last-frame-to-video, pins both endpoints.
- `fal-ai/bytedance/seedance/v1.5/pro/image-to-video` — Seedance 1.5 Pro I2V.

**How it works in the pipeline:**
- Extract canonical character frames from the source film (one per major character) — e.g. a clear frontal shot from the original. Store them alongside `prompts.json` as `characters/{name}.jpg`.
- Add an I2V strategy variant (e.g. `FalWanI2VStrategy`) that wraps `fal-ai/wan-i2v`.
- Per shot, look up which character is present (from the `characters.json` registry built in Option 1) and pass that character's canonical frame as `image_url`.
- Shots with no named character fall back to T2V.

**Effectiveness:** Medium-high for intra-clip consistency (the clip starts from the reference). Cross-clip consistency is determined by how representative the canonical frame is — if every shot for Luke starts from the same Luke frame, his appearance anchors around it. Limitations: the conditioning is strongest on the first frame; drift can accumulate later in the clip; extreme angle changes resist the anchor.

**Compression compatibility:** Requires reference images per character as pipeline metadata alongside prompts. This is a meaningful extension (the pipeline currently carries only text) but not a rewrite.

**Implementation complexity:** Medium. New strategy class, character registry, frame extraction step. The API call change itself is trivial.

**Cost:** Same inference cost as T2V.

---

### Option 3 — Character LoRA Fine-Tuning

**Mechanism:** Train a small LoRA adapter on images of each character extracted from the source film. Load the LoRA at inference time for every clip featuring that character. The LoRA biases the model toward that character's specific appearance.

**Supported today:**
- `fal-ai/wan-i2v-lora` — fal.ai endpoint that accepts a `loras` parameter (URL to weights + strength scalar).
- RunPod self-hosted Wan supports LoRA loading natively in ComfyUI.

**How it works in the pipeline:**
- Extract 20-50 diverse frames per major character from the source film (varying angles, lighting, expressions).
- Train a Wan 2.1/2.2 LoRA for each character — 10-20 epochs on a single RunPod A100. One-time cost per film (~$5-20).
- Host the resulting `.safetensors` weights (e.g. in a public S3 bucket or RunPod volume).
- In `strategies_video.py`, add a `loras` parameter to the RunPod/fal strategy. Look up the relevant character LoRA URL per shot from `characters.json` and inject it.

**Effectiveness:** High. The strongest text-only-compatible approach for consistent character appearance across many clips. Multiple clips using the same LoRA share the same character "prior" regardless of how precisely the text describes them. Used by production pipelines precisely for this reason.

**Compression compatibility:** Excellent. Once trained, you pass a LoRA URL per generation call. The LoRA encodes character appearance implicitly — text prompts don't need to carry more information.

**Implementation complexity:** High upfront (dataset curation, training pipeline, weight hosting), low per-clip thereafter. Training infrastructure is reusable across films.

**Cost:** One-time training cost per character (~$5-20 on RunPod A100). Negligible inference overhead (~2-5% extra latency for LoRA weight injection).

---

### Option 4 — Post-Processing Face Swap (ReActor)

**Mechanism:** Generate all clips ignoring character consistency, then run a face-replacement pass over every clip using a canonical reference face per character. This decouples generation from identity consistency.

**Tools:**
- **ReActor** (ComfyUI node) — current standard for face swapping in video. Uses `inswapper_128` for swapping and `retinaface_resnet50` for detection. Supports batch video processing.
- **CodeFormer / GFPGAN** — face restoration applied after swap to sharpen and improve fidelity. GFPGAN has better temporal stability within a clip.
- Both are available as ComfyUI nodes on the RunPod instance already used by the pipeline.

**How it works in the pipeline:**
- Add a post-decode step (e.g. `python decode.py output/film --strategy runpod-wan --face-swap`) that iterates all generated clips.
- For each clip, identifies shots with known characters (from `characters.json`), and submits a ReActor ComfyUI workflow passing the clip and the canonical face image.
- The output replaces the original clip file; stitch proceeds normally.

**Effectiveness:** Medium. High for facial identity on frontal/near-frontal shots; degrades at extreme angles, with motion blur, or where the character face is small in frame. Does not address body/clothing consistency. Can introduce artifacts at hair/neck boundaries. The face swap is a blunt instrument — it works but doesn't look naturally generated.

**Compression compatibility:** Fully compatible. This is a post-generation step that requires only reference face images. No changes to encoding or prompts.

**Implementation complexity:** Medium. Requires ComfyUI ReActor workflow setup on RunPod (already self-hosted) and a new post-decode script. The operational complexity is running it frame-by-frame across potentially hundreds of clips.

**Cost:** Light — GPU inference for face swap is much cheaper than video generation. ~10-30 minutes of A100 time per film, ~$0.50-2.00.

---

### Option 5 — Model-Native Reference Conditioning (Seedance 2.0 / Wan VACE)

**Mechanism:** Use models with built-in reference conditioning — dedicated pathways that accept character reference images and propagate identity information through every denoising step, not just as a first frame.

**Seedance 2.0 (ByteDance / fal.ai — forthcoming):**
- Accepts up to 12 multimodal reference assets per generation call (images, video clips, audio).
- `@mention` system in prompts: `"Reference @Image1 for the character's appearance"` — role assignment is explicit in the prompt.
- Reported ~80% reduction in facial feature drift vs. text-only (Seedance 1.x baseline).
- Clothing color consistency improved significantly.
- The architecture fuses character reference embeddings throughout the denoising process (not just as a first frame), making it more robust to angle changes and motion.
- API on fal.ai is listed but not yet available at the time of writing; Seedance 1.5 Pro is the current endpoint. Expected: mid-2026.

**Wan 2.1/2.2 VACE module:**
- VACE (Video-Aware Conditioning Engine) is a native conditioning module in the 14B Wan model.
- Fuses a reference image latent directly into the model's image embeddings at inference — analogous to IP-Adapter but integrated into the model architecture.
- Requires ~32GB VRAM. Feasible on RunPod A100/H100, not on the current 24GB consumer GPU setup.
- Available as a ComfyUI workflow today but requires a VACE-capable RunPod pod template.

**Effectiveness:** High for both. Seedance 2.0's `@mention` conditioning is the most ergonomic high-effectiveness option once the API lands. Wan VACE is powerful for self-hosted setups with sufficient VRAM.

**Compression compatibility:** Same as Option 2 — requires reference images per character as metadata. The key difference is that the conditioning operates throughout generation rather than just at the first frame, making it more robust.

**Implementation complexity:** Low for Seedance 2.0 once the API is available (additional parameters on the existing fal.ai call). Medium for Wan VACE (pod template change + ComfyUI workflow update).

**Cost:** Seedance 2.0 generation costs will likely be higher than 1.x; Wan VACE on A100 is ~$2-4/hr on RunPod (vs. ~$0.76/hr for the current 24GB RTX setup).

---

## Comparison

| Option | Effectiveness | Complexity | Cost | Pipeline Change | Available Today |
|---|---|---|---|---|---|
| 1. Locked descriptors | Low | Low | None | Prompts only | Yes |
| 2. I2V first-frame | Medium-high | Medium | None | New strategy + char registry | Yes |
| 3. Character LoRA | High | High (upfront) | ~$5-20/film training | New strategy + LoRA hosting | Yes |
| 4. Face swap (ReActor) | Medium | Medium | ~$0.50-2/film | Post-decode step | Yes |
| 5. Model-native ref | High | Low (when available) | Higher per-generation | API params only | Soon |

## Recommendations

**Best near-term move (low cost, meaningful improvement):** Combine Option 1 + Option 2. Build a character registry at encode time and switch to I2V with canonical frames extracted from the source film. No training cost, no infrastructure change, works with today's APIs. Option 1 alone is worth doing regardless — it makes the text prompts more stable and stacks with everything else.

**Highest ceiling (worth the investment for film-scale work):** Option 3 (Character LoRA). The fal.ai `wan-i2v-lora` endpoint already supports it. Training a LoRA per major character is a one-time cost per film and gives the strongest consistency without changing the per-clip generation cost. The training pipeline is reusable.

**Watch for:** Seedance 2.0's multimodal reference API on fal.ai (Option 5). When it lands, it will make options 2, 3, and parts of 4 redundant — just pass character reference images with each generation call.

**Avoid for now:** Option 6 (multi-shot attention sharing / Video Storyboarding). Effective in research but requires intervention at the diffusion denoising loop level — not compatible with API-based generation and would require full self-hosting of Wan.
