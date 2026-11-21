---
id: 0029-character-reference-continuity
type: note
purpose: "Survey how to give the decoder one reference look per character (sheets made once, fed to every shot) so a recurring character stops looking like a new person each shot; compare reference-to-video, keyframe-first and face-swap routes on identity evidence, multi-character support, cost per film, self-hosting and code change; recommend a small trial. No paid generation."
scope: ["research", "decode", "characters", "ltx-2.5", "video-generation", "continuity"]
tags: ["research", "character-consistency", "reference-images", "ic-lora", "keyframe", "ltx-2.5", "seinfeld"]
related:
  - "research/0011-character-continuity/research.md"
  - "research/0020-wan22-identity-i2v-verification/research.md"
  - "research/0021-training-data-contamination/research.md"
  - "research/0025-video-model-refresh-nov-2026/research.md"
  - "research/0026-seinfeld-s01e03-first-end-to-end-run/research.md"
  - "research/0028-style-drift/research.md"
  - "research/experiments/0004-vace-continuity-test.md"
  - "research/experiments/0005-vace-3condition-continuity-test.md"
---

# Character continuity via reference images

Vision alignment: Have an online presence (goal: complete a movie or TV show reconstruction,
Star Wars IV headline run with a control title)

Steady task `05bd15af`. Will, 15 Nov: before the full Star Wars IV run (job `08db840b`,
scheduled 30 Nov), make each character keep one look across shots. No paid generation in this
job; the trial below needs his go-ahead.

Confidence tags: **[Direct]** read in code or measured here; **[Supported]** a dated external
source says so (I did not reproduce it); **[Inferred]** my reasoning from those;
**[Speculative]** a guess, flagged as one. Prices were read on 2026-11-15 unless a date is
given, and many come from fal.ai pages or resellers, so recheck before spending.

## Answer

- **Bake-off (20-21 Nov): K stays.** Qwen first frames (KQ) match K on identity but fail the
  adherence and motion rule; hosted Wan VACE-14B (W) gives no identity gain at about 20 times the
  cost; Ingredients (I) could not run (no HF token reachable); Kling (R) did not fit the $5 cap.
  Spent <= $3.69. Klein and Qwen can very likely run on the LTX pod, unproven. See "Bake-off".
- **Trial result (16 Nov): keyframe-first (K) passes.** On 16 Seinfeld shots, blind "same
  person" went from 3.5 to 5.0 and face-embedding cosine from 0.23 to 0.55. Adherence and
  motion fell 0.19, inside the rule's 0.3, and photoreal held at 100%. Total spend was ≤$1.04.
  Adopt K for SW4 via a follow-up strategy flag with two fixes: sheet wardrobe leak and
  look-alike sheets. See "Trial results".
- **Arm I (Ingredients) is wired and dry-run checked, not run.** The gate and token are done.
  Two jobs tried (21 Nov): the first lost its pods to ssh, the second priced T + I at about
  $0.85-1.20 against a $0.85 cap and skipped. It needs about $1.3 and a watched first boot: see
  "Arm I rerun".
- **Most of the drift is not solvable by text.** Today a shot's first frame comes from text
  alone. The identity block is byte-identical across shots (0020), but the same 260 characters
  of description can be drawn as many different faces. [Direct: `strategies_video.py`
  `RunPodWanStrategy.generate`, part 0 has `start_image=None`]
- **Recommended route, first: keyframe-first on the current LTX-2.5 path.**
  1. Make one reference sheet per human character, once per decode.
  2. Compose each character shot's first frame with an open image editor that takes those
     references: FLUX.2 [klein] 4B, Apache-2.0, or Qwen-Image-Edit-2511, Apache-2.0.
  3. Feed that frame into the I2V hook LTX-2.5 already has (`LTXVImgToVideo`, used today only
     for chaining split parts).

  Projected cost: about **+$1-11 a film** over today's ~$15. It is the smallest code change of
  the options. It should also pin the photoreal style, the 0028 problem, as a side effect.
  [Inferred]
- **Recommended second arm: LTX-2.5's own reference-sheet adapter.** It exists:
  `Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients`, file `ltx-2.5-22b-ic-lora-ingredients-0.9`
  (1.31 GB), created 2026-09-10. It is gated by an access click and covered by the LTX-2.x
  community licence. It conditions every frame on one composite sheet that can hold several
  characters, which suits 2-3 person shots. Its cost multiplier and identity quality are
  unpublished. [Direct: HF API, `evidence/ltx25_ingredients_hf_api.txt`]
- **Hosted reference-to-video APIs are 10-50x over budget for the film.** Kling O3, Vidu Q2,
  Veo 3.1 and Seedance 2.0 come to about $520-3,200 for Star Wars IV and $100-500 for the
  Seinfeld episode, against a $50-60 goal ceiling. They are only for hero shots, or as a
  measuring stick in the trial. [Supported + arithmetic]
- **Open Wan-based R2V has the best published identity scores**: Phantom-14B, VACE-14B,
  BindWeave, MAGREF and Lynx. Two things count against it: it would mean leaving LTX-2.5, which
  won the 0025 bake-off and carries synced audio, and it costs roughly $80-330 a film.
  [Supported + Inferred]
- **Face swap after generation is a fallback only.** The best-known swapper,
  `inswapper_128`, is licensed for non-commercial research only. All the common swappers are
  weak on small and profile faces, which most film shots have. [Supported]
- **Do not use real-film stills as references.** A still ships the source's pixels, so the
  payload stops being text. It injects the real actor's face, which is the opposite of 0021's
  stance. And the output becomes a likeness of a real person. This overrides 0011's suggestion
  to cut canonical frames from the source. [Inferred, argued below]
- **Payload: 0 bytes if sheets are regenerated at decode, about 14-48 KB per character if
  shipped.** Star Wars IV's ~10 human characters come to about 0.5 MB as shipped AVIF sheets,
  more than the whole brotli-compressed text (366 KB, 0005). [Direct: `size_probe.py`]
- **Trial: Seinfeld S01E03, 16 shots, three arms**: text only, keyframe-first, Ingredients. It
  is scored by a blind identity-consistency rubric plus a face-embedding metric. About
  **$0.90 with a $1.25 cap**, and it needs one human step: accepting the gated adapter and
  providing an HF token.

## Premise check

The task says "today every shot renders a new-looking person". The code agrees:

- **No visual anchor on a shot's first part.** `RunPodWanStrategy.generate` starts every
  shot with `start_image = None`, and LTX-2.5 inherits that. The only image input is chaining a
  split shot's later parts from the previous part's last frame. [Direct: `strategies_video.py`
  `generate`, `_upload_start_frame`] 0028 reached the same conclusion for style. [Direct: 0028
  "Premise check"]
- **Text identity is already as locked as text gets.** `CharacterIdentityMixin` prepends the
  same verbatim description for up to 2 characters a shot, strips personal names, and keeps
  design names (`keep_name`). [Direct: `strategies_video.py` `_prepend_identity`,
  `_cap_to_most_prominent`, `_all_character_names`] So "one look" cannot come from more prompt
  work: identical tokens still sample different faces.
- **A reference path already exists, on a retired model.** `decode.py:generate_portraits` makes
  one fal Flux Schnell portrait per character ("Professional portrait photograph of
  {description}…"). Only `RunPodVaceStrategy` (VACE-1.3B) uses it, passing the first
  character's portrait as `reference_image`. [Direct: `decode.py:72-126`, `:848-872`;
  `strategies_video.py` `RunPodVaceStrategy.generate`] Experiment 0005 found it added only
  marginal value over enriched text, helped in close-ups but not wide shots, and sometimes
  skewed to anime. [Direct: exp 0005 "Interpretation"] That was a 1.3B model fed a single
  portrait, so it says little about current 14B+ R2V models or LTX-2.5's adapter. [Inferred]
- **Nothing in this note was already done on main.** I grepped for `ingredients`,
  `IC-LoRA`, `sheet` and `keyframe` in `strategies_video.py`/`decode.py`/`encode.py` and found
  only the chaining and VACE paths above. [Direct]

## 1. Making the reference images

### Generate, do not cut from the film

| Source | Compression story | Contamination (0021) | Likeness / publication | Verdict |
|---|---|---|---|---|
| **Generated sheet from the encode's name-free description** | Text in, text out. The sheet is a decode-side derivative of the description. | Identity still comes from the description, and no name enters the sheet prompt. | An invented face. Nothing claims to be the actor. | **Use** |
| Real-film still (face crop per character) | Ships the source's pixels, becoming an image codec of the actor's face. Breaks the "film in <1 MB of text" claim. | Worse than naming: injects the real look directly, not via memory. | Renders a real person's face onto new footage, a deepfake of the actor, for a public YouTube/Reddit release. | **Don't** |
| Sheet prompted with the character's name | Text | Exactly 0021's failure: the model draws its memory of the actor or design. | Same concern as stills, indirectly | **Don't** |

[Inferred from 0021's findings and 0005's codec framing.] There is a quality reason as well:
0005 found single keyframe-crop references gave inconsistent, crop-dependent results.
[Direct: exp 0005]

**No sheets for design characters.** These are the `keep_name` entries: the Star Wars droids,
Vader, Stormtroopers, Chewbacca, Jawas. Their name is their description (0023/0028), so any
sheet would be the image model's memory of a famous, copyrighted design: contamination
rendered as a picture. They stay text-only. 0025 already flagged Kandinsky's droids as
"recall, not reconstruction". [Inferred] The sheet path covers human characters only, which is
also where drift is visible.

### What a sheet is

- **Layout.** One character per sheet: front, 3/4 and profile head-and-shoulders, plus one full
  body, in a 2x2 grid on a plain background. It uses neutral expression and lighting and the
  description's stable traits only. Wardrobe stays out of the sheet, as 0023 decided for text:
  costume belongs to the shot. [Inferred]
- **Ingredients adapter.** It wants one composite image per generation ("characters, props, and
  locations laid out on a black background"), resized to a 544 px short side. A per-shot
  composite can be pasted locally from the per-character sheets of the characters in that shot,
  with no generation. [Supported: docs.ltx.io IC-LoRA adapters page; floyo.ai workflow page]
- **Lightricks' guidance on panels.** Small or cluttered panels cause identity drift, so keep
  to 1-3 characters per composite. [Supported: LTX-2.3 Ingredients card, via web research]

### Candidate sheet generators

| Model | Max refs | Price | Weights / licence | Note |
|---|---|---|---|---|
| **FLUX.2 [klein] 4B** | 4 (secondary source) | Self-host: seconds per image | **Apache-2.0**, ungated (HF API, created 2026-01-14) | Small (~8-13 GB VRAM), can share the LTX pod. **First choice.** |
| **Qwen-Image-Edit-2511** | Multi-image, two-person fusion | fal $0.03/MP; self-host | **Apache-2.0**, ungated (HF API, created 2025-12-17) | 20B, stronger edits, heavier |
| Nano Banana Pro (Gemini 3 Pro Image) | 14, "5 people at high fidelity" | fal $0.15 | Closed | Best API option if klein's faces are weak |
| Seedream 4.5 / 5.0 | 10 | $0.04 / $0.035-0.135 | Closed | |
| FLUX.2 dev / Kontext dev | up to 10 | $0.012/MP (fal) | Non-commercial | Licence rules it out for publication |

[Supported: fal.ai model pages, bfl.ai, HF; licence and creation dates [Direct] from the HF
API for klein and Qwen.]

**Sheet cost is negligible either way.** About 10 human characters × 1-4 images is ~$0.40-6 on
an API, or a few GPU-seconds self-hosted. [Inferred]

### Payload size per film

Measured with `size_probe.py`: crops of real LTX-2.5 render frames, resized and encoded locally.
Output is in `evidence/size_probe_out.txt` in the workspace, median of 8 (or 3):

| Image | WebP q80 | WebP q60 | AVIF q60 | AVIF q40 |
|---|---|---|---|---|
| single ref 512x512 | 31.8 KB | 22.4 KB | 27.2 KB | 13.8 KB |
| single ref 768x768 | 53.0 KB | 37.2 KB | 42.7 KB | 21.5 KB |
| 4-view sheet 1024x1024 | 123.6 KB | 84.9 KB | 99.9 KB | **48.4 KB** |

[Direct]

| Film | Human characters with a sheet | Shipped, AVIF q40 4-view | Shipped, WebP q80 | Text payload today (0005) |
|---|---|---|---|---|
| Star Wars IV | ~10 (16 in registry, less ~6 `keep_name` designs; ADR 006) | **~0.48 MB** | ~1.2 MB | 3.9 MB raw / **366 KB brotli** |
| Seinfeld S01E03 | ~4-6 | ~0.2-0.3 MB | ~0.5-0.75 MB | not measured (smaller) |

[Direct sizes; character counts Supported by ADR 006 and 0028]

**Images do not compress further.** Shipped sheets would more than double Star Wars IV's
compressed payload (366 KB → ~850 KB). That is still under 1 MB, so the headline claim
survives. [Inferred]

**The better framing: sheets need not ship at all.** A sheet only has to be the same for every
shot within one decode. It does not have to be bit-identical across decodes. So it can be
regenerated at decode time from the description (already in the payload), a fixed seed and a
pinned open model, for **0 added bytes**. A re-decode may draw a slightly different but equally
consistent cast. Ship a sheet (about 48 KB per character) only when a human has curated a look
and wants it locked. [Inferred]

## 2. Feeding the references into video generation

The film numbers below are Star Wars IV: 2,012 shots, 124.7 min = 7,482 s. Seinfeld S01E03 is
315 shots, 1,382 s. Today's baseline is LTX-2.5 two-stage at ~$14.90 a film, ~$2.67 an
episode, on a community A6000 [Direct: 0025]. "Change" means the change to
`strategies_video.py`.

### 2a. Hosted reference-to-video APIs

| Model | Refs / multi-person | Start frame + refs? | Price | Star Wars IV | Seinfeld | Identity evidence |
|---|---|---|---|---|---|---|
| Kling O3 Std R2V (fal) | 7 inputs; multi | **Yes** | $0.084/s | ~$630 | ~$116+ (3 s minimum clip) | No published face metric |
| Kling O1 R2V (fal) | 7; multi | Yes | $0.112/s | ~$840 | ~$155+ | none |
| Vidu Q2 R2V (fal) | 7; multi | No | $0.30/clip at 720p | ~$630 | ~$100 | Q2 0.596 ref-consistency in SkyReels-V3's paper |
| Veo 3.1 R2V (fal) | 3; weak multi ("a single person…") | **No** (mutually exclusive) | $0.20/s, fixed 8 s, 16:9 | ≥$3,200 | ≥$500 | none |
| Seedance 2.0 R2V (fal) | 9 images; multi via @Image tags | No dedicated param | ~$0.30/s (third-party) | ~$2,250 | ~$415 | none |
| Gemini Omni 1.2 Flash | 2 characters × 3 images | Not documented | ~$0.07/s (press; unverified) | ~$520 | ~$97 | none |
| MiniMax S2V-01 | 1 face | No | $0.50/clip | ~$1,000 | ~$158 | Hailuo tops OpenS2V human domain (FaceSim 57.7) |
| Runway Gen-4 refs | 3 refs, still images only | Image-then-I2V | ~$0.13/s (aggregator) | ~$990 | ~$180 | none |

[Supported: fal.ai `llms.txt` pages for Kling/Vidu/Veo/Seedance/MiniMax; Vertex docs for Veo;
press and segmind for Omni; OpenS2V-Eval leaderboard; arXiv 2601.17323. Film totals are
arithmetic.]

- **Everything is API-only.** Nothing in this table is self-hostable.
- **Two shutdowns.** The Sora 2 API was shut down on 2026-09-24, and the Veo 3.1 Preview models
  in the Gemini Developer API on 2026-10-22. [Supported, single secondary sources; not
  re-verified]
- **Change.** A new fal strategy, roughly `FalSeedanceStrategy`'s size: upload refs, map
  characters to @tags. Small. Licence: commercial API terms. Kling O3's start-frame support
  means it can also keep lossy's I2V chaining. [Inferred]
- **Verdict.** Over budget for any whole run. As a 16-shot trial measuring stick it is about
  $4 (Kling O3). [Inferred]

### 2b. Open reference-to-video models (self-host)

| Model | Multi-person | Licence | Identity evidence | Start frame + refs |
|---|---|---|---|---|
| Phantom-Wan 14B (2025-05) | ≤4 refs, built for multi-subject | Apache-2.0 (from a mirror README) | OpenS2V human: Total 64.2, FaceSim 55.0; Lynx paper: 0.664 | No |
| Wan2.1 VACE-14B (2025-05) | multi | Apache-2.0 | OpenS2V open: FaceSim 55.1; Lynx paper: 0.594 | **Yes** (masked frames + refs) |
| BindWeave (ByteDance, 2025-10) | multi | not verified | OpenS2V open: Total 57.6, FaceSim 53.7 | No |
| MAGREF (2025-05) | any count, per-subject masks | MIT (HF card) | OpenS2V human FaceSim 32.9 | not confirmed |
| Lynx (2025-09) | single face | Apache-2.0 | own paper: 0.779 (best) | No |
| Stand-In (2025-08) | single face | not verified | user study 4.10/5 (own paper) | composes with VACE (claimed) |
| SkyReels-V3 (2026-01) | 1-4 refs | not verified | own paper: 0.670 | No |
| HunyuanCustom | released weights single-subject | Tencent community licence | FaceSim 0.627 (own card) | No |

[Supported: OpenS2V-Eval 1.1 leaderboard (arXiv 2505.20292); papers 2509.15496, 2508.07901,
2601.17323; HF cards. The researcher saw duplicated rows on the live leaderboard, so treat
OpenS2V numbers as ±1-2.]

- **Cost.** All of these sit on Wan2.1-14B. At 720p × 81 frames they take an estimated 15-30
  min a clip at 50 steps on an H100, or 2-5 min with lightx2v/CausVid-style distillation, and
  480p is ~3x faster. [Supported, unbenchmarked; no official timings found]
  - Star Wars IV needs ~2,100 clips × ~3.5 min ≈ 120 H100-hours, about **$250-330**; at 480p
    about **$80-110**. Seinfeld is about $15-50. [Inferred]
- **Change.** Large. It is a new model family with no synced audio (the MMAudio path would have
  to return) and is 2-6x slower than LTX-2.5. Resolution falls below the 0025 winner's 1280x704
  unless we pay 720p prices. VACE-14B is the only one that also keeps chaining.
- **Verdict.** It has the best evidence but is the wrong cost class for the film. A candidate
  only if both LTX-2.5 arms fail the trial. [Inferred]

### 2c. Keyframe-first on LTX-2.5 (recommended)

**Mechanism.**

1. Make the per-character sheets (§1).
2. For each shot with ≥1 sheeted character, ask the image editor for that shot's first frame.
   Inputs: the sheets of the characters present (≤2, the mixin's existing cap), plus the
   style prefix and the shot description that `format_prompt` already builds.
3. LTX-2.5 runs I2V from that frame through the `LTXVImgToVideo` /
   `LTXVImgToVideoInplace` nodes it already uses for chaining.

[Direct: hook exists in `RunPodLtx25Strategy._build_workflow` (`start_image` branch)]

**Identity evidence.**

- **Measured drift.** On plain Wan 2.2 A14B I2V, face similarity between the input image and
  the video is 0.578 (ArcFace-based). Reward tuning raises it to 0.694. The paper notes that
  the problem "becomes critical when the human face occupies only a small fraction of the
  image". [Supported: arXiv 2510.14255, CVPR 2026]
- **No duration curve.** No source measured drift over 5-10 s, and I found no LTX-2.5 figure.
  [Supported: absence, searched]
- **The project's own evidence.** I2V copies a frame's look reliably: chained parts never
  drifted in style (0 of 69, 0028), and no seam appeared at a part boundary (0020). [Direct]
- **The catch.** The identity is only as good as the composed keyframe. Image editors hold
  identity much better than video models do (that is what multi-reference editing is for), but
  this is an assumption to test, not a measured figure. [Inferred]

**Multi-person shots.**

- **Editors.** klein takes 4 refs, enough for 2 characters × 2 views. Qwen-Image-Edit-2511
  advertises two-person fusion. Nano Banana Pro claims 5 people. [Supported]
- **Video stage.** The I2V stage doesn't care how many people are in the frame.

**Cost.** [Inferred; klein and Qwen speeds on an A6000 are estimates]

- **Video:** the same LTX-2.5 cost as today. `LTXVImgToVideo` replaces the empty latent, so no
  extra steps.
- **Keyframes, FLUX.2 klein 4B:** ~5 s each on an A6000, about 2.8 GPU-h for 2,012 shots ≈
  **+$1** (community) per film.
- **Keyframes, Qwen-Image-Edit-2511:** ~30-60 s each, ≈ **+$6-11** per film.
- **Episode:** pennies either way.
- **Pod memory:** LTX-2.5's int8 weights (~40 GB) on a 48 GB A6000 leave little room. Run the
  keyframes as **a separate pass before the video pass** on the same pod rather than swapping
  models per shot. [Inferred]

**Self-host and licence.**

- Self-hosted on the existing RunPod/ComfyUI setup; FLUX.2 klein has native ComfyUI support.
  [Supported]
- klein 4B and Qwen-Image-Edit-2511 are Apache-2.0 and ungated [Direct: HF API]. LTX-2.5 is
  unchanged.

**Change: small to medium.** [Inferred]

1. A sheet pre-pass. This replaces fal-based `generate_portraits` with an on-pod generator and
   writes `characters/<name>_sheet.webp`.
2. A keyframe pre-pass for character shots.
3. A hook so part 0 of a shot can take a start image. Today `generate` hardcodes `None`, and
   parts > 0 keep chaining as now.
4. `PINNED_MODELS` entries for the editor.

That is about 150-250 lines in a `RunPodLtx25KeyframeStrategy(RunPodLtx25Strategy)`. No new
ComfyUI custom nodes if klein's core-node support holds.

**Risks.** [Inferred]

1. The I2V start frame fixes composition. If the editor's composition fights the camera motion
   in the description, the shot inherits it. The two-stage sampler refines at stage 2 only, so
   a bad keyframe isn't repaired.
2. Wide shots with tiny faces gain little (2510.14255; exp 0005 saw the same).
3. The editor may add its own house style. That cuts both ways: it is the 0028 fix if
   photoreal, a new drift source if not. Score style in the trial.

### 2d. LTX-2.5 Ingredients IC-LoRA (recommended second arm)

**What exists** [Direct: HF API, `evidence/ltx25_ingredients_hf_api.txt`]:

- `Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients`, tagged `reference-sheet`,
  `character-consistency`.
- Single file `ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors` (1.31 GB), plus 6 example
  videos.
- Created 2026-09-10, modified 2026-10-22. `gated: auto`, so an access click is needed.
- Licence `ltx-2.x-community-license`, the same family as the LTX-2.5 base already in use.
- The model card itself is gated, so I could not read it.

**How it works** [Supported: docs.ltx.io IC-LoRA adapters page; floyo.ai workflow, undated]:

- **Input.** One composite sheet, fed as a static video guide repeated on every frame and
  cropped out after.
- **Prompt (corrected by the arm I job).** The docs page says `Reference sheet: <panels> /
  Generated video: <action>`, but the model card's six example prompts, which the HF API serves
  as widget data even though the card is gated, all use another form: `### Reference Sheet
  Description **Top Row Left (Character):** <what that panel shows> **Top Row Right (Prop):**
  <...> ### Target Description <the clip>`. Panels are named by their position on the sheet
  (`Top Row`, `Middle Row Left`, ...) and typed Character, Prop, Setting or Logo. The target
  repeats the characters' looks, then adds action, camera and speech. Arm I uses the card's form.
  [Direct: `evidence/ltx25_ingredients_hf_api.json`, widget examples]
- **Settings.** LoRA strength 1, CFG 1, 8 steps on the **distilled** model, the one lossy runs.
- **Nodes (corrected).** `LTXAddVideoICLoRAGuide` is a custom node from Lightricks'
  ComfyUI-LTXVideo, not core. The pinned core ComfyUI v0.39.2 has the same mechanism:
  `LoraLoaderModelOnly` -> `GetICLoRAParameters` (reads `reference_downscale_factor` from the
  LoRA's metadata) -> `LTXVAddGuide` (`iclora_parameters`) -> `LTXVCropGuides`, plus
  `RepeatImageBatch` to make the static video. Arm I uses these, so the pod still needs no
  custom nodes. [Direct: ComfyUI v0.39.2 `comfy_extras/nodes_lt.py`]
- **A workflow mismatch.** The official docs say the published 2.5 workflow JSON "currently
  loads the LTX-2.3 Ingredients checkpoint" and warn not to substitute one for the other.
- **Contradicting report.** A user thread (HF LTX-2.5 discussion #61, opened 19 Aug) says 2.5's
  IC-LoRAs were only Canny/Depth/Pose. That predates the adapter's 10 Sep creation, so it is
  stale. [Inferred]

**Identity evidence.** None published: no metric and no multi-person result. There are 6
vendor example videos, which I did not download. [Supported: absence]

**Multi-person.** It is designed for one sheet holding several characters, props and locations.
That is the most natural fit for 2-3 person shots of any self-hosted option. [Supported:
vendor description]

**Cost.** [Speculative] It is unknown and the main thing the trial measures.

- The guide is a full extra video's worth of frames in the sequence. Attention is quadratic in
  tokens, so expect roughly 1.5-3x LTX-2.5's per-shot time: **~$22-45** a film, ~$4-8 an
  episode, on a community A6000.
- The official workflow is single-stage at a 544 px short side. Whether it combines with
  lossy's two-stage upscale, and with first-frame I2V for chaining, is **not confirmed**.

**Change: medium.** [Inferred]

1. Paste a per-shot composite locally with PIL (no generation).
2. Upload it and add the 3 IC-LoRA nodes.
3. Write the two-part prompt in `format_prompt`, describing panels by position, not by name, to
   keep 0021's stance.
4. Pin one more model, which needs an HF token on the pod. lossy pins the base LTX-2.5 from the
   ungated `comfyicu` mirror for this reason (`RunPodLtx25Strategy` docstring), and no mirror
   for the adapter is known.

About 150 lines in a sibling strategy.

### 2e. Other routes, briefly

| Route | Why not first | Cost |
|---|---|---|
| Character LoRA on LTX-2 (fal trainer) | $9.60 a character at 2,000 steps: Star Wars IV's 10 humans ≈ $96, over budget. Self-host ≈ 30 min of H100 per character ≈ $1.5 each, but needs a 20-50 image dataset per character, which could come from the sheet generator. The highest ceiling, so a later step if the trial shows refs help but are not enough. | $15-100 a film |
| Post face swap (FaceFusion, hyperswap_1a_256) | Weak on profile and small faces; human faces only. `inswapper_128` (ReActor/Roop) is non-commercial research only, and hyperswap's terms are disputed. About 5 min per minute of 1080p on a 4090. | ~$4-8 a film |
| VACE-14B + SAM2 face mask, regenerating the face region | A full diffusion pass per clip. Copes with pose better than 128 px swappers, but costs as much as 2b. | ≈ 2b |
| LTX community ID LoRAs (LTX-Best-Face-ID, ID-LoRA) | Built for LTX-2.3. The licence and training data of community face-ID weights are unknown. | — |

[Supported: fal trainer page, InsightFace licensing page, clore.ai guide, HF cards; costs
Inferred]

## 3. Recommendation and trial

**Do 2c (keyframe-first) and 2d (Ingredients) as two arms of one small trial on the current
LTX-2.5 pod. Defer everything else.**

- **Why both arms.**
  - 2c is cheap and almost certain to work mechanically: the I2V path is proven in 0020 and
    0028. Its weakness is identity decay after frame 0.
  - 2d conditions every frame, which is exactly that weakness, but its cost and 2.5
    compatibility are unknown.
  - One pod session measures both against the same text-only control.
- **Don't touch the Star Wars IV schedule.** Whichever arm wins becomes a decode strategy flag
  in a follow-up job. If the trial has not run by 30 Nov, the run goes ahead as planned.

### Data shape for the follow-up implementation

The `characters.json` entry gains an optional `reference` object:

```json
"reference": {"sheet": "characters/george_sheet.webp", "seed": 1234,
              "generator": "flux2-klein-4b@<hf-rev>", "views": ["front", "three_quarter", "profile", "full_body"]}
```

Invariants:

1. `keep_name` characters never get one.
2. The sheet prompt is built from `description` only, never a name.
3. Every shot of a character in one decode uses the same sheet.
4. A missing `sheet` file is regenerated from (`description`, `seed`, `generator`), so the
   payload need not carry it.

A per-shot keyframe is a decode cache, `keyframes/<NNNN>.png`, never part of the payload.

### Trial: Seinfeld S01E03, 16 shots

**Why Seinfeld, not `sw_r2_leia`.** `sw_r2_leia`'s recurring characters are mostly droids,
which are `keep_name` designs with no sheet (0020: R2-D2 in shots 8/19/26), and experiment 0004
showed the danger of a range with no recurring humans. Seinfeld has four human leads who recur
in nearly every shot (identity blocks in 97% of first parts, 0028 data). It has a 1080p source,
an existing encode, the `--ab-manifest` runner, and a free baseline. [Direct: 0028]

**Shots.**

- 3 single-character shots for each of Jerry, George, Elaine and Kramer, plus 4 two-character
  shots: 16 shots.
- All one-part, 1-3 s, chosen evenly through the episode, in medium or close framing so a face
  is scoreable.
- Exclude shots under 1 s.
- Built by a `make_trial_manifest.py` modelled on `0028-style-drift/make_ab_manifest.py`, with
  seed = shot index throughout.

**Arms (48 clips).**

| Arm | Prompt | Image input |
|---|---|---|
| T: text control | today's `format_prompt` with the photoreal style prefix (e54fa16) and photographic identity blocks | none (current production) |
| K: keyframe-first | same text | klein-composed first frame from the characters' sheets → `LTXVImgToVideo` |
| I: Ingredients | the same text inside the two-part sheet prompt | per-shot composite of the characters' sheets via IC-LoRA |
| (optional) R: Kling O3 R2V | same text | same sheets as elements; a hosted ceiling, ~$4, only if Will wants a ceiling number |

**Scoring.** A blind identity-consistency rubric, the 0025 method adapted to identity:

1. **Per-character, per-arm grids.**
   - For each character, take the middle frame of every shot it appears in (6-7 shots per
     character). Crop to the character with a face detector, or use the whole frame if none is
     found.
   - Build one grid per arm. Shuffle arm identity, tile order and position with a seeded RNG,
     using `blind_pack.py`'s approach.
   - A fresh vision-model judge, Gemini or a Claude subagent with no key, scores each grid:
     - **"same person in every tile?"** 1-5;
     - **odd-one-out**: which tile is least like the rest, if any;
     - **"matches this description?"** 1-5, given the identity block text.
   - Two passes with different shuffles, as in 0025, to check position bias. Read differences
     under ~0.3 as noise.
2. **Cross-arm pairwise.** For each character, show one tile from each arm's grid side by side
   and ask which set is most internally consistent.
3. **Objective face metric.** Mean pairwise cosine of face embeddings across a character's
   shots, per arm. Use InsightFace `buffalo_l` (non-commercial research licence, fine for an
   internal measurement, not shipped) or facenet-pytorch. It runs locally on CPU, for free.
   - For K, also measure frame 0 vs the middle frame vs the last frame against the sheet. That
     gives the I2V decay curve no source publishes.
4. **Regressions.** `0028-style-drift/classify_style.py` on the middle frame of each clip
   (photoreal rate), plus 0025's adherence/artefacts/motion 1-5 on the same packs. Check that K
   and I do not freeze motion or ignore the action.

**Decision rule.**

- **Pass.** Ship the arm as a strategy flag if all of these hold:
  - its identity score beats T by ≥1.0 point, or its embedding cosine beats T by ≥0.10;
  - its adherence and motion are no worse than T by more than 0.3;
  - its photoreal rate is ≥ T's.
- **Tie between K and I.** Pick the cheaper per film.
- **Both fail.** Next is the character-LoRA route (2e), or Phantom/VACE-14B (2b) at 480p as its
  own costed test.

**Cost estimate** (A6000 secure at $0.59/h, 0026's fitted rate). [Inferred]

| Item | Time |
|---|---|
| Pod boot + model download (LTX-2.5 already pinned; klein ~8-13 GB; adapter 1.3 GB) | ~25 min |
| Sheets (4 characters × 1 sheet, plus 2 retries) and 16 keyframes, klein | ~3 min |
| T and K arms, 16 clips each at ~1 min | ~32 min |
| I arm, 16 clips at ~2 min (unknown multiplier) | ~32 min |
| **Total** | **~90 min ≈ $0.90** |

- Run it under `tools/capped_run.py --cap 1.25 --margin 0.05`, no `--keep-pod`.
- The judge costs pennies. The face metric is free.
- Arm R adds ~$4 of fal spend if wanted.

**Needs from Will before it runs.** [Inferred]

1. Go-ahead for ~$0.90 (cap $1.25).
2. Accept the gated `Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients` repo on HF and put an HF read
   token where the pod setup can read it.
3. Optionally, approve arm R (~$4).

**Code needed before the trial.** About a day, in one job:

- the two strategy subclasses (or an `--ab-manifest` extension that carries an optional image
  input per job);
- the sheet and keyframe pre-passes;
- the manifest builder;
- the scoring scripts.

All of it can be validated with `--dry-run` and no spend, as 0028's runner was.

## Trial results (run 15-16 Nov 2026)

(Arm I was wired afterwards; see "Arm I is wired" below. It has not run.)

Steady task `274d2328`. Will approved the run at 19:10 NZ on 15 Nov with a $1.25 cap. Arms T and
K ran; arm I (Ingredients) and arm R (Kling) were skipped as instructed.

**Verdict: K passes the decision rule on every line. Recommendation for SW4: adopt K, as a
strategy flag built and checked in a follow-up job before 30 Nov, with the two fixes below.
Don't wait on I.** [Direct: `trial/results.json`]

### What ran

- **Shots.** 16 one-part Seinfeld S01E03 shots, medium or close, 1.6-2.9 s, spread through
  the episode: 3 each of Jerry, Elaine, Kramer and George, plus 4 Elaine+Jerry shots, the only
  pair in the episode that met the filter. George had too few clean shots of 3 s or less, so his
  were topped up from shots of up to 4 s. Seed = shot index. [Direct: `trial/make_trial_manifest.py`,
  `trial/trial_plan.json`]
- **Text, identical in both arms.** 0028's photoreal style prefix plus one name-free photographic
  identity block per character. Jerry, George and Elaine were converted from the episode's own
  `characters.json` text (wardrobe and demeanour dropped, nothing added from knowing the actors).
  Kramer's is 0028's `KRAMER_PHOTO`. The blocks are prepended by the real
  `CharacterIdentityMixin`, and no prompt contains a name. [Direct]
- **K's references.** FLUX.2 [klein] 4B made one 2x2 sheet per character (front, 3/4, profile,
  full body) from the block alone, then one 1280x704 first frame per shot from that shot's
  sheets (at most 2) plus the shot's own prompt. It ran on fal rather than on the pod. These
  are the same Apache-2.0 weights, and this kept the pod to the already-proven LTX-2.5 graph.
  [Direct: `trial/make_refs.py`, `trial/k_keyframes.jpg`]
- **Video.** LTX-2.5 two-stage via `decode.py --ab-manifest`. K jobs carry a `start_image`
  (new optional manifest field), which feeds the existing `LTXVImgToVideo` branch, with the
  stage-2 re-pin. All 32 clips completed with no failures. [Direct: `evidence/capped_run.log`]

| Spend | USD |
|---|---|
| RunPod (balance before minus after, under `capped_run --cap 0.87`; no pods left) | 0.60 |
| fal klein: 4 sheets + 16 keyframes (upper bound, counting input MP; ~$0.17 if billed on output only) | ≤0.38 |
| Gemini judge (`gemini-3.1-flash-lite` on Vertex; AI Studio was rate-limited) | 0.06 |
| **Total** | **≤1.04 of the 1.25 cap** |

### Scores

| Measure | T (text) | K (keyframe) | Rule | |
|---|---|---|---|---|
| Blind grid "same person", 1-5 (both passes identical) | 3.50 | **5.00** | K ≥ T + 1.0 | pass |
| Face-embedding cosine, mean pairwise across a character's shots | 0.234 | **0.547** | K ≥ T + 0.10 | pass |
| Same, single-character shots only | 0.305 | 0.524 | | |
| Prompt adherence, 1-5 (per-shot clip judge, 2 passes) | 4.50 | 4.31 | ≥ T − 0.3 | pass (−0.19) |
| Motion, 1-5 | 4.44 | 4.25 | ≥ T − 0.3 | pass (−0.19) |
| Artefacts, 1-5 | 4.41 | 4.28 | | −0.13 |
| Photoreal rate (0028 classifier prompt) | 100% | 100% | K ≥ T | pass |
| Measured motion, mean abs frame difference / frozen-step share | 2.61 / 10.3% | 3.26 / 4.6% | | K moves *more* |
| Matches the description, 1-5 | 4.25 | 5.00 | | |

[Direct: `trial/results.json`. Face cosine uses OpenCV-zoo YuNet + SFace on CPU, where SFace's
same-person threshold is ~0.36.]

Per character, face cosine T → K: Jerry 0.33 → 0.54, George 0.34 → 0.62, Kramer 0.16 → 0.64,
Elaine 0.11 → 0.40. The judge's per-character "same person" scores, T → K: Jerry 3 → 5,
George 5 → 5, Kramer 1 → 5, Elaine 5 → 5.

**Read the judge with care.** [Direct + Inferred]

- **Lenient grid judge.** It gave T's Elaine 5/5 although her face cosine is 0.11. By eye,
  T's Elaine is several different women who share curly hair (`trial/identity_grids_T_vs_K.jpg`).
  The cosine and the contact sheet are the stronger evidence here. The judge is the lite model,
  not 0025's `gemini-2.5-flash`, which AI Studio was rate-limiting and Vertex does not serve
  on this project.
- **Position bias in the pairwise judge.** Set A won 7 of 8 comparisons. Counting only
  verdicts that held across both passes, K wins Jerry and Kramer, and George and Elaine are
  split. The raw tally is K 6, T 2.
- **Mis-crops in T's grids.** In T's two-person shots, two of Elaine's tiles are wrong crops
  (a man's head, and a poster face). Those faces are matched to the arm's own single-shot faces,
  and T has no stable face to match to. That lowers T's pair-shot cosine slightly. The
  singles-only row avoids it and still shows +0.22.

**K decay curve** (no published figure exists). Mean face cosine to the character's sheet:

| Keyframe | Clip first frame | Middle | Last |
|---|---|---|---|
| 0.689 | 0.631 | 0.619 | 0.622 |

T's floor against the same sheets is ~0.25. Most of the drop happens between the keyframe and
the clip's first frame: the stage-1 half-resolution encode plus the stage-2 refine. After that,
identity holds flat across these 1.6-2.9 s clips. Shots longer than ~3 s were not tested.
[Direct]

**Reference: the real episode on the same metric.** Single-character shots, source keyframes,
scored locally only: mean 0.45 (Jerry 0.62, George 0.69, Kramer 0.27, Elaine 0.20). Real
footage varies in pose and lighting. K scoring above the source suggests K is quite uniform,
partly because of the next point. [Direct]

### Problems found (fix before SW4)

1. **The sheet's clothes leak into the shot.** Many K frames keep the sheets' plain grey
   t-shirt even when the shot describes a blazer or a suede jacket. That is most of K's −0.19
   adherence. Fix: head-and-shoulders-only sheets, or a keyframe prompt that names the shot's
   wardrobe first and tells the editor to take only the face and hair from the sheet. [Direct:
   `trial/k_keyframes.jpg`; Inferred fix]
2. **The sheets can invent look-alikes.** Jerry's and Kramer's generated sheets have face cosine
   0.78 with each other. Their blocks both say dark hair and slim, and klein drew two similar
   young men. K's cross-character cosine is 0.45, against T's 0.26. Continuity within a
   character went up, but separation between characters went down. Fix: a distinctness gate
   when sheets are made (reseed any sheet whose face cosine to another sheet is over ~0.5).
   [Direct; Inferred fix]

### Decision and SW4 recommendation

- **Adopt K.** It clears every line of the decision rule, and it costs pennies per episode.
- **Ingredients (I) is not needed to pass.** K's weakness was meant to be within-shot decay,
  and the curve shows little of it over ≤3 s. Test I only if SW4's longer chained shots
  later show drift.
- **Build and verify in a follow-up job before 30 Nov.** Build it as a
  `RunPodLtx25KeyframeStrategy` flag with the two fixes above, then check it on a short Star
  Wars segment. If it is not ready and checked by 30 Nov, run SW4 as scheduled with text.
  This trial changed no production default and not the schedule.
- **Where the keyframes run.** fal klein edit is about $0.009-0.03 per keyframe (billing
  basis unclear), so roughly $10-60 for SW4's character shots. That is too much of a $50-60
  budget. Self-hosting klein on the LTX pod (§2c, ~+$1 a film) is the production route. It
  needs a klein ComfyUI graph validated on a pod, which this trial skipped by using fal.
  [Inferred from fal list prices]
- **No sheets for design characters.** `keep_name` characters (droids, Vader, stormtroopers)
  still get none (§1). K only touches human-character shots.

### Arm I is wired; what Will must do to run it [HUMAN]

Arm I is built and dry-run checked (job `1a4bcabe`, no pod, no spend). Only the gate click and
a token are missing.

**Steps for Will**

1. Signed in at huggingface.co, open
   https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients and click **"Agree and
   access repository"** (the LTX-2.x community licence; the button also consents to Lightricks
   marketing emails).
2. At https://huggingface.co/settings/tokens, create a **Read** token **on that same account**.
   A fine-grained token also works if it has "Read access to contents of all public gated repos
   you can access".
3. Add `HF_TOKEN=hf_...` to `/Users/will/projects/lossy/.env`. It is not there today. The token
   stays on this machine and is sent to the pod over ssh stdin, not on any command line.
4. Optional free check, before spending anything. It builds the arm and tests the gate:
   ```
   uv run python docs/research/0029-character-reference-continuity/trial/make_ingredients_arm.py RUN_DIR
   ```
   It ends with `HF gate OK` or an error that names the missing step.

**The one command** (from anywhere; RUN_DIR is the trial's run directory):

```
docs/research/0029-character-reference-continuity/trial/run_arm_i.sh RUN_DIR 1.00
```

- **What it does.** It composes the per-shot sheets, checks the gate (it stops there at no cost
  if the gate is closed), then runs `tools/capped_run.py --cap 1.00 --margin 0.05 --no-project`
  around `decode.py RUN_DIR --strategy runpod-ltx25-ingredients --ab-manifest
  RUN_DIR/trial_manifest_I.json`.
- **What RUN_DIR needs.** `trial_plan.json` and `trial_manifest.json` (both committed under
  `trial/`) and `sheets/<name>.png`, the four klein sheets. Reuse the run directory of the T/K run:
  arm I then sees the same sheets as K, which is the fair comparison. If the sheets are gone,
  `make_refs.py RUN_DIR --sheets-only` remakes them for about $0.02 on fal, and the script says so.
- **Cost.** About $0.5-0.9 [Speculative]: ~25 min of boot and downloads (+1.3 GB for the adapter),
  then 16 clips at an unknown 1.5-3x the ~1 min of a T clip. The cap is a hard stop. Clips land in
  `RUN_DIR/clips/runpod-ltx25-ingredients-ab/I/`.
- **Then score.** Arms T and K ran earlier, so score I against T:
  `uv run python docs/research/0029-character-reference-continuity/trial/score.py RUN_DIR MODELS_DIR --arms T,I`
  (`--arms T,K` still reproduces the first result). Apply the §3 decision rule to I's row.

**What was built and checked here (no pod)**

- `ingredients.py`: the sheet panel model, the card's two-part prompt and the composite drawing.
- `RunPodLtx25IngredientsStrategy` (`--strategy runpod-ltx25-ingredients`): the LTX-2.5 two-stage
  graph plus the guide, with the adapter pinned by revision and sha256 (`ff873a5b...`, from the
  HF API). It only runs with `--ab-manifest`: the sheet is per shot, and the normal decode path
  has no sheet source. No production default or schedule changed.
- **The gate fails cleanly.** HF answers 401 for a missing or bad token and 403 for a valid token
  whose account has not accepted the gate. Both, and a missing `HF_TOKEN`, stop the run before a
  pod exists with: "Accept the gate at <url> ... then set HF_TOKEN=hf_...". The same error, with
  the pod terminated and no re-provision retry, covers a refusal that only shows at download time
  (wget exit 6). Checked live against huggingface.co with no token (clear message) and a bogus
  token (401, the same message). The 403 case, a valid token without the click, is covered by a
  mock only, because the job has no token. [Direct: `evidence/arm_i_dry_run.txt`]
- The graph is checked structurally in `tests/test_ingredients.py`: every link resolves, the guide
  is cut out before the upsample, stage 2 uses the plain model, and the 16 trial jobs build.
  Node names and inputs were read against ComfyUI v0.39.2's source. It has never run on a pod.

**Risks the run will answer** [Inferred]

- **Two-stage compatibility.** The vendor's 2.5 workflow is single-stage. Arm I applies the guide
  at stage 1 only and refines with the plain model at stage 2, the LTX-2 pipeline's way. If clips
  show the sheet's layout or lose identity at stage 2, the next try is single-stage at full size.
- **Guide resolution.** The sheet is encoded at stage-1 size (640x352), so each face is about
  150 px on the guide. Two-character sheets are worse than one. Head-and-shoulders panels would help.
- **Wardrobe.** The sheets show a plain grey t-shirt and the panel text says so, so the model may
  keep it, as K did. Compare the shot's wardrobe against the clips.
- **Prompt cost.** Sheet text plus the repeated looks makes the prompt about 2x as long as T's.

### Keyframe strategy built

Built and mock-tested in job `0163ceb5` (no pod, no fal, no Gemini, no spend). Decisions are in
`decisions.tsv` (rows `kf-*`).

**What exists**

- `--strategy runpod-ltx25-keyframe` (`RunPodLtx25KeyframeStrategy`, a subclass of
  `RunPodLtx25Strategy`). Opt-in: no default, no schedule, no other strategy's behaviour changed.
- `keyframes.py` holds the logic. Before the video pass, `prepare()` (called by `run_decode`
  before any pod exists) draws one sheet per person and one first frame per keyframe shot, cached
  under `characters/<name>_sheet.png` and `keyframes/kf_NNNN.png` (keyed on prompt, generator and
  sheets, so a re-encode regenerates). `generate` then runs the unchanged LTX-2.5 I2V graph with
  that frame as **part 0's** start image; later parts chain as before.
- **Routing.** A shot gets a keyframe only if every character in it is a person with a description.
  Any `keep_name` design character in the shot (droid, Vader, trooper), no character, or a missing
  description sends the whole shot to the text path. I chose "any" rather than "only design
  characters" because the editor would have to invent the droid and I2V would lock the clip to it.
  This will cost coverage on Star Wars shots with Luke and a droid together; the dry run prints
  the count.
- **Fix 1, wardrobe.** Sheets are head-and-shoulders only, framed on head and neck with no clothing
  in frame (the full-body panel and the grey t-shirt are gone), and the keyframe prompt now leads
  with the scene, which carries the shot's wardrobe, and takes only face, hair, skin, age and
  build from the sheet. The trial had the clothes instruction last.
- **Fix 2, look-alikes.** Sheets are drawn most-prominent character first. Each new sheet's
  face centroid (OpenCV-zoo YuNet + SFace, as the scorer used) is compared with every accepted
  sheet. Cosine over 0.5 means a redraw with a new seed, three draws in all, and then the sheet is
  dropped: that character's shots use text rather than ship a look-alike. A sheet with no
  detectable face gets the same treatment.
- **Failure handling.** One failed or blank editor call, a missing sheet, a spent budget or a
  failed upload leaves that shot on text. The shot is never failed. A missing `FAL_KEY` or missing
  face models stops the run in `prepare()`, before any pod.
- **Generator interface.** `KeyframeGenerator` (`preflight`, `make_sheet`, `make_keyframe`).
  `FalKleinGenerator` is the trial's fal klein call. A self-hosted klein is a second
  implementation; its ComfyUI graph is not built here.
- **fal cap.** `--keyframe-fal-cap USD` (default 1.00) is an estimated-spend hard stop that counts
  what `keyframes/cost.jsonl` already holds, because `capped_run` caps RunPod only and restarts
  decode as a new process.
- **Dry run, free:** `uv run python decode.py OUTPUT_DIR --strategy runpod-ltx25-keyframe --dry-run
  [--start-index N --limit M]` prints each shot as `KEYFRAME <characters>` or `text <reason>`, the
  sheets it would draw, one sample editor prompt and a fal upper bound. No network call, nothing
  written, works with no `FAL_KEY`. (`--dry-run` on any other strategy without `--ab-manifest` now
  exits with an error; before, it was ignored and started a real decode.)
- Checked with `tests/test_keyframes.py` (60 mock tests): keyframe passed as part 0's start image,
  later parts still chaining, fallbacks for design characters, nobody, no description, a failed
  edit, a failed upload and a dropped sheet, both sheet fixes, the fal call and its cap, the
  decode-loop hook, and the dry run with sockets blocked. The dry run was also driven through
  `decode.py` on a synthetic 9-shot output directory (`evidence/keyframe_dry_run.txt` in the job
  workspace). No Seinfeld or Star Wars output directory was available to the job, so it has **not**
  been run on real manifest shots.

**What is unverified** (everything that needs a paid call or a real model file)

1. The fal calls are the trial's own (`subscribe`, `upload_file`, same models and arguments) but
   were not run here; only fakes ran.
2. The OpenCV face embedder is a port of the scorer's `Faces` class. It has not run: the ONNX
   models are not in the repo or this workspace (`LOSSY_FACE_MODELS`, default `models/face`).
3. The head-and-shoulders sheet prompt has never been drawn. klein may still show a collar, or
   draw a worse likeness without the full-body panel.
4. That the reordered keyframe prompt stops the grey-t-shirt leak. The gate checks distinctness
   and that a face exists, not that the sheet is clothing-free.
5. Star Wars coverage: how many shots survive the "no design character" rule.
6. Shots over ~3 s, split shots (keyframe is part 0 only), wide shots with small faces, and a
   keyframe whose composition fights the camera move, as before.

**The paid check, for Will to approve** (not run; filed as an attended Steady task)

One short Star Wars IV segment, about 30 shots, with the text strategy as the control.

1. Free first. Put `yunet.onnx` and `sface.onnx` in `models/face/`. Run the dry run above on the
   v2 output and pick a 30-shot window with at least 20 `KEYFRAME` shots (and a few droid shots,
   to see the fallback). Check the sheet list and the sample prompt.
2. K arm: `uv run python tools/capped_run.py --cap 1.0 --margin 0.05 --no-project -- uv run python
   decode.py output/star_wars_iv_v2 --strategy runpod-ltx25-keyframe --start-index S --limit 30
   --keyframe-fal-cap 1.0 --style-prefix ...` (same style prefix as the T arm).
3. T arm: the same window with `--strategy runpod-ltx25` under its own `capped_run --cap 1.0`
   (skip it if clips from this encode and prefix already exist).
4. Look at `characters/*_sheet.png` and a contact sheet of `keyframes/` before judging the clips:
   clothing-free sheets, distinct faces, no droid keyframes. Then score K against T with the §3
   rule (face cosine K ≥ T + 0.10; blind grid ≥ +1.0; adherence ≥ T − 0.3). `trial/score.py` reads
   a trial plan, so it needs a small SW4 plan file first (a free step).

| Line | Estimate |
|---|---|
| RunPod, K arm (30 clips of ~3.5 s, one ~20 min boot, the 0026 model at $0.59/h) | ~$0.70 |
| RunPod, T arm, same shape | ~$0.70 |
| fal klein: ≤ 6 sheets × 3 draws ≈ $0.09, ≤ 30 keyframes × ≤ $0.03 | ≤ $1.00 (the cap) |
| Gemini judge (`gemini-3.1-flash-lite`, the trial's $0.06) | ~$0.10 |
| **Total** | **~$2.5, hard-capped at about $3.0** |

The trial measured $0.60 of RunPod for 32 shorter clips including boot, so the RunPod lines are
likely high. Both RunPod runs are under `capped_run`; fal is under `--keyframe-fal-cap`. If K
fails the rule, SW4 runs on text on 30 Nov, as planned.

### SW4 paid check (21 Nov 2026): free prep done, renders stalled on flaky pod SSH, no verdict

Steady task `a2c14bfa`, job `61de4147`. Will approved ~$2.5 (cap $3.0). **Only 12 K clips and 1 T clip
were rendered, so there is no K-vs-T result and no verdict yet. SW4 stays on text as scheduled** (30 Nov).

**Free prep (done)**
- Face models: opencv_zoo `face_detection_yunet_2023mar.onnx` and `face_recognition_sface_2021dec.onnx`
  in `models/face/` as `yunet.onnx` / `sface.onnx` (not committed; `LOSSY_FACE_MODELS` default).
- Dry run on the whole SW4 v2 encode (2070 shot entries): **795 keyframe shots, 1275 text** (design
  character 1047, no character 228). So the "no design character" rule keeps 38% of the film for K
  (this closes unverified item 5 above). 12 sheets would be drawn for the full film.
- Window chosen: shot indices 684-713 (30 shots, 64 s, Luke/Obi-Wan/Han scene with 5 droid/design
  shots for the fallback). The dry run shows 20 KEYFRAME shots. Picked for short shots (cost) and three
  distinct lead faces; windows with 20+ keyframe shots are common (best: 30/30 at index 370).
- Plan builder for `score.py`: `trial/make_sw4_score_inputs.py` (lays out the run dir `score.py` reads,
  from two plain decode output dirs; **untested**, no clips existed). `trial/sw4_prepare.py` draws the
  K arm's sheets and keyframes with no pod, so they can be inspected before RunPod is paid for.

**Sheets and keyframes, looked at before any pod (fal spend $0.31)**
- **Look-alike gate dropped Han Solo's sheet.** Han's face cosine to Luke's was 0.51, 0.53, 0.55
  over three draws (limit 0.5), so all 7 Han shots in the window fell back to text. The window
  therefore had **13 keyframe shots, not 20** (Luke 9 incl. pairs, Obi-Wan 6 incl. pairs, Luke+Owen 1).
  Han and Luke have near-identical text descriptions (young, light-brown hair, fair skin), so the gate
  fires. On a full film the gate will route many shots of look-alike pairs to text. [confirmed, run log]
- **Fix 1 (clothing-free sheets) holds for Luke only.** Luke's sheet is bare-shouldered. Obi-Wan's sheet
  shows a cream robe collar and Uncle Owen's a navy shirt: klein still draws clothes on older men
  despite "no clothing in frame". [confirmed, `evidence/sw4_sheets.jpg` in the job workspace]
- **Keyframes look right.** The 13 frames show one consistent Luke and one consistent Obi-Wan, distinct from each
  other, each in the scene's own wardrobe (white tunic, grey/black tops, brown robe); the grey t-shirt
  leak of the first trial is not visible. No droid keyframes. [confirmed by eye, `evidence/sw4_keyframes.jpg`]

**Render attempts: K 12 of 30 clips, T 1 of 30, no verdict.** Pod SSH is intermittent, not dead.
In the first attempt (cap $1.0, 14 pods) every pod failed at setup with `ComfyUI upgrade to v0.39.2
failed: Permission denied ... Too many authentication failures` (ssh to the pod). A later one-shot
probe booted, upgraded ComfyUI, downloaded the models and rendered shot 684 (T arm, $0.04), and one K
run then rendered 12 clips (shots 684-694, $0.30) before the session was cut off. Every later attempt
(two more K, two more T, retries on) failed the same way: about 50 of ~55 pods across the job were
refused at the ssh step, on every GPU type and both clouds (A40, A6000, L40, L40S, RTX 6000 Ada), so it is not
a bad host. `runpod_pod._ssh_opts` passes no `-i`/`IdentitiesOnly`; the pod is given
`~/.ssh/id_ed25519.pub` as `PUBLIC_KEY` and ssh then offers whatever keys it finds. Cause not established:
the job was denied access to `~/.ssh` and I did not work around that. Plausible [unverified]: the key
is not reliably installed on the pod by the time `Waiting for SSH` returns (a race), an `id_ed25519.old`
pair sits beside the live key, or the job sandbox only sometimes reaches the key. Each failed pod costs
$0.01-0.05, but 23 of them in the last two runs burned ~$0.8 with no clip, which is what exhausted the budget.
One pod was left running when a `capped_run` was killed by the Bash time limit; I terminated it by hand.

**Spend**: fal $0.43 (`keyframes/cost.jsonl`, sheets and 13 keyframes); RunPod about $2.26 by
balance (3.96 -> 1.71; this may include another lossy run's pod on the same account, since `capped_run`
saw one already running at start). Gemini $0. **About $2.7 of the $3.0 cap**, of which only the 12 K
clips and one T clip are usable. I stopped there: finishing both arms needs about $0.9 more.

**Usable now**: K clips for shots 684-694 (7 of the 13 keyframe shots), T clip for 684 only.
Not enough for a K-vs-T comparison; I did not score a partial result.

**To finish**: (1) make pod SSH reliable: retry the upgrade/setup ssh step until the key is accepted,
or pass an explicit key with `IdentitiesOnly=yes`, and confirm `id_ed25519` matches the `.pub` (needs
Will). (2) About $1.0 more RunPod plus ~$0.1 Gemini, which is over the $3.0 cap Will approved, so it needs
a new approval. The K run resumes from the 12 clips on disk and fal is already paid. (3) Run T on 684-711
(28 shots minimum; only shots with both arms are scored), then `make_sw4_score_inputs.py` and
`score.py`. The K arm will be judged on at most 13 keyframe shots; if that is too thin, redraw Han with
a more distinct description first.

### Trial files

- `trial/make_trial_manifest.py`, `trial/make_refs.py`, `trial/score.py`: the builder, the
  reference maker and the scorer (`--arms T,I` for arm I).
- `trial/make_ingredients_arm.py`, `trial/run_arm_i.sh`: arm I's sheet composer, gate check and
  manifest, and the one command that runs it.
- `trial/trial_manifest.json`, `trial/trial_plan.json`, `trial/refs_cost.jsonl`,
  `trial/results.json`, `trial/grid_key.json`: inputs and results.
- `trial/identity_grids_T_vs_K.jpg`: the sheets and per-arm face grids. `trial/k_keyframes.jpg`:
  the 16 K first frames.
- The clips, full-size sheets, keyframes, frames and grids (61 MB) are outside git, in the job
  workspace `evidence/trial0029/`, with the run log in `evidence/capped_run.log`.

## Bake-off: Qwen first frames, Wan VACE, Ingredients, Kling (20-21 Nov 2026)

Steady task `efc02dbe`, job `9b74e2bf`. Will approved up to $5 (all providers) to try more routes on the
same 16 Seinfeld shots. No production default and no schedule changed. **Result: neither arm that
ran beats K. KQ (Qwen first frames) fixes identity as well as K does but costs more adherence and
motion than the rule allows; W (Wan VACE-14B reference-to-video) does not hold identity better than
text. Arm I could not run (no `HF_TOKEN` reachable from the job), and arm R (Kling) did not fit the
budget. K stays the recommendation.**

### What ran, and where it differs from the first trial

| Arm | What | Status |
|---|---|---|
| T | text only | **Re-rendered** (see below) |
| K | klein first frame -> LTX-2.5 I2V | Not rerun: the first trial's numbers are quoted |
| I | LTX-2.5 Ingredients | **Not run**: no HF token (below) |
| KQ | Qwen-Image-Edit-2511 first frame (fal) -> the same LTX-2.5 I2V | Ran |
| W | Wan 2.1 VACE-14B on fal, the klein sheets as `ref_image_urls`, T's text | Ran |
| R | Kling O3 reference-to-video | Skipped: about $4.03 against about $1.3 left |

Deviations from "reuse T and K, don't rerun". [Direct]

- **T clips were re-rendered.** The first trial's clips and sheets live in an earlier job's workspace,
  which this job may not read, and `score.py` judges two arms' clips side by side. So T ran again in
  the same pod session as KQ, same 16 shots, seeds and text (about $0.2). It reproduced the first
  T closely: face cosine 0.266 against 0.234, blind "same person" 3.75 against 3.50, adherence
  4.50 and motion 4.44 (both identical). K's row below is the first trial's, not rerun.
- **The four klein sheets were drawn again** (same prompt and seeds, $0.02), so they may differ a
  little from the first trial's. KQ and W use these.
- **Each new arm is scored against the new T in its own pair**, with the first trial's protocol:
  `score.py --arms T,KQ` and `--arms T,W` (results in `trial/results_T_KQ.json`,
  `trial/results_T_W.json`). The K column is not in a pair with them, so read K vs KQ as two
  separate comparisons against T, not a head-to-head.
- **Judge.** `gemini-3.1-flash-lite` on Vertex (AI Studio was rate-limited again), the same as the
  first trial. It is lenient: it gave T's George 1/5 here and T's Elaine 5/5 in the first trial
  although both have a low face cosine. Lean on the cosine and the contact sheets.
- **W is a different resolution and frame rate**: 832x480 at 16 fps (VACE's output; LTX is 1280x704
  at 24 fps). 480p was the cheapest tier and the only one that fit. Small and profile faces score
  worse on the embedder at 480p. The "motion" measure (mean frame difference) is per frame, so it
  is not comparable across frame rates; read only the judge's motion score for W.

### Scores

| Measure (rule) | T first trial | T here | K first trial | **KQ** | **W** |
|---|---|---|---|---|---|
| Blind "same person", 1-5 (>= T + 1.0) | 3.50 | 3.75 | 5.00 | **5.00** (+1.25, pass) | 4.25 (+0.50, **fail**) |
| Face cosine across a character's shots (>= T + 0.10) | 0.234 | 0.266 | 0.547 | **0.506** (+0.24, pass) | 0.270 (+0.004, **fail**) |
| Same, single-character shots | 0.305 | 0.270 | 0.524 | 0.537 | 0.301 |
| Prompt adherence, 1-5 (>= T - 0.3) | 4.50 | 4.50 | 4.31 | 3.94 (-0.56, **fail**) | 3.91 (-0.59, **fail**) |
| Motion, 1-5 (>= T - 0.3) | 4.44 | 4.44 | 4.25 | 3.91 (-0.53, **fail**) | 4.09 (-0.35, **fail**) |
| Artefacts, 1-5 | 4.41 | 4.41 | 4.28 | 4.06 | 4.28 |
| Photoreal rate (>= T) | 100% | 100% | 100% | 100% (pass) | 100% (pass) |
| Cosine between different characters (lower is better) | 0.26 | 0.30 | 0.45 | 0.43 | 0.33 |
| Pairwise "most consistent" wins | | | K 6, T 2 | KQ 6, T 2 | W 4, T 4 |
| **Rule** | | | **pass** | **fail (adherence, motion)** | **fail (identity, adherence, motion)** |

[Direct: `trial/results_T_KQ.json`, `trial/results_T_W.json`, `trial/results.json`. n = 16 clips, one
judge call per grid and per clip, two passes, so a 0.3 gap is near the noise floor.]

- **KQ holds identity as well as K.** The cosine is 0.51 against K's 0.55, and the judge gives all
  four characters 5/5. Against its own sheet, KQ's frames lose little from keyframe to last frame
  (Jerry 0.64 -> 0.63, Elaine 0.64 -> 0.58), the same flat decay as K.
- **KQ loses on adherence and motion**, by about three times K's margin (-0.56 against -0.19).
  The keyframes show it: Qwen kept the sheets' plain grey t-shirt in most of the 16 frames
  (`trial/kq_keyframes.jpg`; counted by eye, about 11), the same wardrobe leak as K. Measured motion also fell: 9.7% of
  frame steps are near-frozen, against 6.9% for T and 4.6% for K. This is **one editor-swap on 16
  shots, so I would not call Qwen worse than klein for sure**, only that it gave no gain and
  cost more. [Direct for the numbers; the cause is Inferred.]
- **W does not hold identity.** Cosine to T is +0.004. Its George and Kramer are close to their
  sheets (0.44-0.47) but Jerry and Elaine are not (0.28-0.31), and the contact sheet
  (`trial/w_contact.jpg`) shows many profile and turned-away faces plus the same grey t-shirt.
  Part of the miss may be 480p; the judge still lifted it by 0.5.
- **Cross-character separation.** KQ's 0.43 matches K's look-alike effect (0.45). The two fixes
  built into `runpod-ltx25-keyframe` (a distinctness gate and head-and-shoulders sheets) are for
  this and were not applied to KQ here.

### Cost, time and projection for Star Wars IV (2,012 shots, 7,482 s)

Spend this job, every charge logged in `trial/cost_bakeoff.jsonl`. [Direct]

| Line | USD |
|---|---|
| fal klein: 4 sheets | 0.02 |
| fal Qwen-Image-Edit-2511: 16 keyframes (upper bound, input MP counted; $0.43 on output MP only) | <= 1.06 |
| fal Wan VACE-14B: 16 clips, 480p, 40.8 video-s at $0.04/s | 1.63 |
| RunPod, balance before minus after: T (16) and KQ (16) clips, with three pod boots (two hit a RunPod API error that the retry handled, one I stopped by hand and terminated) | 0.86 |
| Gemini judge, two scoring runs | 0.11 |
| **Total, upper bound** | **<= 3.69 of the 5.00 cap** |

| Arm | Seconds per shot | Cost, this trial | Projected for SW4 (all 2,012 shots) |
|---|---|---|---|
| T | 93 (A6000/A40 pod, 16 clips) | RunPod share about $0.4 | about $14.90 (0025) |
| K, from the first trial | not timed per clip (planned at about 60) | <= $0.38 fal + RunPod | $14.90 + $17-38 klein on fal, or about +$1 self-hosted |
| **KQ** | 107 (median of 16), plus the fal edit | <= $1.06 fal + about $0.4 RunPod | $14.90 + **$54-133** Qwen on fal (every shot; fewer in practice, since design-character shots get none), about +$6-11 self-hosted [Inferred, unbenchmarked] |
| **W** | 57-273 each with 8 submitted in parallel; 474 for a lone probe (queue time) | $1.63 | **about $299** at 480p ($0.04/s x 7,482 s), $599 at 720p, before any audio |
| I | not measured | not run | $22-45 [Speculative, from the earlier estimate] |
| R | not run | about $4.03 for 16 clips (3 s minimum at $0.084/s) | about $630 (table in 2a) |

W at 480p is 20 times the $14.90 baseline for the same 16 shots and gave no identity gain. KQ on
fal is 4-9 times the keyframe cost of klein for a worse adherence score.

### Arm I: what Will must do

The preflight stopped at the gate: no `HF_TOKEN` was reachable. `.env` has none, and `op run
--env-file .env.tpl` fails inside a Manager job session with "No accounts configured for use with
1Password CLI" (the job's private `op` home has no service-account token). The run, the retry at the
end of the job, and the steer that `.env.tpl` holds the token, all ended the same way. So arm I is
not run, and nothing was spent on it. To run it from a terminal where `op` is signed in, in the
lossy checkout:

```
# 1. free: remake the sheets if RUN_DIR has none (about $0.02) and make the composites
uv run python docs/research/0029-character-reference-continuity/trial/make_refs.py RUN_DIR --sheets-only
# 2. one command; the HF gate check is its first step and stops before any pod if it fails
op run --env-file .env.tpl -- docs/research/0029-character-reference-continuity/trial/run_arm_i.sh RUN_DIR 1.00
# 3. score against T (T's clips must be in RUN_DIR: render the T arm of trial_manifest.json first)
uv run python docs/research/0029-character-reference-continuity/trial/score.py RUN_DIR MODELS_DIR --arms T,I
```

`RUN_DIR` needs `trial_plan.json` and `trial_manifest.json` (both in `trial/`). Its estimate is
$0.5-0.9 [Speculative]. The $1.31 left of this job's $5 would have covered it; it was not spent.

### Arm I attempt (21 Nov, job `dd91522b`): not run, no verdict

Will's follow-up gave arm I the bake-off's last ~$1.3 (hard cap $1.30, all providers) and put
`HF_TOKEN` in the `proj-lossy` vault. **Arm I still has no result.** [Direct: run log, balance]

- **Preflight worked.** Inside the job, `op` resolved the token, the four sheets were redrawn on fal
  ($0.02), the 16 composites and `trial_manifest_I.json` were built, and the gate check printed
  `HF gate OK`. The previous job's T clips and sheets were in another job's workspace, so T had to be
  rendered again for the `--arms T,I` pair, as the bake-off did.
- **Priced before spending.** T about $0.3 (16 clips, ~93 s each, one boot) plus I about $0.5-0.9
  ([Speculative]: 1.5-3x a T clip, +1.3 GB adapter download, another boot), plus ~$0.06 judge: about
  $0.9-1.3, against $1.30. It was marginal and was started because it could fit.
- **What went wrong.** The T render, under `capped_run --cap 0.45`, never produced a clip. Over three
  attempts, every pod boot failed in `RunPodSession` setup at the ComfyUI upgrade step, with ssh `Permission denied,
  please try again` and `Too many authentication failures`; one pod never became reachable in 600 s.
  `~/.ssh/id_ed25519.pub` exists on this machine, so a missing public key is not the cause. I did not
  find the cause: the job has no ssh agent, and diagnosing it needs a live pod. Spend: $0.43 on RunPod
  for zero clips, balance 3.9627 -> 3.5351, no pod left running. [Direct: `capped_run` FINAL line]
- **Why it stopped.** With about $0.85 left, T plus I no longer fits under the cap even if the ssh
  problem were fixed, and an unfixed ssh problem would burn it again. The task says to skip and say so.
  Total spent this job: $0.45 (sheets $0.02, RunPod $0.43). Judge: none.
- **Verdict.** None. Does I beat K is still open; **K stays** the recommendation for SW4. I did not
  run I alone against the quoted T numbers: the rule needs the paired score, and a single unpaired run
  would have cost most of what was left.
- **To finish it.** Fix or find the pod ssh problem first, preferably from a terminal where
  `ssh root@<pod>` works by hand (the previous bake-off job booted pods fine from its session, so the
  difference is this job's environment, not the pod code). Then, from the lossy checkout with `op`
  signed in: render the T arm, `run_arm_i.sh RUN_DIR 0.9`, and `score.py RUN_DIR MODELS_DIR --arms T,I`
  (see "Arm I: what Will must do"). Budget about $1.2-1.5 for T and I together; the $1.30 is not
  enough margin after this attempt's loss, so it needs a fresh approval.

### Arm I rerun (21 Nov, job `7acf878b`): priced, does not fit the $0.85 cap, not run

Will approved a rerun after the pod ssh fix with a hard cap of $0.85 new spend, all providers, and
"price first; skip and say so if it won't fit". **It does not fit, so nothing was spent and there is
no verdict. K stays the recommendation.** [Direct for the checks and the balance; the price is Inferred
from measured per-clip times plus one Speculative number]

**Premise check**

- The ssh fix is on main (`13cbaf1`: one explicit key, `IdentitiesOnly`, sshd probe with backoff,
  fail fast on a rejected key) but is **mock-tested only**. No pod has been booted with it. The
  earlier SW4 check lost about 50 of 55 pods to ssh before the fix (`ec8e131`), so a live boot is
  still the unproven step. [Direct: `git log`, commit messages]
- Access is fine: `op read` resolves `HF_TOKEN` in the job, `pod_ssh.resolve_ssh_identity()` finds
  `~/.ssh/id_ed25519`, and RunPod shows balance $1.685 with no pods running. [Direct]
- T's clips and the four sheets from the earlier jobs are not reachable (other jobs' workspaces), so
  the `--arms T,I` pair needs T rendered again, as in the last two attempts.

**Price against the $0.85 cap**

| Line | USD | Basis |
|---|---|---|
| Four klein sheets (fal) | 0.02 | logged, 21 Nov |
| T, 16 clips, one boot | about 0.30 | 93 s per clip measured (about 25 min), plus boot and downloads; the KQ session cost 0.20 for 7 clips |
| I, 16 clips, one boot | about 0.45-0.80 | one boot plus the 1.3 GB adapter, then 16 clips at 1.5-3x a T clip [Speculative: never timed] |
| Judge, `score.py --arms T,I` | 0.06 | logged, same as the two earlier scoring runs |
| **Total** | **about 0.85-1.20** | |

- The low end only just equals the cap with no margin, and the high end is about 40% over it. The
  last attempt lost $0.43 to boots alone, and `capped_run` stops a run at its cap, so a failed boot
  or a slow I clip leaves a half-rendered arm that cannot be scored.
- **Cheaper variants I rejected.** Running I alone and comparing it with the quoted T numbers (cosine
  0.234/0.266, "same person" 3.50/3.75) is unpaired, and the decision rule needs the paired score;
  the last job refused this for the same reason. Rendering fewer than the 16 shots changes the
  protocol the task says to keep, and at n = 8 a 0.3 gap on the judge is inside the noise.
- **A live ssh probe alone** (one pod boot, about $0.05-0.10) would prove the fix, but it would take
  from a budget that is already short, and it answers the ssh question, not the Ingredients one.

**What it would take.** About $1.3 for T and I together (this note's high end plus margin),
rendered in a terminal where the first boot can be watched. Run `run_arm_i.sh RUN_DIR 0.8` after
the T render, then `score.py RUN_DIR MODELS_DIR --arms T,I`. A cheaper way to buy the pair is to keep
the T clips: this job's cost came mostly from re-rendering T three times because no run directory
survived. Whoever runs it should keep `RUN_DIR` outside a job workspace. That needs a fresh approval
at about $1.3, or a decision that the K recommendation is good enough and arm I is dropped.

**Spend this job: $0.** No pod, fal, or Gemini call. The cap was not touched.

### Can klein and Qwen run on the LTX RunPod pod?

**Yes in principle, probably with no custom nodes. Nobody has run either on the pod, so it is not
proven.** [Direct from sources; Inferred where marked]

- **ComfyUI support.** The pod upgrades itself to v0.39.2. That release's source has
  `Flux2` in `supported_models.py`, the nodes `EmptyFlux2LatentImage`, `Flux2Scheduler`,
  `FluxKontextMultiReferenceLatentMethod` and `ReferenceLatent`, and, for Qwen, `QwenImage`,
  `TextEncodeQwenImageEdit` and `TextEncodeQwenImageEditPlus` (multi-image). ComfyUI's own
  tutorials cover both. [Direct: raw v0.39.2 source; docs.comfy.org klein and Qwen-Image-Edit-2511 pages]
- **Weights, all ungated and public.** [Direct: HF API, 20 Nov]

  | Piece | File | Size |
  |---|---|---|
  | klein 4B | `Comfy-Org/flux2-klein-4B` `flux-2-klein-4b.safetensors` (fp8 variant: 4.07 GB, Apache-2.0) | 7.75 GB |
  | klein text encoder | `qwen_3_4b.safetensors` (fp4 variant: 3.85 GB) | 8.04 GB |
  | klein VAE | `flux2-vae.safetensors` | 0.34 GB |
  | Qwen-Edit-2511 | `Comfy-Org/Qwen-Image-Edit_ComfyUI` `qwen_image_edit_2511_fp8mixed.safetensors` | 20.5 GB |
  | Qwen text encoder | `qwen_2.5_vl_7b_fp8_scaled.safetensors` | 9.4 GB |
  | Qwen VAE | `qwen_image_vae.safetensors` | 0.25 GB |

- **Disk.** `RunPodLtx25Strategy.DISK_GB` is 120 for about 40 GB of LTX weights. klein adds about
  16 GB and Qwen about 30 GB, so both fit. Download time is the cost: about 3 and 5 minutes of
  an already 20-minute boot. [Inferred]
- **Memory.** klein's published footprint is 8-9 GB of VRAM (docs.comfy.org). Qwen's fp8 weights
  are 20 GB plus a 9 GB encoder. The LTX int8 model takes most of a 48 GB card, so the keyframes
  must be a separate pass before the video pass, with ComfyUI told to free models in between
  (`/free`), as 2c already says. Not measured. [Inferred]
- **Speed.** klein: ComfyUI lists about 1.2 s distilled on an RTX 5090; an A6000/A40 is several
  times slower, so seconds per frame (my estimate was 5 s). Qwen: no published A6000 figure, and the
  full 28-step model is 20B; 30-60 s is a guess, and the 4-step Lightning LoRA exists for 2511.
  Neither was benchmarked. [Supported + Speculative]

**What is left to make it work (klein first; Qwen is the same shape)**

1. **A ComfyUI API graph** per editor: loaders (`UNETLoader`, `CLIPLoader` type flux2, `VAELoader`),
   a `LoadImage` plus `VAEEncode` and `ReferenceLatent` chain per reference, `EmptyFlux2LatentImage`,
   `Flux2Scheduler`, sampler and `VAEDecode`. Check node input names against the pod's
   `/object_info`: only the node names are confirmed here.
2. **A `KeyframeGenerator` that calls it** (`keyframes.py` already has the interface and the fal
   implementation `FalKleinGenerator`; this is its second implementation), with `PINNED_MODELS`
   entries and hashes for the weights above.
3. **A pod before `prepare()`.** `RunPodLtx25KeyframeStrategy.prepare()` runs before any pod exists,
   because fal needs none. A self-hosted generator needs the pod first, so sheets and keyframes
   must be made after boot and before the first video, then the models freed.
4. **One paid validation on a pod** (about $0.3): run the graph on the 16 shots, compare its face
   cosine with fal klein's (0.55) so the self-hosted output is not worse, and time it.
5. **Whether it is worth it.** At $0.009-0.019 a shot on fal klein, SW4's keyframes cost about
   $17-38 at most; self-hosting saves most of that but is the main remaining engineering. Qwen is
   not worth the work on this evidence.

### Decision

- **K stays the choice for SW4**, as a strategy flag after its own pod check (see "Keyframe
  strategy built"). Neither new arm gives a reason to change it.
- **Qwen-Image-Edit-2511 as the editor: no.** It matches klein on identity but scored worse on
  adherence and motion in this sample, and costs 4-9 times more per frame on fal.
- **Wan VACE-14B reference-to-video: no.** No identity gain over text, about 20 times the cost, and
  it leaves LTX-2.5.
- **Open:** Ingredients (I) has still never run (a 21 Nov attempt failed at pod ssh setup, see "Arm I attempt"; a second job skipped it on price, see "Arm I rerun"), and Kling (R) is unmeasured.

### Bake-off files

- `trial/make_arm_kq.py`, `trial/make_arm_w.py`: the KQ and W builders (both resumable, both log to
  `refs_cost.jsonl`). `trial/score.py` now takes `--arms T,KQ` and `--arms T,W`.
- `trial/results_T_KQ.json`, `trial/results_T_W.json`, `trial/cost_bakeoff.jsonl`,
  `trial/kq_keyframes.jpg`, `trial/w_contact.jpg`.
- The clips, frames and grids are in the job workspace under `evidence/run0029/`, outside git.

## What we don't know

- **Whether identity survives LTX-2.5 I2V across a whole shot.** No published decay curve
  exists for any model. The trial's frame-0/mid/last measurement is the first number.
- **The Ingredients adapter's quality, speed and compatibility.** The card text is gated and was
  not read (only its example prompts, via the widget data). No identity metric is published. The
  official 2.5 workflow reportedly loads the 2.3 checkpoint. Compatibility with two-stage upscale
  is what arm I tests; first-frame I2V chaining with a sheet is untested and not wired.
- **klein's actual identity hold across 2-person compositions,** and its real reference cap. 4
  comes from a secondary source; the official README was not read.
- **Self-hosted speeds** for klein, Qwen-Image-Edit and Wan-14B R2V on an A6000 are estimates
  from secondary sources, not benchmarks.
- **Hosted R2V identity.** Kling O3, Seedance 2.0, Veo 3.1, Omni Flash and Wan 2.6/2.7 have no
  published face metrics, only marketing.
- **OpenS2V numbers.** The live leaderboard scrape showed duplicated rows, so its figures are
  approximate.
- **Single-source claims.** Gemini Omni 1.2 Flash pricing and the Sora 2 and Veo 3.1 preview
  shutdowns each rest on one secondary source and were not verified here.
- **Licences for BindWeave, Stand-In and SkyReels-V3** were not verified.
- **Whether an invented, consistent face is what Will wants for Star Wars,** as opposed to a
  face that reads as the famous cast. This note argues for invented faces on codec-honesty and
  likeness grounds. The trial's contact sheet is where he decides.

## Files

- `research.md`: this note.
- `size_probe.py`: the free local payload-size probe. Its output is in
  `evidence/size_probe_out.txt` in the job workspace, reproduced in §1.
- `decisions.tsv`: the decision trail.
- Sources: OpenS2V-Eval leaderboard https://bestwishysh-opens2v-eval.hf.space/ and arXiv
  2505.20292; Phantom arXiv 2502.11079; Wan2.1-VACE-14B
  https://huggingface.co/Wan-AI/Wan2.1-VACE-14B; Lynx arXiv 2509.15496; Stand-In arXiv
  2508.07901; SkyReels-V3 arXiv 2601.17323; BindWeave https://huggingface.co/ByteDance/BindWeave;
  HunyuanCustom https://huggingface.co/tencent/HunyuanCustom; identity-preserving I2V arXiv
  2510.14255; fal model pages (Kling O1/O3, Vidu Q2, Veo 3.1, Seedance 2.0, MiniMax S2V,
  Qwen-Image-Edit-2511, Nano Banana Pro, Seedream, FLUX.2, LTX-2 trainer); Vertex AI Veo
  reference images doc; LTX IC-LoRA adapters
  https://docs.ltx.io/open-source-model/integration-tools/ic-lo-ra-adapters; Ingredients
  workflow https://www.floyo.ai/workflows/ltx-2-5-iclora-ingredients-for-refer-9zmb6zyv2p3a;
  HF API for `Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients`, `black-forest-labs/FLUX.2-klein-4B`
  and `Qwen/Qwen-Image-Edit-2511` (read 2026-11-15); InsightFace commercial licensing
  https://www.insightface.ai/services/models-commercial-licensing; FaceFusion guide
  https://docs.clore.ai/guides/face-and-identity/facefusion.
