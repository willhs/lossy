---
id: 0025-video-model-refresh-nov-2026-survey
type: note
purpose: "Survey of video generation models released or updated Jun-Nov 2026 (focus Sep-Nov), against lossy's current LTX-2.3 setup: open weights, licence, rankings, identity/I2V, duration, speed, cost per film."
scope: ["research", "decode", "model-selection", "survey"]
tags: ["research", "survey", "ltx-2.5", "minimax-h3", "kandinsky-6", "wan-3", "hosted-api", "runpod-pricing"]
related:
  - "research/0025-video-model-refresh-nov-2026/research.md"
  - "research/0022-ltx2-vs-wan22-trial/research.md"
  - "research/0023-ltx2-dress-rehearsal/research.md"
---

# Survey: video models since Sep 2026 (supporting note for 0025)

Gathered 2026-11-07. Two web-research passes (open weights; hosted + GPU pricing), then every
shortlisted repo was checked directly against the Hugging Face API, and the MiniMax H3 licence was
read in full. Confidence tags: **[verified]** = checked first-hand against the primary source
today; **[sourced]** = one dated secondary source, not re-checked; **[unverified]** = claimed, could
not confirm.

**Baseline being beaten:** `runpod-ltx2` = **LTX-2.3 22B distilled FP8** in ComfyUI, 1280x704,
25fps, 8 steps, I2V chaining, A6000 48GB [verified: `strategies_video.py:1291`]. On the Artificial
Analysis (AA) arena, LTX-2.3 Fast sits at **838 Elo text-to-video / 832 image-to-video** — near the
bottom of both boards [sourced: AA leaderboards, fetched 2026-11-07, page undated].

## Open-weights models (self-hostable on RunPod)

| Model (release) | Licence | Size / VRAM | I2V / identity | Duration | AA Elo T2V / I2V | Speed reported | Verdict |
|---|---|---|---|---|---|---|---|
| **LTX-2.5** distilled (11 Aug 2026; HF card updated 11 Oct) | LTX-2.x Community: free under $10M revenue [verified: HF card] | 22B DiT + Gemma-4 12B TE; bf16, ComfyUI **int8-convrot**, NVFP4 (Blackwell) [verified: HF file list] | T2V, I2V, first/last frame, V2V, A2V; native multishot; no reference-image identity [verified: HF card] | `frames % 8 == 1`; optional duration head [verified] | **942 / 912** (+104 / +80 over 2.3 Fast) [sourced: AA] | RTX 5090 int8: 1280x704x121f in 28.7s [sourced: ai-muninn 2026-09-11] | **Shortlist (drop-in)** |
| **MiniMax H3** (weights 3 Aug 2026) | Custom; **grant excludes US, EU, UK, South Korea**; outputs may not be displayed outside the territory; >$20M needs auth; must display "MiniMax H3" in commercial UI [verified: LICENSE lines 10, 37, 42, 65] | 33B dense, 61.7GB bf16 transformer + separate TE; FL2VA and Ref2VA checkpoints [verified: HF file list] | Ref2VA: up to 9 reference images [sourced: HF card via comfyui-wiki] — the only open model with real multi-reference identity | 4-15s, 768p, 24fps [sourced] | **1160 / 1133** — #1 open, top-6 overall [sourced: AA] | 3-7x slower than LTX-2.5 on a 5090 [sourced: ai-muninn, Nereo 2026-08-19 (vendor)] | **Excluded: licence** (below) |
| **Kandinsky 6.0 Pro** distill (9 Sep 2026) | **MIT** [verified: HF card] | 29B Pro / 3B Lite; ~81GB repo incl. Qwen2.5-VL TE [verified: HF API] | T2AV + TI2AV (first frame); no reference identity [verified: card] | trained on 5s; pipeline accepts other lengths (rounds `num_frames`) [verified: diffusers source on pod] | not on AA yet | 4090 SD 5s: 235s; H100: 88s [sourced: official README via survey] | **Shortlist (clean licence, native audio)** |
| Cosmos 3 (NVIDIA, Jun 2026) | OpenMDW-1.1 [sourced] | Nano 16B needs ~80GB; Super 64B needs H200 [sourced: Spheron] | I2V | — | not on current board | ~40 H200-min per output minute at 720p [sourced] | Too slow/expensive |
| MAGI-2 Preview (Sand.ai, 5 Aug 2026) | Apache-2.0? [unverified] | 114B MoE, ~307GB, 8x Hopper [sourced] | — | — | 1040 (no open badge on AA) | — | Impractical |
| SkyReels V3 R2V (Jan 2026) | [unverified] | 14B | 1-4 reference images | — | not ranked | — | Older than window; licence unknown |
| HunyuanVideo 1.5 (Nov 2025) | Tencent Community | 8.3B | I2V | — | — | vendor: 480p I2V <75s on 4090 | No 2.x release found |
| Wan 2.5 / 2.6 / 2.7 / **3.0** | **API only — no open weights** [sourced: Morphic, Atlas Cloud] | — | — | Wan 3.0 up to 30s | Wan 3.0 **1172** (#1 overall) | — | Not self-hostable; Wan 2.2 (Jul 2025) remains the last open Wan |
| Ovi 1.1, LongCat-Video, Mochi/Step-Video successors | — | — | — | — | — | — | Nothing new in the window |

Native synced audio: LTX-2.5, MiniMax H3, Kandinsky 6, MAGI-2, Ovi. (LTX-2.3 also generates it;
lossy currently discards it — see the LTX-2 native-audio memory note.)

### Why MiniMax H3 is excluded despite topping the open board

The licence grants rights "solely within the Applicable Territory" (worldwide minus the EU, UK,
South Korea and the USA) and separately forbids using, distributing **or displaying** "any of their
Outputs" outside it (s.V.4, and Exhibit A item 1). The goal ends with an unlisted YouTube upload and
a reply on an r/StableDiffusion thread — an audience that is mostly in the excluded territories.
Running the model from NZ on a non-US pod would be fine for evaluation, but the published film
would not be. On top of that it is 3-7x slower than LTX-2.5 [sourced], which puts a full film at
roughly $50-100 on A6000-class hardware — at or over the whole goal ceiling. It is noted as the
quality ceiling, not tested.

## Hosted APIs

No hosted API comes close to self-hosting on cost. Cheapest credible options for 125 min of film
(raw seconds x list price; many APIs also bill a 5-6s minimum per clip, which adds 40-65% at a
3.6s mean shot):

| Model | Price / s | 125 min raw | Notes |
|---|---|---|---|
| Creatify Boreal (Sep 2026) | $0.01 (720p, fal) | ~$75 | AA T2V 828 — *below* LTX-2.3 |
| Pruna P-Video-2 draft | $0.015 | ~$113 | 1-20s clips, no minimum |
| HeyGen Video 1 / Grok Imagine 1.5 Lite | $0.02 (480-768p) | ~$150 | HeyGen fal endpoint is text-only |
| MiniMax H3 Max Turbo (hosted) | $0.04 ($0.028 promo to 15 Nov) | ~$210-300 | same licence question applies to hosted outputs? [unverified] |
| Veo 3.1 Lite / Sora 2 batch | $0.05 | ~$375 | |
| LTX-2.5 Fast (Lightricks API) | $0.09 | ~$675 | self-hosting the same weights is ~30x cheaper |

[sourced: fal pricing via teamday.ai 2026-11-02, Pruna docs, invideo 2026-08-06; individual prices
disagree across aggregators — treat as ±50%.] The cheapest hosted route is 3-10x the self-hosted
LTX projection and above the $50-60 goal ceiling, so none went into the pilot.

## GPU pricing (RunPod, on-demand, 2026-11-07)

| GPU | Community | Secure |
|---|---|---|
| RTX A6000 48GB | $0.33 | $0.59 |
| A40 48GB | $0.35 | $0.59 |
| RTX 4090 24GB | $0.34 | $0.89 |
| RTX 5090 32GB | $0.69 | $1.19 |
| L40S 48GB | $0.79 | $1.09 |
| RTX PRO 6000 96GB | $1.69 | $2.49 |
| H100 SXM | $2.69 | $3.99 |

[sourced: runpod.io/pricing via survey agent; RunPod raised several tiers in Aug-Sep 2026.] Spot /
interruptible prices are not published on the page; third parties report 30-75% off [unverified].
RunPod's own Oct 2026 guide measured LTX-2.3 distilled (768x512, 97 frames, 8 steps) at
$0.0033/clip on both A6000 and 4090 — the cheapest two cards per clip [sourced: runpod.io guide
2026-10-12].

## What could not be verified

- No published A6000/L40S timings for any new model — hence the pilot.
- LTX-2.5 official max duration/fps (ltx.io would not load for the researcher).
- Kandinsky 6 has no independent ranking; its "preferred over LTX-2.5" claim is the authors' own.
- Whether H3's territory clause binds on GPU location, user location or viewer location.
