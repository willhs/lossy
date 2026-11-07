---
id: 0025-video-model-refresh-nov-2026
type: note
purpose: "Before any full run: survey video models released since Sep 2026, pilot the best against LTX-2.3 on a fixed 20-shot hard benchmark (cap $10), check the cheaper levers, and recommend the strategy, config and TV episode for the first full-length run."
scope: ["research", "decode", "model-selection", "cost", "pilot", "go-no-go"]
tags: ["research", "ltx-2.5", "ltx-2.3", "kandinsky-6", "minimax-h3", "benchmark", "cost", "runpod", "seinfeld"]
related:
  - "research/0025-video-model-refresh-nov-2026/survey.md"
  - "research/0024-full-run-checkpoint/research.md"
  - "research/0023-ltx2-dress-rehearsal/research.md"
  - "research/0022-ltx2-vs-wan22-trial/research.md"
  - "research/0021-training-data-contamination/research.md"
---

# Video model refresh, Nov 2026: pilot and recommendation

Vision alignment: On a new endeavour — Want to reach my potential with software

## Question

Steady task `72d1f878` (Will, 7 Nov). The full run keeps being deferred because it is expensive
and a disappointing result would mean paying for it twice. New models have shipped since lossy
last moved in September. Which setup gives the best output for the least money before the first
full-length run, and which TV episode should prove the pipeline end to end?

## Answer

- **Switch from LTX-2.3 to LTX-2.5 distilled** (open weights, same licence, ComfyUI-native).
  In a blind scoring of 20 hard shots it beat LTX-2.3 on every axis: **overall 2.70-2.80 vs
  2.05-2.10 (out of 5)**. It took **9 and 7 of 20 shot wins against LTX-2.3's 0**. Letterboxing
  fell from **6/20 shots to 1/20**.
- **Config: two-stage distilled + conv VAE.** It costs about what LTX-2.3 costs today: **~$15
  for the whole film**, ~$2.70 for a 22-minute episode, on a community A6000.
  - **Cheapest acceptable:** one-stage 640x352 then upscale, **~$8.70 a film**. Same scores at
    thumbnail size, visibly softer at full size.
- **Kandinsky 6.0 Pro** scored about the same as LTX-2.5 (2.85-2.95) and was best on droids and
  identity. But it costs **~3x LTX-2.5** (~$63 a film, over the whole goal ceiling) and renders
  only at 480p. Not recommended.
- **MiniMax H3** tops the open-weights leaderboard but was **excluded on licence**. It forbids
  displaying outputs in the US/EU/UK, which is where a YouTube/Reddit release would be seen.
- **Episode pick: Seinfeld S01E03 "The Robbery".** 22.1 min, 343 shots. It is on the NAS as a
  1080p BluRay and already partly encoded. Projected **~$4.50-7 all-in**, ~8 GPU-hours.
- **Pilot spend: $2.29 of the $10 cap.** Measured from the RunPod account balance. 0 pods
  remain.

Confidence tags: **[measured]** = observed in this pilot; **[sourced]** = dated external source
(see `survey.md`); **[inferred]** = my reasoning from the above.

## Premise check

- **Holds:** `runpod-ltx2` is LTX-2.3 22B distilled FP8 (`strategies_video.py:1291`), and
  defects 5+6 landed 2026-09-12 (`aac8ee2`). [measured]
- **Not in the notes: a third of the film has already been rendered.**
  - `output/star_wars_iv_v2/clips/runpod-ltx2/` in the main checkout holds 690 completed shots,
    690-1379, rendered 9-10 Aug.
  - That run stopped when the RunPod balance hit zero (the comment at `runpod_pod.py:63`).
  - It predates the defect 5+6 fix, so those clips carry the identity-bleed and
    droid-in-costume defects. [measured]
- **"Expensive" is relative.** The measured LTX-2.3 projection for the whole film is
  ~$16 on a community A6000 (0024 said $14-31). The real cost risk is a rerun, which is what this
  note tries to remove. [measured]

## Method

1. **Survey.** Two web-research passes: open weights, then hosted APIs + GPU prices. Every
   shortlisted repo was then verified against the Hugging Face API, and the H3 licence was read
   in full. See `survey.md`.
2. **Benchmark.** `tools/model_pilot/build_benchmark.py` picks 20 single-part shots (2-4.8s)
   evenly spread through the Star Wars IV encode, from 7 categories:
   - 3+ characters (×3)
   - droids and non-humanoids (×3)
   - wide/long shots (×3)
   - dark interiors (×3)
   - fast action (×3)
   - dialogue close-ups (×3)
   - the 0023/0021 named failures: Vader's costume and the Leia hologram (×2)

   Every prompt is composed by the production `RunPodLtx2Strategy.format_prompt` (identity
   mixin, 2-block cap, name stripping), so all models get byte-identical prompts. Reference
   frames are cut from the source film at each shot's midpoint.
3. **Renders.** All on one RTX A6000 48GB pod (secure cloud, $0.59/hr; community had no
   capacity).
   - **LTX-2.3:** through the real `decode.py`, which reconnected to the pilot pod via the shared
     state file.
   - **LTX-2.5:** ComfyUI v0.39.2, graph copied from the official blueprint.
   - **Kandinsky 6:** diffusers.
   - Seed = shot index everywhere.
4. **Scoring.** Two blind passes by a vision model on a fixed rubric: identity, prompt
   adherence, artefacts, motion and overall, each 1-5.
   - Gemini was the plan (`tools/model_pilot/score.py`, inline video). The Gemini project
     returned 429 on every model all afternoon, so it scored nothing.
   - **Fallback:** `blind_pack.py` renders each shot as one image: the original frame plus 4
     frames per candidate, labelled A/B/C(/D) in a seeded shuffle, every tile the same size. A
     fresh Claude subagent scored the pack with no access to the key. `unblind.py` maps letters
     back to models.
   - The second pass used a different shuffle and added the one-stage variant, to check
     position bias.
5. **Cost.** Wall-clock per shot per model is converted to **GPU-seconds per second of output**
   (`tools/model_pilot/project.py`). The first shot of each run is excluded as model load.
   Cost scales with film-seconds, not shot count (0024).

## Results

### Shortlist scores (blind; mean of 20 shots, 1-5)

| Model | Pass | Identity | Adherence | Artefacts | Motion | **Overall** | Wins |
|---|---|---|---|---|---|---|---|
| LTX-2.3 (current) | 1 | 2.25 | 2.70 | 2.30 | 2.65 | **2.10** | 0 |
| | 2 | 2.40 | 2.60 | 2.35 | 2.75 | **2.05** | 0 |
| **LTX-2.5** two-stage | 1 | 2.60 | 3.20 | 3.40 | 2.95 | **2.80** | 9 |
| | 2 | 2.55 | 3.35 | 3.45 | 3.20 | **2.70** | 7 |
| LTX-2.5 one-stage | 2 | 2.55 | 3.35 | 3.40 | 3.15 | **2.70** | 4 |
| Kandinsky 6.0 Pro distill | 1 | 3.30 | 3.05 | 3.70 | 2.65 | **2.95** | 11 |
| | 2 | 3.10 | 2.90 | 3.60 | 2.50 | **2.85** | 9 |

[measured: `evidence/blind_judge_claude*_unblinded.json`]

- **Ranking:** the order is the same across both shuffles, and LTX-2.3 won no shot in either.
- **Limit on the judge:** it is one model family, judging frame grids rather than playing
  video, so read differences under ~0.2 as noise.
- **Category split (pass 1, overall):**

| Category | LTX-2.3 | LTX-2.5 | K6 |
|---|---|---|---|
| droids / non-humanoids | 2.33 | 2.00 | **4.00** |
| dark interiors | 2.00 | **3.33** | 3.00 |
| fast action | 2.00 | **3.33** | 2.33 |
| known failures (Vader, hologram) | 2.50 | **3.50** | 2.50 |
| wide / long shots | 1.33 | 2.00 | **3.00** |
| 3+ characters | 2.00 | 2.67 | **3.00** |
| dialogue close-ups | 2.67 | **3.00** | 2.67 |

What the contact sheets show beyond the numbers [measured, `contact_sheet_1.jpg`, `contact_sheet_2.jpg`]:

- **Letterbox bars** (0024's defect): LTX-2.3 **6/20**, LTX-2.5 **1/20**, one-stage 2/20, K6
  2/20. Counted by `cropdetect` with the stitch's own 6-40% rule (`evidence/letterbox_counts.txt`).
  - LTX-2.5 mostly removes the problem that `stitch.py` currently crops away.
- **Kandinsky's droids are suspiciously good.** Its C-3PO and R2-D2 look close to the film's
  props, more like recall than reconstruction.
  - That is the contamination 0021 warns about. The prompt carries the "C-3PO:" / "R2-D2:"
    design labels the mixin deliberately keeps.
  - Its droid win is partly memory, not prompt-following.
- **Kandinsky fails space action.** Shots 1784 and 1890 (X-wing/TIE dogfights) came back as
  pilot close-ups.
- **LTX-2.5 follows the prompt less literally in places.** It rendered one sun in the
  "two suns" shot (405), where LTX-2.3 got two, but with ring artefacts.
- **Leia hologram (551):** LTX-2.5 gives the recognisable hair buns. None of the three renders
  it as a projection.

### Speed and cost (measured on an RTX A6000)

| Config | Output | Warm s/shot | **GPU-s per output-s** | Star Wars IV (124.7 min) | Episode (22.1 min) |
|---|---|---|---|---|---|
| LTX-2.3 1280x704 (current) | 1280x704, 25fps | 67.5 | **21.9** | $15.90 / 48 GPU-h | $2.85 |
| LTX-2.3 960x544 *(5 shots)* | 960x544 | 39.8 | 10.6 | $7.74 | $1.44 |
| LTX-2.5 two-stage, DiffVAE | 1280x704 + audio | 90.9 | **30.3** | $22.03 / 67 GPU-h | $3.90 |
| **LTX-2.5 two-stage, conv VAE** *(5 shots)* | 1280x704 + audio | 77.8 | **20.5** | **$14.90** / 45 GPU-h | **$2.67** |
| LTX-2.5 one-stage 640x352 | 640x352 + audio | 36.9 | 11.8 | $8.69 / 26 GPU-h | $1.59 |
| Kandinsky 6.0 Pro distill (fp8) | 864x480 + audio | 263.2 | **87.2** | $63.38 / 192 GPU-h | $11.25 |

- **Price basis:** $0.33/hr (A6000 community), plus 20 min setup per pod per 12 GPU-hours
  (`projections.json`).
- **Secure cloud:** at $0.59/hr, multiply by ~1.8. The recommended config is then ~$27 a film
  and ~$4.80 an episode.
- **Conv VAE:** saves ~32% over DiffVAE on a 5-shot timing sample. A full-size crop of shot 551
  shows no visible difference (`diffvae_vs_convvae_551.jpg`). [measured]
- **One-stage:** saves ~60%. It ties two-stage at judge-tile size but is visibly softer at full
  size after a Lanczos upscale (`onestage_vs_twostage_891.jpg`). [measured]
- **Kandinsky 6's speed is partly the A6000's fault** [inferred]: Ampere has no FP8 compute, so
  FP8 storage only saves memory. An H100 runs it about 3x faster but costs about 8x more per
  hour. Its cost per film does not come down.

## Cheaper levers

| Lever | Finding | Status |
|---|---|---|
| Resolution | Rendering half the pixels roughly halves cost for both LTX versions | [measured] |
| Step count | Distilled LTX is already 8 steps (+3 refine for 2.5). Nothing left to cut. | [sourced: model cards] |
| Distilled / quantised weights | Already used: LTX-2.3 FP8, LTX-2.5 int8-convrot, K6 distill. NVFP4 is Blackwell-only. | [measured] |
| VAE | Conv VAE: -32% for no visible loss | [measured, 5 shots] |
| Batching | ComfyUI runs one prompt at a time. Shots differ in length and prompt, so batching doesn't fit. Concurrent pods help wall time, not cost. | [inferred] |
| Spot vs on-demand | RunPod's API lists **no spot price for any of these GPU types** today | [measured: `evidence/runpod_gpu_prices.txt`] |
| Community vs secure cloud | Community A6000 was out of stock, so the pilot fell back to secure at $0.59. Plan budgets at the secure rate and treat community as a bonus. | [measured] |
| Cheaper GPU (RTX 4090, $0.34) | RunPod's own guide finds it the cheapest per LTX clip. LTX-2.5 int8 reportedly fits 24GB. 8 attempts got no 4090 with a public IP, so it is untested. | [sourced; not measured] |
| Hosted APIs | The cheapest credible option is ~$75-225 a film, before per-clip minimums | [sourced] |

## Recommendation for the first full-length run

**Strategy:** a new `runpod-ltx25` strategy. **Config:** LTX-2.5 22B distilled int8-convrot,
two-stage (640x352 → x2 latent upsample → 3-sigma refine), conv VAE, 1280x704, 25fps, CFG 1,
prompt enhancer off, identity mixin unchanged. Run on an A6000/A40 48GB, community first, then
secure.

Expected cost:

| | Community | Secure |
|---|---|---|
| Star Wars IV | ~$15 | ~$27 |
| Seinfeld S01E03 episode | ~$2.70 | ~$4.80 |

Both are well inside the $50-60 goal ceiling.

**Why:**
- It is the only option that is clearly better than today **and** no more expensive.
- It scored 2.7-2.8 vs 2.05-2.1, cut letterboxing about 6x, and produced cleaner artefacts and
  motion.
- It is the same family, licence and graph style, so the identity mixin, chaining and native
  audio design carry over. [measured + inferred]

**Work needed before the run** (not done here; filed as a follow-up):

1. **Implement `runpod-ltx25`.**
   - Port `tools/model_pilot/run_ltx25.py`'s graph into `strategies_video.py`.
   - Upgrade ComfyUI on the pod: the stock `runpod/comfyui:latest` image is v0.30.0, and LTX-2.5
     needs v0.32+ (v0.39.2 tested).
   - Add I2V chaining through `LTXVImgToVideo` into the stage-1 latent. **Untested on 2.5.**
2. **Weights access.**
   - `Lightricks/LTX-2.5` is gated, and accepting the gate means consenting to marketing. Will's
     account has not accepted it, so the official files return 403.
   - The pilot used the `comfyicu/LTX-2.5` mirror. Its transformer, VAEs and upscaler are
     **byte-identical** to the official LFS sha256s. The int8 text encoder differs by 2,412
     bytes, which looks like a header/metadata change.
   - Either Will accepts the gate, or the strategy pins the mirror by hash.
3. **Use LTX-2.5's native audio?** It generates synced audio for free. Worth an ear test against
   the MMAudio ambience path.
4. **Fix the cost reporting bug.** `RunPodSession` records the community rate when it falls back
   to secure cloud. This pilot's session reported $1.22 against a real $2.29.

**Not recommended:**
- **Kandinsky 6:** 3x the cost and 480p, for a score within noise of LTX-2.5.
- **MiniMax H3:** licence.
- **Any hosted API:** cost.
- **Reusing the 690 August LTX-2.3 clips:** they are pre-defect-fix and lower quality.

**If budget pressure returns:** use LTX-2.5 one-stage (~$9 a film community / ~$16 secure)
and accept softer frames.

## Episode pick

**Seinfeld S01E03 "The Robbery"**

| | |
|---|---|
| Length | 22.1 min (1325.6 s) |
| Shots | 343, mean 3.86s (Star Wars IV's mean is 3.62s) |
| Split shots | 72 over 4.84s, so I2V chaining gets exercised |
| Source | `/volume2/media/tv/Seinfeld (1989)/Season 1/Seinfeld.S01E03.1080p.BluRay.x264-BORDURE.mkv` on the NAS. `media/seinfeld_s01e03.mp4` is only a 480x360 copy. |
| Existing encode | `output/seinfeld_s01e03/` (Apr 2026): shots, descriptions and a 4-character registry, **but no dialogue and no `speakers.json`**. It predates the current describe prompt and `ENCODE_MODEL`. |

[measured]

**Why this episode:**
- It is closest to the "about 22 minutes" Will asked for. Curb S01E01 is 28.8 min.
  Better Call Saul S01E01 is ~53 min, and only a 10-min slice is encoded.
- The shot rhythm matches the film, so its cost and timing extrapolate to the film.
- It is a live-action dialogue show. That exercises the speech, voice-casting and
  speaker-attribution stages the film depends on, which no full run has proven yet.
- It avoids the film's hardest classes (droids, space battles). The first end-to-end run then
  tests the pipeline, not model limits.

**Projected cost and time** (recommended config):

| Item | Community | Secure |
|---|---|---|
| Video, 8.1 GPU-h | ~$2.70 | ~$4.80 |
| Fresh encode from the 1080p source (stages 1-4, Gemini) | ~$0.10 | ~$0.10 |
| Speech, ~300-400 lines at the 0023 rate | ~$0.70-0.90 | ~$0.70-0.90 |
| Ambience, 343 shots | ~$0.80 | ~$0.80 |
| **Total** | **~$4.50** | **~$7** |

**Wall clock:** ~8.5 h on one pod, or ~4.5 h split across two pods. [inferred from measured rates]

**Caveats:**
- Seinfeld is famous, so the 0021 contamination question applies. Faces are stripped by the
  identity path, but the show's look is well known.
- The Gemini project was rate-limited during this pilot. Confirm the quota before re-encoding.

## Spend

Ground truth is the RunPod account balance, read before and after (`evidence/spend_ledger.jsonl`).

| | |
|---|---|
| Start balance | $17.6501 |
| End balance | $15.3600 |
| **Pilot spend** | **$2.29** (A6000 secure, ~3.7 h incl. setup, downloads, 3 K6 OOM restarts and the lever sweep) |
| Gemini judge | $0 (every call 429'd) |
| Pods remaining | **0** (RunPod API, after `pod.py down` and the stray sweep) |

**Pod hygiene:**
- The self-destruct was re-armed (45 min) after every command.
- `pod.py down` terminated the pod and swept strays. `runpod.get_pods()` then returned `[]`.

## Gotchas found along the way

- **The encode's `keyframes/` dir is misnumbered** from about shot 405 on. It shows the wrong
  shot and predates the August re-encode.
  - `shots.json` itself is aligned: frames cut at `start_s`/`end_s` match the descriptions.
  - Anything that reads `keyframes/NNNN-*.jpg` for evaluation will compare against the wrong
    shot. [measured]
- **RunPod pods cap container RAM at 50GB** (cgroup `memory.max`) even when `free` shows 503GB.
  `enable_model_cpu_offload()` OOM-killed Kandinsky silently three times. [measured]
- **A process launched with `nohup … &` over SSH died with the session.** Use `setsid nohup …
  </dev/null`. [measured]
- **`set -x` in a pod script echoed the HF token into a log.** The log and the token file were
  deleted, and the pod is gone. Rotate the token as a precaution.

## Artifacts

- `survey.md`: model survey, licences, rankings, hosted prices, GPU prices.
- `contact_sheet_1.jpg`, `contact_sheet_2.jpg`: original frame, then 30%/70% frames for
  LTX-2.3, LTX-2.5, LTX-2.5 one-stage and K6, all 20 shots.
- `onestage_vs_twostage_891.jpg`, `diffvae_vs_convvae_551.jpg`: full-size crops.
- `decisions.tsv`: decision trail.
- `tools/model_pilot/`: the harness. Every number here can be regenerated from it plus the
  clips:

  | File | Role |
  |---|---|
  | `build_benchmark.py` | builds the 20-shot benchmark |
  | `pod.py` | pod + spend harness |
  | `run_ltx25.py`, `run_ltx23_variant.py`, `remote_k6.py` | renderers |
  | `score.py` | Gemini judge |
  | `blind_pack.py`, `unblind.py` | blind fallback judge |
  | `sheets.py` | contact sheets + comparison clip |
  | `project.py` | cost projections |

- Not committed (media):
  - Clips: `output/pilot0025/clips/{ltx23,ltx25-base,ltx25-onestage,ltx25-convvae,ltx23-960x544,k6}/`
  - Side-by-side comparison clip: `output/pilot0025/comparison.mp4`, copied to the job's
    `evidence/pilot0025_comparison.mp4`
