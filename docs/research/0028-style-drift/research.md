---
id: 0028-style-drift
type: note
purpose: "Find why the Seinfeld S01E03 runpod-ltx25 reconstruction switches between photoreal and cartoon shot to shot, what predicts it, and a cheap A/B test of the fixes. No GPU spend."
scope: ["research", "decode", "prompt", "characters", "ltx-2.5", "quality"]
tags: ["research", "style-drift", "cartoon", "identity-blocks", "ltx-2.5", "seinfeld", "ab-test"]
related:
  - "research/0026-seinfeld-s01e03-first-end-to-end-run/research.md"
  - "research/0025-video-model-refresh-nov-2026/research.md"
  - "research/0021-training-data-contamination/research.md"
---

# Style drift: why some shots come out as cartoons

Vision alignment: Complete a movie or tv show Reconstruction (Star Wars IV headline run, with a control title)

Steady task `3a168ecf`. Follows research 0026, whose contact sheet showed 2 of 5 stills as
3D cartoons.

## Answer

**The cartoon shots come from one character's identity block, not from chance.** Every prompt
starts with the identity block for each character in the shot (up to 2), a fixed text taken from
`characters.json`. Kramer's block reads like a cartoon character sheet: "very thin, angular
build", "voluminous mop of dark, curly hair that stands up high", "sharp, pointed nose", "wide,
expressive mouth", "slightly hunched posture". **55 of the 62 shots that carry it (89%) render
as animation. Only 25 of the 237 shots without it do (11%).** No prompt says "live action" or
"photograph" anywhere, so nothing pulls the other way. [Supported: data/analyse_out.txt]

Fix, in order:

1. **Add a fixed style prefix** to every prompt (for example "Live-action 1990s American TV
   sitcom footage of real actors, photographed on 35mm film"), set per film.
2. **Rewrite identity blocks in photographic terms.** Change `IDENTITY_DESCRIPTION_SPEC` in
   `encode.py` so stage 3 stops writing caricature cues.
3. **Not negatives.** LTX-2.5 runs at `cfg 1.0`, so the negative prompt has no effect (see Fixes).

The A/B test below costs about $0.44 and checks fixes 1 and 2 on 21 clips. **It has not been
run; it needs Will's go.**

## Premise check

The task listed six things to check. Two of them do not vary in this run:

- **Conditioning keyframe: there isn't one.** The first part of every shot is pure text-to-video.
  `RunPodLtx25Strategy` never sees the source keyframes. The only image input is I2V chaining,
  where each later part of a split shot starts from the last frame of the part before it
  (`strategies_video.py`, `RunPodWanStrategy.generate`). So a shot's style is decided by the
  text alone, in its first part. [Direct: code]
- **Pod and model build: one of each.** The weights are pinned by sha256 (`PINNED_MODELS`).
  Clip write times show one unbroken session: the longest gap between clips across the 11.1 h
  is 152 s, and a new pod takes about 20 min to boot. The second pod from 0026's stall wrote no
  clips. Neither can explain drift here. [Supported: data/clip_mtimes.txt, data/analyse_out.txt]
- **Seed is the shot index** (`decode.py`: `seed=idx`, `+part_idx` per part). Stage 2 uses a
  fixed seed of 42. So seed and shot content always change together, and this data cannot
  separate them. The A/B test holds the seed constant to settle it. [Direct: code]

The prompts analysed here are rebuilt with the same `format_prompt` code. `strategies_video.py`,
`prompt_format.py` and `decode.py` match the run's workspace byte for byte. [Direct: diff]

## Method

1. **Stills.** One middle frame from each of the 427 clips (299 shots, shots 0-298; 69 shots are
   split into several parts), scaled to 640 px. Taken from `clips/runpod-ltx25/` in the run's
   output dir, read only.
2. **Classify.** `classify_style.py` asked `gemini-3.1-flash-lite` (Vertex backend, temperature
   0, JSON schema) to pick one of photoreal / 3d_animation / 2d_animation / other, plus a
   "looks like show X" field. 583k input tokens and 22k output tokens. **Cost about $0.18**,
   assuming list prices of $0.25 per million input tokens and $1.50 per million output tokens.
   Under the $0.30 limit. Output: `data/styles.jsonl`.
3. **Check the labels.** I looked through 36 random stills from each class
   (`sheet_photoreal.jpg`, `sheet_3d_animation.jpg`, `sheet_2d_animation.jpg`) and found no
   wrong labels. A simple frame statistic (variance of a Laplacian edge filter) agrees with the
   labels at AUC 0.79 (cartoon frames have hard line edges). Saturation and texture measures do
   not separate the classes at all (AUC 0.48). So a frame statistic alone would not have been
   good enough, and the vision labels are the ones used. [Supported: sheets,
   data/framestats_out.txt]
4. **Correlate.** `analyse.py` joins each clip's label with its shot's inputs: the rebuilt prompt,
   the characters in the shot, which identity blocks were used after the 2-character cap, seed,
   duration, number of parts, setting, dialogue, and pod session. Output: `data/clips.csv`.

## Results

Of 427 clips, 301 are photoreal, 93 are 3D cartoon and 33 are 2D cartoon. **Counting each
shot's first part only, 80 of 299 shots are cartoon (27%).**

### The strongest predictor: which identity blocks are in the prompt

First parts only:

| Identity blocks prepended | Shots | Cartoon | Rate |
|---|---|---|---|
| Kramer only | 45 | 38 | **0.84** |
| Kramer + Elaine | 14 | 14 | **1.00** |
| Kramer + George | 3 | 3 | **1.00** |
| Elaine + George | 33 | 13 | 0.39 |
| Elaine + Jerry | 33 | 6 | 0.18 |
| Jerry + George | 4 | 1 | 0.25 |
| George only | 49 | 4 | 0.08 |
| Elaine only | 34 | 1 | 0.03 |
| Jerry only | 74 | 0 | 0.00 |
| none | 10 | 0 | 0.00 |

**Checked against setting.** The comedy-club shots are all Jerry and all photoreal (0 of 34),
so they could be what keeps Jerry's rate low. Leaving the club out:

| Non-club shots | Shots | Cartoon | Rate |
|---|---|---|---|
| Kramer block | 62 | 55 | 0.89 |
| No Kramer block | 203 | 25 | 0.12 |
| No Kramer, one block | 126 | 5 | 0.04 |
| No Kramer, two blocks | 70 | 20 | 0.29 |
| Jerry block only | 46 | 0 | 0.00 |
| **Kramer in the shot, but his block dropped by the 2-character cap** | 15 | 3 | **0.20** |

The last row is the nearest thing to a natural experiment here. The 2-character cap ranks
characters by how often they appear in the film, so Kramer (77 shots) is dropped before the
other three (109-116). Shots where Kramer is in the scene but his block is not in the prompt
render cartoon at 20%. With the block, the rate is 89%. Since the scene is the same, **the
evidence points to the block text, not the scene, as the trigger.**
[Supported: data/analyse_out.txt "controlled cuts"]

**Second effect: two identity blocks together.** With no Kramer, a single block is 4% cartoon
and two blocks are 29%. Elaine + George is the worst pair (13 of 33). Elaine's block has the
same kind of wording as Kramer's ("voluminous, curly hair", "large dark eyes", "heart-shaped
face", "pointed chin"). On its own it stays photoreal (1 of 34). Paired with George's "round
face, thick-rimmed glasses, receding chin" it does not. Two face descriptions back to back
appear to read more like a character-design sheet. [Inferred: from the table; n is small]

### Words in the prompt

Smoothed log-odds per word, first parts. The top cartoon words are all from Kramer's block:
*lanky, angular, mop, hunched, very, distinctive, deep-set, tall, stands up*. Each appears in 62
prompts, 55 of them cartoon. After removing Kramer shots, the next tier is Elaine's and
George's block words (*heart-shaped, arched, delicate, chin, tense*). The top photoreal words
come from the comedy-club setting (*red, comedy, stage, club, spotlight*) and Jerry's block
(*pompadour, fair skin*). [Supported: data/analyse_out.txt, data/secondary_words_out.txt]

The scene descriptions themselves do not push towards animation:

- "animated" (as in "an animated conversation") appears in 36 prompts. Outside Kramer and the
  club it scores 3 of 25 (0.12), the same as the baseline.
- "comedic/lighthearted" moods score 7 of 60 (0.12), also the baseline.
- No prompt contains "live-action", "photograph", "photoreal" or "realistic".
- 108 prompts say "sitcom" or "TV". [Supported: data/style_words_out.txt]

### What doesn't predict it

| Input | Finding |
|---|---|
| Seed | Odd seeds 0.29, even seeds 0.25. Seed always changes with shot index, so this is not conclusive (see the A/B test). [Inferred] |
| Shot length | <2 s: 0.28, 2-4.84 s: 0.21, >4.84 s (split): 0.32. Flat. [Supported] |
| Character count | 2 characters: 0.52, 1: 0.21, 3+: 0.19. Explained by which blocks are used. With 3+ characters the cap often drops Kramer. [Supported] |
| Prompt length | Almost every prompt is over 110 words, so there is too little range to test. [Unknown] |
| Pod / model build | One pod session and one pinned build. Not a variable in this run. [Supported] |
| Dialogue | 0.27 with dialogue, 0.20 without (15 shots). [Supported] |
| I2V chaining | **Holds the style:** 0 of 69 split shots change style between parts. Chaining at strength 1.0 copies the first part's style forward, good or bad. [Supported] |

Consecutive shots switch style 84 times in 298 cuts. Random independent draws would switch about
117 times. The streaks follow scenes (Kramer scenes come in runs), not time or pod.

### The "looks like another show" effect

Gemini's lookalike field is noisy, but it groups by block. George's shots are tagged *The
Sopranos* 12 times out of 49. That matches the Sopranos-looking still in 0026, likely because
"balding, stocky, thick-rimmed glasses" points the model at a famous face. Kramer's shots are
tagged *Despicable Me* and *Ratatouille*. Kramer + Elaine shots are tagged *The Mitchells vs. the
Machines*. [Speculative: classifier free text; data/lookalike_out.txt]

## Fixes

| Fix | Expected effect | Cost | Verdict |
|---|---|---|---|
| **Style prefix.** One fixed sentence per film at the front of every prompt, before the identity blocks. It could live in shots.json `format` or a `--style` flag. | Gives a style anchor to the 100% of prompts that have none today | Free, about 20 tokens | **Do first** |
| **Photographic identity blocks.** Add to `IDENTITY_DESCRIPTION_SPEC`: "describe a real actor as a camera sees them; no caricature intensifiers (very, mop, stands up high, sharp/pointed, exaggerated)". Re-run stage 3, or hand-edit `characters.json` for this episode. | Removes the main trigger (89% to roughly the 4-12% baseline, if the cap row holds) | Stage 3 re-run, about $0.01 | **Do second**, and check Star Wars blocks too |
| Negative prompt ("cartoon, animation, 3D render") | **None.** Both stages use `LTXVDualCFGGuider` with `video_cfg 1.0`, and at cfg 1 the negative branch cancels out (`u + 1·(c−u) = c`). Raising cfg about doubles the cost per step and is not the distilled model's recipe. | about 2x stage-1 GPU | Don't |
| Fixed seed for every shot | Seed is not the driver. The style follows the text. A fixed seed might also correlate compositions between shots. | Free | Don't, unless the A/B control shows seed matters |
| Style anchor frame (I2V from a photoreal frame at low strength) | Chaining shows that I2V copies style reliably. But conditioning a new shot on another shot's frame also copies its layout, which breaks the cut. Lower strength is untested. | One more conditioning pass | Fallback only, if 1 and 2 fall short |
| Change chaining conditioning strength | Chained parts never drift (0 of 69). Nothing to fix. | n/a | Don't |

**Star Wars IV.** Its human blocks (`output/star_wars_iv_v2/characters.json`) are plainer than
Kramer's ("light brown hair cut short", "straight nose"), so the main trigger is weaker there.
The creature and droid blocks ("large, expressive photoreceptor eyes that glow", Jawa "glowing
orange eyes", Chewbacca's "wide mouth") are the ones to watch. The style prefix covers both.
[Inferred: blocks read, not rendered]

## Proposed A/B test (not run; needs Will's approval)

`ab_manifest.json` lists the exact 21 jobs (prompt, seed, frame count, baseline clip). It is
generated by `make_ab_manifest.py` using the real strategy code.

- **Shots:** six one-part shots that carry only Kramer's block and came out cartoon (79, 98,
  106, 152, 246, 256; 106 is 2D), plus two Elaine + George shots that came out cartoon (125,
  221). All are 1.1-1.8 s, so 25-41 frames each, to keep the cost down.
- **Baseline: free.** Every job reuses the original seed (= shot index) and frame count, so the
  existing clip is the baseline for its shot.
- **Arms:**
  - `control_rerender`, shot 98, prompt unchanged (1 clip). Checks that the same seed and prompt
    reproduce the cartoon. If it comes back photoreal, the seed is part of the story and the
    other arms need more than one seed each.
  - `A_prefix`: style prefix only. 6 Kramer shots + 2 Elaine + George shots (8 clips).
  - `B_photo_block`: Kramer block rewritten in photographic terms, no prefix. 6 clips.
  - `C_prefix_and_block`: both. 6 clips.
- **Cost:** 21 clips, about 24 GPU-min plus a 20 min boot on an A6000 at $0.59/h (0026's fitted
  rate), about **$0.44**. Run it under `tools/capped_run.py` with a $0.50 line, no kept pod.
- **Labels:** the same `classify_style.py` on the middle frame (under $0.01), plus Will's eye on
  a contact sheet.
- **Decision rule:**
  - If A turns at least 5 of 6 Kramer shots and both Elaine + George shots photoreal, ship the
    prefix on its own.
  - If only B or C does, ship the spec change as well.
  - If no arm gets past 3 of 6, the cause is deeper than the prompt text. Next step would be the
    low-strength style anchor, as its own test.
  - Also check that B/C still look like the same tall, curly-haired man across shots.
- **Running it needs a small change.** `decode.py` can't take per-job prompts today. Either add a
  style-prefix option (fix 1, needed anyway) and run decode on per-arm copies of the output dir
  with the selected shots and the edited `characters.json`, or write a 30-line runner over
  `RunPodLtx25Strategy._generate_one_clip`. This job did neither, to keep GPU and pod code out of
  an investigation.

## What we don't know

- Whether the style prefix is enough by itself. That is the main question the A/B answers.
- Whether seed contributes at all. Seed always changes with shot index in this data. The control
  re-render answers it for one shot.
- Whether the LTX-2.5 Star Wars pilot (0025) had cartoon shots. Searching 0022, 0023 and 0025
  for "cartoon", "pixar", "photoreal" and "style drift" found nothing, so it is either unmeasured
  or absent.
- Shots 299-314 were not generated when this was measured, so they are not in the data.
- The classifier was checked by eye on 108 of 427 stills (36 per class), not all of them.
  Borderline "stylised but photographic" frames could be under-counted as photoreal.

## Files

- `classify_style.py`, `analyse.py`, `make_ab_manifest.py`: rerunnable from the repo root. The
  usage line is in each docstring.
- `data/styles.jsonl`: label for each clip. `data/clips.csv`: each clip's inputs with the rebuilt
  prompt. `data/clip_mtimes.txt`: clip write times. `data/*_out.txt`: printed analysis output.
- `sheet_*.jpg`: 36 random stills per class (the label check).
- `ab_manifest.json`: the proposed test, job by job.
- `decisions.tsv`: decision trail.
