---
id: 0031-strategic-overview-after-sw4
type: note
purpose: "Ranked strategy memo for lossy after the 30 Nov Star Wars IV run: the most interesting direction, the biggest risk to the current approach, and seven experiments the research has not tried, each with evidence, smallest next step and rough cost."
scope: ["research", "strategy", "evaluation", "roadmap"]
non_goals: ["paid runs", "code changes", "Steady tasks"]
tags: ["research", "strategy", "evaluation", "rate-distortion", "contamination", "star-wars", "ltx-2.5"]
related:
  - "philosophy/vision.md"
  - "work/roadmap.md"
  - "research/0005-compression-analysis/research.md"
  - "research/0021-training-data-contamination/research.md"
  - "research/0025-video-model-refresh-nov-2026/research.md"
  - "research/0026-seinfeld-s01e03-first-end-to-end-run/research.md"
  - "research/0028-style-drift/research.md"
  - "research/0029-character-reference-continuity/research.md"
  - "research/0030-sw4-full-run-preflight/research.md"
---

# Where lossy goes after the Star Wars run: a ranked memo (21 Nov 2026)

Vision alignment: Have an online presence (goal: complete a movie or TV show reconstruction, Star
Wars IV headline run)

Steady task `705815b1`. This is a desk study: no paid runs, no code changes and no Steady tasks.

**What was read:**
- `docs/philosophy`, the roadmap and the git history;
- research 0025-0030 directly;
- research 0001-0024 and experiments 0001-0006 through two parallel readers, whose key claims I
  spot-checked;
- the pipeline code through one reader, also spot-checked;
- the public July write-up;
- one web pass for prior art.

The decision trail is in [`decisions.tsv`](decisions.tsv).

**Confidence tags:**
- **[Direct]**: a repo doc, the code, git, or arithmetic on stated figures says it.
- **[Supported]**: a dated external source or several converging pieces support it.
- **[Inferred]**: my reasoning from the evidence shown.
- **[Speculative]**: a guess, flagged as one.

## Premise check

- **Part of "beyond finishing one reconstruction" is already done.**
  - **Seinfeld S01E03** was rebuilt end to end on 8 Nov: all 315 shots, 23.0 min, about $10.13 in
    total. Nobody has watched it through or heard the voice casting (0026 "Result" and "Finish
    job": "nobody has listened to it yet"). [Direct]
  - **A public write-up already exists.** The "1mb movie" post at willhs.me/posts/1mb-movie
    (12 Jul 2026) has been linked from the README since `12964b5`. It covers Wan 2.1/2.2 scene
    reconstructions and the ~320 KB / ~330 bit/s figure. Its "further improvements" list
    (continuity, a first frame from an image model, per-character voices) is what the Aug-Nov work
    went on to build.
  - **The roadmap is stale:** it still lists "Write the blog post" under Later. [Direct: post,
    copy in the job's `evidence/`]
- **The 30 Nov run was NO-GO as it stood on 21 Nov** (0030, blockers B1-B4).
  - **Cleared since:** the keyframe cost. klein now runs on the pod, and the run totals
    $44.5-50.5.
  - **Still open, with no commit in `git log`:**
    - B1: budgie has no encode.
    - B2/B4: Will's calls on the spend cap and the style prefix.
    - F5/F6: from the 0026 review.
  - The RunPod balance was $0.388 after the last check (0029 "Self-hosted klein"). [Direct]
- **0005's "~1,400x less data rate" is a units slip.**
  - 632 MB over 7,484 s is 84 KB/s, not 710 KB/s.
  - So the rate ratio equals the size ratio: 162x raw, ~1,800x after xz.
  - The July post gets it right (~330 bit/s). [Direct: arithmetic]

## The three answers

**Most interesting direction: make lossy a measurement of what survives language.** The project
says its interest is "what survives the round-trip" (`design/identity.md`), but no number has ever
compared a reconstruction with the film. The chart worth making is fidelity against bits per
second of text, with AV1's floor on the same axis (items 1-3). It is the most shareable single
image the project could produce, and it costs about $5.

The web pass found no published text-only rate-distortion curve for video. Text-conditioned
codecs report single operating points:
- CMVC: "the bit consumption associated with motion text has not been calculated".
- GVSC: one "description only" point, on 8-frame 256 px clips.
- DiSCo (CVPR 2026 Findings): an LTX-Video decoder whose lowest point is 3.58 kbps, and that is
  text *plus* degraded video.

lossy's whole text stream runs at about a tenth of DiSCo's lowest rate. [Supported: web pass,
arXiv 2408.08093, 2502.13838, 2512.00408; this is an absence finding, see "What we don't know"]

As a thing to show people, the two-hour film is the proof, but the units that travel are a 30-60 s
side-by-side and one chart. [Inferred]

**Biggest risk: the project optimises what it can measure, never what it is about.** Every number
since August is one of:
- consistency across shots (face cosine, the 0029 judge);
- style (photoreal rate, 0028);
- seams and drift (0023/0024);
- spend.

None compares the output with the source. That causes two problems:

1. **Fixes that make output uniform score as wins, even when they move away from the film.**
   - Keyframe-first scores *above the real episode* on consistency: face cosine 0.55 against the
     source's 0.45 (0029, "Reference").
   - The photographic Kramer fix swapped a cartoon for a "generic 90s sitcom actor", labelled
     "Friends" on 5 of 12 clips (0028).
   - The July post already named the symptom: "it goes to the middle". [Direct]
2. **The 30 Nov run will end with two hours of output, an exact spend line, and no figure for how
   much of Star Wars survived.** It also has no answer to the first comment it will get: "the model
   already knows Star Wars". [Inferred]
   - 1,032 of 2,070 SW4 prompts carry a design name (Vader, C-3PO, R2-D2, Stormtrooper and so on)
     on purpose (0030 `decisions.tsv`; `strategies_video.py` `keep_name`).
   - Gemini rewrote the Leia hologram as Obi-Wan from memory (0021).
   - The cheap decoder-memory test 0021 proposed was never run.

Effort shows the same skew: about 2,550 lines of pod infrastructure (`runpod_pod.py`,
`pod_journal.py`, `pod_ssh.py`, `capped_run.py`) and 15 November commits on it, against zero
commits to `eval.py` since 9 Aug. eval.py's round-trip scorer has never produced a committed
report. [Direct: `wc -l`, `git log`]

**Runner-up risk, to the run itself: the headline forecast has come in low every time.**
- 0018 put overhead at "$1-5"; 0023 spent ~$9.70 of $12.30 on overhead and waste.
- 0025's headline ~$15 a film (community rate) became $35-41 in 0030 (0025's own secure-rate
  figure was $27).
- Seinfeld was forecast at $4.50-7 and cost $10.13.

So the overrun has run 1.4-2.7x against the forecast that was quoted, and the account sits under
$1. 0030 already prices SW4 from the measured Seinfeld fit, which is the fix. Budget its top-up
with the same margin. [Direct figures; Inferred ratio]

**Experiments the research has not tried.** Each was confirmed untried by grep across
`docs/` and the code:
- a reference-based fidelity metric;
- a detail-level (bit budget) sweep;
- 0021's name present/absent render;
- iterating the codec on its own output;
- LTX-2.5's native multi-shot mode;
- an ear test of LTX-2.5's native audio;
- more than one seed per shot (0028 ran a single control);
- any human scoring beyond one unblinded person on at most 15 shots.

The ranking below uses them.

## Ranked top 7

| # | Direction | Why, in one line | Smallest next step | Rough cost |
|---|---|---|---|---|
| 1 | Fidelity yardstick: N-way shot identification against the source | The missing instrument; everything below needs it | Score the existing Seinfeld 315 shots | $0 (local) to ~$0.30 (Gemini) |
| 2 | Rate-distortion curve of language | The project's own question as one chart; nobody has published one | 10 shots x 3 budgets | $0.75; full chart $5-8 |
| 3 | "Does it already know Star Wars?" decoder-memory bound | The first objection the SW4 release will get, and a showpiece in itself | 6 design shots, name vs no name | $0.30; full ~$1 |
| 4 | Decode scenes, not shots (LTX-2.5 native multi-shot) | Attacks continuity at its cause and may cut the fixed per-shot cost | Read the card; then one exchange as T, K and multi-shot | $0, then ~$1 |
| 5 | Generation loss: the codec run on its own output | Viral shape; untested for video; lossy already shows the attractor | 3 generations of a 10-shot scene | ~$0.80; 10 generations ~$2.50 |
| 6 | Edit the compressed film: alternate versions | A real codec can't do this; the tooling mostly exists | One scene, one style edit | ~$0.75 |
| 7 | "Name that film from 1 MB": audience in the loop | Fits the online-presence goal; gives human N-way scores | 10 existing clips as a gallery | $0 plus Will's time |

**How they are ranked:** decision value for the next month first, then technical interest, then
showability per dollar. Items 1-3 are a package: 1 is the instrument, 2 the chart, and 3 the
honesty footnote that keeps 2 credible on Star Wars.

### 1. A fidelity yardstick: can each reconstructed shot be matched to its source shot?

**Why.** It measures fidelity to the film, which nothing does today. It needs no rate-limited
service. It turns the SW4 report from "it finished" into "X% of shots identifiable, against 10%
chance".

**Evidence.**
- **The quality measures so far:**
  - blind scoring of frame grids (0025);
  - face cosine *across a character's shots* (0029);
  - photoreal rate (0028);
  - seam ratio and timeline drift (0023/0024);
  - naming rate (0021/0024).

  No CLIP/SigLIP, LPIPS, VMAF or output-vs-source face score exists in the code. [Direct: code
  reader, spot-checked `eval.py`]
- **`eval.py video` is built but unused.** It re-describes the output and compares it, field by
  field, with the input text. That is decoder adherence measured in text, not fidelity to pixels,
  and it has two confounds: a different model from the encoder, and a sound field guessed from
  frames. [Direct: `eval.py:94-315`, `config.py:73-74`]
- **The Gemini judge has failed on 429s twice:** in 0025 ("all afternoon") and in 0029's SW4
  check (every model). [Direct]
- **Precedent:**
  - fMRI-to-video decoding reports 2-way and 50-way top-1 against chance (MinD-Video, NeuroClips).
  - "Score a caption by regenerating from it" is established for images, and VidCapBench (2025)
    finds caption scores track the quality of the regenerated video.

  [Supported: web pass, abstracts]

**Design.**
- For each shot, embed one midpoint frame of the reconstruction and one of the source.
- Report top-1 among N = 10 candidates at three levels:
  - neighbours from the same scene: shot-level information, i.e. subject and composition;
  - random shots from the same film: scene-level;
  - frames from other films: film-level look.
- Chance is 10% at every level. The ceiling is a different frame of the same source shot.

My guess is near-ceiling at film level and near-chance at shot level; that split would itself be
the headline. [Speculative]

**Smallest next step.**
- Score Seinfeld S01E03's 315 existing shots.
- Cut the source frames fresh from the video at each shot's `start_s`/`end_s`. The SW4 encode's
  `keyframes/` is misnumbered from about shot 405 (0025 "Gotchas"). [Direct]

**Cost.**
- $0 compute on CPU with a local image-embedding model. That is a new dependency, which needs
  Will's OK under AGENTS.md "Guardrails".
- Or about $0.10-0.30 of Gemini per 300 shots for an N-way VLM judge, with the 429 risk.
- Either way, a ~150-line tool script. [Inferred]

### 2. The rate-distortion curve of language

**Why.** It answers "how many words is a shot worth?" with one chart, puts the codec joke on a
real axis, and fills a gap in the published work.

**Evidence.**
- **Never run.** "Different description detail levels" has sat on the roadmap's Later list
  (`work/roadmap.md`), and the schema has one fixed detail level with no budget control.
  [Direct: code reader, `encode.py:637-675`]
- **Today's operating point:** ~320-350 KB after xz for 124.7 min, i.e. about 330-375 bit/s
  (July post; 0005). [Direct]
- **The classical half of the chart already exists.** The July post measured AV1 at
  100/30/10 kbps, calling 10 kbps (~23 MB a film) "the codec floor". [Direct: post]
- **Prior art:** DiSCo's floor is 3.58 kbps with an LTX-Video decoder, and there is no text-only
  curve (see "The three answers"). [Supported]

**Design.**
- **Hold fixed:** encoder, decoder and seed, on one ~40-shot SW4 scene.
- **Budgets:** rewrite each existing description to 0 words (timing plus "a shot from a 1977
  science-fiction film"), 8, 32, 128, and full (~200). A Gemini rewrite of the existing text keeps
  the encoder's perception fixed and changes only the budget.
- **Measure:** bits per second after xz; decode; score with item 1.
- **Seeds:** 2-3 per budget, so the chart separates what the text loses from what the decoder
  randomises (0028 tested one seed per shot).
- **Baseline:** put AV1 encodes of the same scene at 10 and 30 kbps on the same axis, scored the
  same way.

**Smallest next step.** 10 shots x 3 budgets (8, 32, full) x 1 seed = 30 clips.

**Cost.** ~$0.75 for the first step. The full chart is ~200-300 clips, about $5-8 at the measured
~$0.024 a shot (0026). The rewrite pass costs cents. [Inferred from Direct rates]

### 3. "Does it already know Star Wars?": bound the decoder's memory

**Why.** A Star Wars reconstruction invites the objection "the 320 KB is a lookup key". The
current plan can't answer it, and the side-by-side ("Darth Vader" vs "a tall figure in black
armour") is a showpiece in itself.

**Evidence.**
- **0021, "Finding 1":**
  - World knowledge *replaced* the description: the female hologram was written as Obi-Wan and
    rendered as a Vader-like figure.
  - "Star Wars ... is the most contaminated plausible choice."
  - Its open question 3 ("Does a name in the prompt change the rendered face? ... Testable
    cheaply") was never run. [Direct]
- **Design names are kept on purpose.** `keep_name` keeps them because "stripping it turned a
  corridor of stormtroopers into generic soldiers in white" (`strategies_video.py:133-190`), and
  1,032 of 2,070 SW4 prompts carry one (0030 `decisions.tsv`, "1032 -> 0 shots"). [Direct]
- **The same leak is treated two ways.** 0029 refuses sheets for these designs as "contamination
  rendered as a picture" but keeps the name in the text. No doc reconciles the two. [Direct;
  inconsistency Inferred]
- **Memory shows up elsewhere too:**
  - Kandinsky's droids read "more like recall than reconstruction" (0025).
  - 0014's audio judge heard "a recognizable rendition of the Star Wars main theme" from MMAudio,
    which is never reconciled with 0019. [Direct]

**Design.**
- 12 design-character shots x 3 arms:
  - name kept (as today);
  - the registry description only;
  - "title only" (the 0-word point of item 2).
- Score with item 1 and present the arms side by side.
- Report it next to the SW4 result, with a less-famous control title, as 0021 recommends.
  Seinfeld is only partly that.

**Smallest next step.** 6 shots (Vader, C-3PO, stormtroopers, 2 each) x 2 arms = 12 clips.

**Cost.** ~$0.30 for the first step; ~$1 for all 36 clips.

### 4. Decode scenes, not shots: LTX-2.5's native multi-shot mode

**Why.** The July post's "every shot is from a different universe" comes from generating each shot
in isolation. LTX-2.5 lists native multi-shot generation, and nothing in the research has tried it.
It could also lower cost.

**Evidence.**
- **The capability is listed.** 0025's survey gives LTX-2.5's capabilities as "T2V, I2V,
  first/last frame, V2V, A2V; native multishot [verified: HF card]" (`survey.md:30`).
- **It was never tested.** A grep for multi-shot finds only that line and 0011's "Option 6 ...
  avoid for now", which assumed API generation. [Direct]
- **The claim is vendor-only.** Press coverage says LTX-2.5 holds "character, environment,
  lighting, and voice across the cuts", with no independent benchmark. [Supported: VentureBeat,
  11 Aug 2026, via the web pass]
- **Cost structure:**
  - The Seinfeld fit is ~46 s fixed + 16 s per video-second per shot (0026), and SW4's median shot
    is 2.34 s (0001). The fixed part is over half of a median shot's cost. [Direct]
  - One 15 s pass covering 4 shots would take ~286 s, against ~424 s for the 4 shots
    separately: about a third less. That holds only if the fixed cost is paid per generation.
    [Inferred; unmeasured]
- **It reaches shots keyframes don't.** Keyframe-first covers 795 of 2,070 SW4 shots. The other
  1,275 stay text (1,047 with design characters, 228 with no characters) (0030). Scene generation
  would also cover settings and designs. [Direct numbers; Inferred benefit]
- **The model picks the cut points.** The stitch would re-detect cuts with PySceneDetect, which is
  already in the pipeline, and retime each shot to the original. [Inferred]

**Smallest next step.**
- $0: read the LTX-2.5 card's multi-shot section and the ComfyUI v0.39.2 nodes for it.
- Then one Seinfeld exchange (4-6 shots) and one SW4 exchange, each rendered as text (T),
  keyframe-first (K) and multi-shot. Score with face cosine and item 1.

**Cost.** ~$1 for the test, plus a new graph builder if it works. VRAM at 1280x704 for ~375 frames
is unknown.

### 5. Generation loss: run the codec on its own output

**Why.** It is the most viral-shaped experiment available, and an open question for video: what
does a film collapse to?

**Evidence.**
- **Image loops collapse.** Hintze et al. (*Patterns*, Dec 2025) found 100 image describe-generate
  loops settled into 12 generic motifs, across 4 generators and 4 captioners. Telephone-game
  studies show drift that compounds after the first critical error (Mollah et al., 2025).
  [Supported: web pass]
- **The format already travels.** One repost of the ChatGPT "100x replica" image loop drew 9.1M
  views (Daily Dot, 2 May 2025). The web pass found no video loop. [Supported]
- **lossy already shows the attractor:**
  - "Friends" as the lookalike on 5 of 12 clips (0028).
  - At 1210-1330 s of the Seinfeld output, the stills "look like a different well-known drama"
    (0026).
  - The July post: "it goes to the middle". [Direct]

**Design.**
- Take one ~30-40 s scene.
- Encode the reconstruction with the same stage 2 and the original shot boundaries; the stitch
  keeps the cut list to ~2 frames (0026).
- Decode, then repeat.
- Track item 1's score against generation 0, and the drift of the text between generations.

**Smallest next step.** 3 generations of a 10-shot scene.

**Cost.** ~$0.80; 10 generations is ~$2.50. [Inferred from Direct rates]

### 6. Edit the compressed film: alternate versions from the text

**Why.** lossy's bitstream is a script, so it can be edited, and no real codec allows that. It is
showable, nearly free, and needs little new tooling.

**Evidence.**
- **Planned, never run.** 0010 listed genre presets, subject swaps, and alternate dialogue that
  reuses the video; none was run. [Direct]
- **The style lever exists and works.** `--style-prefix` is already built, and 0028 shows a prefix
  moves the whole look when no caricature block fights it: Elaine + George went from 0/2 to 2/2
  photoreal. [Direct]

**Design.** The garbage-masher or cantina scene in three versions:
- a one-line style edit (for example "1950s black-and-white B-movie", or "claymation");
- a subject swap by find-and-replace in `shots.json`;
- alternate dialogue (TTS only, video reused, about $0.05).

**Smallest next step.** One scene with one style edit, about 30 shots.

**Cost.** ~$0.75 for the first step; ~$2-3 for all three versions.

### 7. Put the audience in the loop: "name that film from its 1 MB reconstruction"

**Why.** The vision is about contributing on X and reddit, and a guessing game is a native format
there in a way a two-hour upload isn't. The guesses are also human N-way identification, the gold
standard behind item 1.

**Evidence.**
- **Human scoring so far is thin:** one unblinded person on at most 15 shots (exp 0003: 15;
  exp 0005: 5 fully scored). 0029's 16-shot trial was scored by a model judge. [Direct]
- **Material exists:** reconstructions from the July post (Matrix lobby, Moria, Lebowski, binary
  sunset) plus Seinfeld. [Direct]
- **Famous shots should be easier to name.** That is item 3's result, played as a game. [Inferred]
- **IP context:**
  - Disney sent ByteDance a cease-and-desist over Seedance 2.0 (Feb 2026).
  - Warner's automated takedown hit Blade Runner - Autoencoded (2016).

  Short clips with commentary are a safer format than a full-length upload. [Supported: web pass;
  the format call is Inferred]

**Smallest next step.** 10 existing clips as a reddit gallery or a static page.

**Cost.** $0 plus Will's time; $0-3 for extra clips.

## Before 30 Nov: three $0 moves

1. **Watch the Seinfeld reconstruction end to end and listen to the casting** (Will, about 25 min).
   It is the only finished artifact, and the cheapest test of whether SW4 should ship as two hours
   or as curated scenes. [HUMAN]
2. **Ear-test LTX-2.5's native audio on 10 Seinfeld clips against the MMAudio ambience.** Nobody
   has listened to it (0025 smoke test; 0025 recommendation, item 3). If it holds up, the ~$8
   ambience pass, about a sixth of the SW4 budget (0030), can be dropped.
3. **Keep what item 1 will need after the run:** the per-shot clips (not only the stitched film),
   the exact `keep_name` list used, and a note that source frames must be cut fresh, because the
   encode's keyframes are misnumbered.

## Ruled out, and why

- **Another model bake-off now:**
  - 0025 is two weeks old.
  - Kandinsky 6 is 3x the cost at 480p.
  - MiniMax H3's licence bars display in the US/EU/UK.
  - Wan 3.0's weights are closed (web pass).

  Revisit when an open model claims reference identity.
- **Hosted reference-to-video (Kling, Veo, Seedance 2):** 10-50x the film budget (0029 2a).
- **More text-side identity work:** identical tokens still sample different faces (0029 "Premise
  check"), and K already passed.
- **Ingredients (arm I):** only if long chained shots drift (0029 "Decision").
- **Per-character LoRA or face swap:** cost and licence (`inswapper_128` is non-commercial). It
  would also render likenesses of real actors, which 0029 argues against.
- **Pod reliability beyond F5/F6:** it already has the 2,550 lines and 15 November commits;
  measurement has had none. The returns are smaller there.
- **Music (0019's Bandit + MusicGen pilot):** a real gap, but small next to fidelity. The July
  post already says only the score's mood survives.

## What we don't know

- **How multi-shot works in practice.** I did not read how LTX-2.5's multi-shot mode is driven,
  or whether ComfyUI v0.39.2 exposes it. I only have the survey line and press coverage.
- **Whether shot-level identification works on generated footage.** An image embedding may sit at
  chance for every arm. That would itself be a finding, but it would flatten item 2's curve. A
  20-shot Seinfeld pilot settles it.
- **What the 46 s fixed cost is made of** (model load, text encoder, VAE, transfer). Item 4's
  saving depends on it.
- **Whether Will wants SW4 shown as two hours or as scenes.** That is a product call.
- **How complete the prior-art search was.** It was about 15 searches, so "no feature-length text
  round trip" and "no text-only curve" are absence findings. The closest precedents found:
  - Blade Runner - Autoencoded (2016, pixels);
  - Star Wars Uncut (2010, human remake);
  - Our T2 Remake (2024, reinterpretation).
- **Whether the July clips and the Seinfeld output still exist.** They are on the Mac, gitignored,
  and not visible from this workspace.
- **Total spend to date.** No doc tallies it. November's paid research comes to roughly $22,
  summed from the upper bounds in 0025-0029. [Inferred]

## Sources

- **Repo:**
  - research 0001, 0005, 0010, 0014, 0018, 0019, 0021, 0023-0030 and experiments 0005;
  - `docs/work/roadmap.md`, `docs/design/identity.md`, `README.md`;
  - the code (`eval.py`, `strategies_video.py`, `encode.py`, `prompt_format.py`, `keyframes.py`,
    `config.py`);
  - `git log`.
- **Public write-up:** "1mb movie", willhs.me/posts/1mb-movie (12 Jul 2026).
- **Web pass (21 Nov 2026):**
  - Text+Sketch: arXiv 2307.01944
  - Txt2Vid: 2106.14014
  - CMVC: 2408.08093
  - GVSC: 2502.13838
  - DiSCo: 2512.00408
  - Blau & Michaeli, ICML 2019
  - VidCapBench: 2502.12782
  - NeuroClips: 2410.19452
  - Hintze et al., *Patterns* 2025 (EurekAlert, 19 Dec 2025)
  - Mollah et al.: 2509.04438
  - Daily Dot, 2 May 2025
  - VentureBeat on LTX-2.5, 11 Aug 2026
  - Wikipedia: Star Wars Uncut, Seedance 2.0
  - Techdirt on Blade Runner - Autoencoded, 1 Jun 2016

---

5h quota (claude, `manager quota status`): before **0%**, after **32%** (7d: 67% before, 70% after).
The 5h figure covers the whole session, including four parallel reader subagents (about 590k
subagent tokens).
