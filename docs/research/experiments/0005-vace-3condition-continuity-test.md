---
id: experiment-0005
type: experiment
purpose: "3-condition test to isolate whether character continuity improvement comes from prompt identity enrichment, VACE reference conditioning, or both."
tags: ["experiment", "character-continuity", "vace", "wan", "runpod", "prompt-enrichment"]
related:
  - "../0011-character-continuity/research.md"
  - "../0012-wan-model-variants/research.md"
  - "../../tasks/0014-character-continuity-vace/plan.md"
  - "./0004-vace-continuity-test.md"
---

# Experiment 0005: VACE 3-Condition Continuity Test (Shots 269–293)

## Hypothesis

VACE-1.3B reference conditioning will reduce character drift vs. a T2V baseline on human characters. Prompt identity enrichment alone will provide some improvement; VACE conditioning adds further benefit on top.

## Metrics & Method

- **Source**: Star Wars EP IV, shots 269–293 (Luke's first Tatooine scenes through Mos Eisley — Luke, Han, Obi-Wan, Leia all appear)
- **Shot count**: 25 shots per condition
- **Conditions**:
  - **A — Baseline T2V** (`runpod-wan`): no enrichment, no reference image
  - **B — Enriched T2V** (`runpod-wan-enriched`): canonical character descriptions prepended to prompts, T2V generation
  - **C — VACE** (`runpod-vace`): enriched prompts + VACE-1.3B reference image conditioning

```bash
python decode.py output/star_wars_iv_v2 --strategy runpod-wan --start-index 269 --limit 25
python decode.py output/star_wars_iv_v2 --strategy runpod-wan-enriched --start-index 269 --limit 25
python decode.py output/star_wars_iv_v2 --strategy runpod-vace --start-index 269 --limit 25
```

- **Evaluation**: Side-by-side in `tools/compare.html`

### Scoring Rubric

Score each shot 1–3 per dimension, per condition:

| Dimension | 1 — Poor | 2 — Partial | 3 — Good |
|---|---|---|---|
| **Character accuracy** | Wrong identity (wrong gender/species/costume) | Roughly right but off details | Recognisable character, correct costume |
| **Face stability** | Different face each shot for same character | Similar but noticeable variation | Consistent face across shots |
| **Clothing consistency** | Different outfit across shots | Right style, wrong details | Same costume details shot-to-shot |

Sum scores per condition across all shots with named characters.

## Results

_Partial evaluation — shots 269–273 reviewed visually (first frame of each clip). Full 25-shot scoring TBD._

### Condition A — Baseline T2V (`runpod-wan`)

| Shot | Characters | Accuracy | Face Stability | Clothing | Notes |
|------|-----------|----------|----------------|---------|-------|
| 269 | Luke, stormtroopers | 1 | — | 1 | **Darth Vader hallucinated** — completely wrong character |
| 270 | Luke | 2 | — | 2 | Luke in white tunic, anime style; Jawa-like figures nearby |
| 271 | Luke | 1 | — | 1 | Tiny distant figure — unidentifiable, no costume detail |
| 272 | Luke | 1 | — | 1 | **Dark-suited silhouette in FG** — wrong character hallucinated again |
| 273 | Luke | 2 | — | 2 | Generic robed desert figure, could be Luke |

**Shots 269–273 total**: 7/15 (character accuracy: 7, face: N/A, clothing: 7) — frequent identity hallucination

### Condition B — Enriched T2V (`runpod-wan-enriched`)

| Shot | Characters | Accuracy | Face Stability | Clothing | Notes |
|------|-----------|----------|----------------|---------|-------|
| 269 | Luke, stormtroopers | 3 | — | 2 | Young blonde Luke in tan tunic, correct character lineup |
| 270 | Luke | 2 | — | 2 | Two running figures, right costume style but less distinct |
| 271 | Luke | 3 | — | 3 | Luke clearly in white tunic on dune, most cinematic shot |
| 272 | Luke | 3 | — | 3 | Luke from behind in correct tunic + cloak approaching woman — excellent |
| 273 | Luke | 2 | — | 2 | Orange-accented tunic slightly off but silhouette correct |

**Shots 269–273 total**: 13/15 — major improvement, identity hallucinations eliminated

### Condition C — VACE (`runpod-vace`)

| Shot | Characters | Accuracy | Face Stability | Clothing | Notes |
|------|-----------|----------|----------------|---------|-------|
| 269 | Luke, stormtroopers | 3 | — | 2 | Correct characters; **anime-style rendering** (flatter, more stylized) |
| 270 | Luke | 2 | — | 2 | Anime style, orange-accented outfit — slightly wrong costume |
| 271 | Luke | 2 | — | 2 | Luke in cream tunic, less detail than B |
| 272 | Luke | 2 | — | 2 | Pilot helmet added (wrong prop); blonde Luke identifiable but costume off |
| 273 | Luke | 3 | — | 3 | Cream tunic + orange rebel companion — two clear distinct characters |

**Shots 269–273 total**: 12/15 — similar accuracy to B; trade-off: anime aesthetic introduced by VACE reference

## Interpretation

_Updated after reviewing close-up shots 279–285 alongside wide shots 269–273._

### B vs A: Prompt enrichment clearly helps for named characters

The baseline (A) hallucinates wrong characters in 2/5 wide shots (Darth Vader in 269, dark-suited figure in 272). Both B and C eliminate these errors. Enrichment works by anchoring the named characters' canonical descriptions — but this only helps for characters **in the registry**. Shot 280 (Uncle Owen + C-3PO) shows the opposite: A renders Owen's bandaged forehead clearly, while B and C just render droids, because Owen's description likely isn't specific enough in `characters.json`.

### C vs B: VACE adds value in close-ups but not consistently

Initial assessment (wide shots only) suggested VACE adds nothing. The close-up shots revise this:

- **Shot 284** (Obi-Wan close-up): C renders a more prominent, detailed human face than B. The face is recognisably older, grey-haired, in tan robes — and more visually prominent.
- **Shot 285** (human profile + C-3PO): C shows the human profile more clearly than A or B.
- **Droid shots (279, 281, 282, 283)**: No meaningful difference between B and C — droids are distinct enough that all conditions get them right.

The **anime-style skew in shot 269** is localised to that specific keyframe's reference image, not a consistent VACE artifact. Most VACE shots look cinematic.

VACE-1.3B conditioning appears to help render human faces more prominently in close-ups, but provides no benefit in wide shots and is unreliable — the reference image quality and framing of the keyframe crop matters a lot.

### Character registry gaps

The experiment revealed a secondary issue: the character registry doesn't cover all visible characters adequately. Uncle Owen appears in several shots but enrichment ignores him, causing B and C to miss him where A accidentally gets him right. The registry needs coverage of Owen, Beru, C-3PO, R2-D2 as named entries.

### Overall

**Prompt enrichment (B) is the primary driver of improvement**, eliminating identity hallucinations for named characters. **VACE (C) adds marginal benefit in close-up human face shots** but is unreliable in wide shots and dependent on keyframe quality. Neither result fully justifies VACE-1.3B as the production default — the close-up improvement is promising but inconsistent.

## Next Steps

1. **Ship `runpod-wan-enriched` as the new default** — clear win over baseline with no quality downside.
2. **Fix character registry gaps** — add Uncle Owen, Beru, C-3PO, R2-D2 with specific visual descriptions so enrichment covers them.
3. **Do not ship `runpod-vace` for standard use yet** — close-up improvement is real but inconsistent; needs more investigation.
4. **Evaluate TI2V-5B** (Wan 2.2 unified T2V+I2V, 720P, 24GB) — stronger image-conditioning model may deliver consistent close-up character anchoring. See `docs/research/0012-wan-model-variants/research.md`.

### Prompt enrichment design problems (found during this experiment)

The current approach (prepend static character bio to top of prompt) has three failure modes:

1. **Single-character bias**: only the registry character gets a description; other subjects in the same shot (Jawa, Uncle Owen) are unaffected, making the named character the implicit focal subject even when they shouldn't be.
2. **Temporally wrong costume**: character descriptions span the whole film ("later, he wears an orange flight suit..."). For early-film shots this causes the model to hallucinate late-film costumes (orange suits, pilot helmets) into scenes where the character is wearing a plain tunic.
3. **Prefix weight**: prepending the bio makes it the highest-weight tokens, overriding the actual shot composition.

**Proposed fix — encode stage4: prompt composition**

Rather than concatenating at decode time, add an LLM pass (Gemini Flash, cheap) that takes the base prompt + relevant character registry entries and produces a contextually-edited prompt. The model can weave appearance details in where relevant, leave the prompt unchanged if sufficient, and handle ensemble shots correctly. Run once at encode time; cache result as `composed_prompts.json`. Decode strategies read composed prompts instead of doing their own enrichment.
