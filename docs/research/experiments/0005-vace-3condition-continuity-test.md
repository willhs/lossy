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

<!-- Fill in after running -->

### Condition A — Baseline T2V (`runpod-wan`)

| Shot | Characters | Accuracy | Face Stability | Clothing | Notes |
|------|-----------|----------|----------------|---------|-------|
| ... | | | | | |

**Total**: —

### Condition B — Enriched T2V (`runpod-wan-enriched`)

| Shot | Characters | Accuracy | Face Stability | Clothing | Notes |
|------|-----------|----------|----------------|---------|-------|
| ... | | | | | |

**Total**: —

### Condition C — VACE (`runpod-vace`)

| Shot | Characters | Accuracy | Face Stability | Clothing | Notes |
|------|-----------|----------|----------------|---------|-------|
| ... | | | | | |

**Total**: —

## Interpretation

<!-- Fill in after evaluation -->

Key questions to answer:
1. **Does prompt enrichment help (B vs A)?** If B > A, locking canonical descriptions reduces drift even without reference images.
2. **Does VACE conditioning add further benefit (C vs B)?** If C ≈ B, the reference image conditioning at 1.3B scale isn't contributing; the prompt enrichment is doing all the work.
3. **Is VACE-1.3B quality sufficient overall?** Even if C > A, is the improvement meaningful enough to ship, or is TI2V-5B (Wan 2.2, fits 24GB, 720P) the right next step?

## Next Steps

<!-- Fill in after evaluation -->

If C ≈ B (VACE not adding value): VACE-1.3B's image conditioning is too weak at this model scale. Next candidate: **TI2V-5B** (Wan 2.2, unified T2V+I2V, 720P, fits 24GB — see research/0012).

If C > B (VACE adds value): VACE-1.3B works. Consider whether to ship as-is or invest in TI2V-5B for higher quality.

If B ≈ A (prompt enrichment not helping): Review character registry quality — check whether `characters.json` descriptions are specific enough and whether shot assignment is correct.
