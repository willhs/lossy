---
id: experiment-0004
type: experiment
purpose: "Initial test of VACE-1.3B reference conditioning for character continuity on Star Wars IV shots 0–19."
tags: ["experiment", "character-continuity", "vace", "wan", "runpod"]
related:
  - "../../research/0011-character-continuity/research.md"
  - "../../research/0012-wan-model-variants/research.md"
  - "../../tasks/0014-character-continuity-vace/plan.md"
---

# Experiment 0004: VACE Continuity Test (Shots 0–19)

## Hypothesis

Running `runpod-vace` with canonical character portraits and prompt identity enrichment will produce visually consistent characters across shots compared to the T2V-1.3B baseline (`runpod-wan`).

## Metrics & Method

- **Source**: Star Wars EP IV, shots 0–19 (first 20 shots)
- **Treatment**: `--strategy runpod-vace` with `characters.json`, portraits via fal.ai Flux Schnell, VACE reference conditioning
- **Control**: `--strategy runpod-wan` (T2V-1.3B baseline, same shots)
- **Evaluation**: Visual inspection via `tools/compare.html`
- **Cost**: ~$0.30 treatment group (RunPod pod, 24 GB)

## Results

- Encode stage 3 produced `characters.json` with named characters extracted from all shots
- Portraits generated and uploaded to pod successfully (no upload errors after fix in 2f42f5e)
- **Only shot 19 (of 20) used VACE reference conditioning** — shots 0–18 are the opening Star Destroyer/corridor chase with no named human characters; C-3PO and R2-D2 were the only characters assigned to this range
- Shot 19 comparison:
  - **VACE**: R2-D2 prominently in foreground, C-3PO visible; R2-D2 reference portrait appears to have influenced the framing
  - **T2V**: Stormtrooper-helmeted humanoid alongside R2-D2 — neither closely matched the source
- No OOM errors; generation quality comparable between strategies; concurrent audio worked

## Interpretation

The test range was poorly chosen for evaluating human character drift. Shots 0–18 are purely vehicle/spacecraft shots, so 19 of 20 clips fell through to the T2V fallback. The one VACE-conditioned shot (droids) is insufficient to assess whether the approach works for human characters.

No conclusions can be drawn about drift reduction. The pipeline mechanics are confirmed working (portrait upload, character mapping, VACE workflow, T2V fallback). The quality question remains open.

## Next Steps

Rerun from **shot 269** (Luke Skywalker's first appearance in the Cantina scene, Tatooine exterior) with 20–30 shots. This range features Luke, Han, Obi-Wan, and Leia — all multi-shot characters. A side-by-side comparison here will give real signal on whether VACE-1.3B reduces drift for human faces.

If VACE-1.3B shows insufficient quality improvement, the next candidate is **TI2V-5B (Wan 2.2)** — unified T2V+I2V, 720P, fits on 24 GB without pod changes (see research/0012).
