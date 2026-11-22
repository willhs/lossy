---
id: 0033-fidelity-yardstick-seinfeld
type: note
purpose: "Build and run research 0031 item 1, the fidelity yardstick: can each reconstructed Seinfeld S01E03 shot be matched back to its source shot among 10 candidates, at shot, scene and film level, with a local image-embedding model?"
scope: ["research", "evaluation", "fidelity", "seinfeld"]
non_goals: ["paid runs", "Gemini judge", "changing the decoder"]
tags: ["research", "evaluation", "fidelity", "siglip", "n-way", "seinfeld", "ltx-2.5"]
related:
  - "research/0031-strategic-overview-after-sw4/research.md"
  - "research/0026-seinfeld-s01e03-first-end-to-end-run/research.md"
  - "research/0028-style-drift/research.md"
  - "research/0029-character-reference-continuity/research.md"
---

# Fidelity yardstick: Seinfeld S01E03

Steady task `51077770`. Will approved it on 22 Nov 2026, including the new local embedding
dependency. Spec: research 0031 item 1. Spend: $0 (local CPU/MPS, no RunPod, no Gemini).

## Answer

**Shown one midpoint frame of a reconstructed shot and 10 source frames, a SigLIP2 embedding
picks the right one 35% of the time when the other 9 are its neighbouring shots (chance 10%,
ceiling 81%).** Against random shots from the same episode it picks the right one 59% of the time
(ceiling 95%). Against frames from 12 other films it gets 67% (ceiling 99%).

| Level (N = 10, chance 10%) | Distractors | Top-1 | 95% CI | Ceiling |
|---|---|---|---|---|
| **Shot** | the 9 nearest shots in time | **34.9%** | 29.9-40.3% | 80.6% |
| Scene | 9 random shots of the same episode | 58.8% | 53.3-64.1% | 94.9% |
| Film | 9 random frames from 12 other titles | 67.4% | 62.0-72.3% | 99.3% |
| All 315 at once (chance 0.3%) | every other shot | 9.8% top-1, 28.6% top-5, median rank 13 | | 75.9% top-1 |

315 shots, all of them, scored against the 1080p BluRay the 0026 run encoded.
[Direct: `results.json`; the CI is a Wilson interval over shots.]

How to read it:
- **Two-fifths of the way from chance to ceiling at shot level.** The text keeps real
  shot-specific information: (34.9 - 10) / (80.6 - 10) = 35%. [Direct, arithmetic]
- **Scene level is about two-thirds of the way:** (58.8 - 10) / (94.9 - 10) = 57%. [Direct]
- **The film level is the surprise.** One reconstructed shot in three sits closer to some
  other film's frame than to its own source. That fits 0026/0028's style drift, where some
  shots come out as 3D cartoon or another show's look, and a cartoon frame lands near SpongeBob.
  This was not checked per shot. [Speculative]
- **0031 guessed "near-ceiling at film level and near-chance at shot level".** Both halves were
  wrong. Shot level is well above chance, and film level is well below ceiling. [Direct]

This gives the SW4 report its line: "X% of shots identifiable against 10% chance". For Seinfeld
S01E03 with runpod-ltx25 text-only decoding, X = 35% at shot level.

## What was measured, exactly

`tools/fidelity.py` (about 190 lines, `uv run --extra fidelity python tools/fidelity.py ...`):

1. **Shots:** the source's own cuts, found by the same detector `encode.py stage1 -d content`
   uses (scenedetect `ContentDetector`, threshold 27). On the BluRay this reproduces 0026's
   **315 shots** exactly, so `keyframes/` and `shots.json` numbering are not involved. [Direct]
2. **Frames:** for each shot, the reconstruction's midpoint frame, the source's midpoint frame
   (the truth), and the source's quarter-point frame (the ceiling probe). All are cut fresh with
   OpenCV and centre-cropped to a square, because the reconstruction is 1280x704 and other
   sources vary. Film-level distractors are about 400 frames sampled from 12 other titles in
   `media/` (300, Apollo 13, Better Call Saul, The Big Lebowski, Curb, Harry Potter, LOTR Moria,
   Planet Earth, SpongeBob, Star Wars IV, The Matrix, Phantom Menace). [Direct]
3. **Scoring:** a shot is a hit when cos(recon, true source) beats all 9 distractors. Shot level
   uses one fixed set, the 9 nearest shots by index, standing in for "same scene" because these
   shots have no scene labels. Scene and film level average over 50 random draws per shot,
   seed 0. The ceiling reruns every level with the quarter-point source frame in place of the
   reconstruction. [Direct]
4. **Time map:** identity. The stitched reconstruction keeps the source's shot timing (0026:
   1382.48 s vs 1382.43 s, largest drift 0.087 s). `--scan` checks this independently: mean
   recon-vs-source cosine over offsets of -15 to +15 s peaks at exactly **b = 0** (0.700, against
   a median of 0.667), and the PAL rate 25/23.976 gives no peak. [Direct: `run.log`, scan JSON in
   the evidence dir]

### Model: SigLIP2 SO400M/14, pinned

`hf-hub:timm/ViT-SO400M-14-SigLIP2` through `open_clip_torch` 4.2.2 and `torch` 2.15.0. It is the
optional uv extra `fidelity` in `pyproject.toml`/`uv.lock`, so the default install does not pull
in torch. HF snapshot `f9e2bfdc4807cd0fbadf67301e45193d6f330c20`.

Why this model:
- It is the strongest open image embedder that runs on the Mac's MPS. All 315 x 3 frames plus
  400 distractors take a few minutes.
- It is semantic. It scores subject, setting, composition and look, not pixels, and those are
  what a text-mediated codec can carry. Pixel metrics (LPIPS, SSIM) would score every shot near
  zero.
- The weights were already in the local HF cache.

DINOv2 (also cached) is the obvious second opinion and was not run. [Direct]

### Best and worst shots

`contact_sheet.jpg` (135 KB): source | reconstruction, the 10 best shot-level margins on the
left and the 10 worst on the right. Margin = cos(true) minus the best neighbour's cos.

- **Best:** establishing exteriors and title cards. Brick apartment fronts (#14, #142, #168),
  the closing lighthouse (#312), the "Fred Barron Productions" card (#311), plus a few George
  medium shots (#44, #66, #166). Places and graphics survive the text, and so does a bald man in
  glasses.
- **Worst:** 4 of the 10 have a black source midpoint (a fade): #0, #13, #167, #305. Excluding
  all 12 dark midpoints (mean luma < 12) moves shot level only from 34.9% to 35.6%, so the
  headline does not depend on them. [Direct: `dark.json` in the evidence dir]
- **The real failures** are the end-card logo, where Castle Rock became a generic sunset (#313),
  and character shots whose reconstruction is a different person in a different set (#114,
  #147, #224, #265).

### Text-only vs keyframe

This breakdown is moot for this episode. All 315 shots of the 0026 run are runpod-ltx25
text-only. The only Seinfeld keyframe clips are 0029's 16-shot trial, and they are not on this
Mac. They lived in that job's workspace. Run the tool on the SW4 full render, which mixes routes,
to get the split. [Direct]

## Gotcha: `media/seinfeld_s01e03.mp4` is the wrong episode

The local `media/seinfeld_s01e03.mp4` (480x360, 25 fps, 1325.6 s) is **The Stake Out**, not
S01E03 The Robbery:
- Its dialogue, from a local faster-whisper transcript, includes "Lonnie Anderson" and "the
  whole elevator business".
- The reconstruction's dialogue includes "master packer" and "where's the TV, where's the VCR".
- Only 8 word 4-grams that occur once in each transcript are shared, out of about 2,700 words.

[Direct: evidence `fidelity_wrong_episode/transcripts.json`]

The April 343-shot encode in `output/seinfeld_s01e03/` was therefore made from The Stake Out
under the S01E03 name. Anything that pairs that directory with the 0026 reconstruction is
comparing two episodes. The real S01E03 is on the NAS:
`/mnt/nas-media2/tv/Seinfeld (1989)/Season 1/Seinfeld.S01E03.1080p.BluRay.x264-BORDURE.mkv`
on budgie.

**That mistake made a useful negative control.** Scored against the wrong episode, with a PAL
time map, the yardstick gives:

| Level | Top-1 |
|---|---|
| Shot | 9.7% |
| Scene | 11.7% |
| Film | 34% |

Shot and scene level are at chance, as they should be. Film level is above chance because the
source is still Seinfeld. The offset scan for that run was flat. So the instrument does not
invent fidelity, and a flat `--scan` is a cheap tell that the source is wrong.
[Direct: `results_wrong_episode.json`]

## How the alignment was settled (ruled out)

- **Matching cuts between the two videos:** ruled out. Reconstructed clips carry their own
  internal cuts, so the best fit matched about 45 of 340 cuts, near chance.
- **Cross-correlating audio envelopes:** ruled out. The reconstructed audio is synthetic and
  correlated at r ≤ 0.07.
- **Comparing dialogue transcripts:** this is what exposed the wrong episode.
- **Using the BluRay:** this is what settled it. The detector reproduces 315 shots and the scan
  peaks at offset 0.

Full trail: `decisions.tsv`.

## Reproduce

```bash
uv sync --extra fidelity
uv run --extra fidelity python tools/fidelity.py <S01E03 BluRay .mkv> \
  output/seinfeld_s01e03/seinfeld_s01e03_reconstructed_runpod-ltx25.mp4 \
  --others media/300.mp4 media/apollo13_houston.mp4 ... -o /tmp/fidelity --scan
```

Scratch frames, the 3.1 GB BluRay copy and the per-run evidence stay out of git. `results.json`
(per-shot ranks and margins) and `contact_sheet.jpg` are committed here.

## Follow-ups

- Rename or replace `media/seinfeld_s01e03.mp4`, which is The Stake Out, and mark the April
  `output/seinfeld_s01e03/shots.json` as that episode. This needs the primary checkout, so it
  is Will's or an attended job's to do.
- Score the SW4 full render with the same tool: the "X% identifiable" line, plus the
  text-vs-keyframe split.
- Per-shot film-level misses against 0028's photoreal classifier, to test the style-drift
  explanation above.
