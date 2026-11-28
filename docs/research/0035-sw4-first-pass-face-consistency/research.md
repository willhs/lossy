---
id: 0035-sw4-first-pass-face-consistency
type: note
purpose: "Can SW4 faces be made consistent in the first pass (stop LTX drifting inside a shot), and does the HyperSwap inner-face post-pass look wrong on heads whose hair and shape differ from the sheet? Go/no-go for the 30 Nov run."
scope: ["research", "decode", "keyframes", "character-continuity", "star-wars"]
tags: ["research", "star-wars", "ltx-2.5", "keyframe", "anchor", "face-swap", "go-no-go"]
related:
  - "research/0034-sw4-mixed-shot-keyframes/research.md"
  - "research/0030-sw4-full-run-preflight/research.md"
  - "research/0029-character-reference-continuity/research.md"
---

# SW4: first-pass face consistency and the inner-face swap look (24 Nov 2026)

Vision alignment: Have an online presence (goal: complete a movie or TV show reconstruction, Star Wars IV)

Steady task `dffa7b47`. Will would rather fix character consistency in the render than paste
faces on afterwards, and worries that an inner-face swap looks odd when hair and head shape
change from shot to shot. The 30 Nov full run (task `0725e309`) is on hold until this reports.

## Pre-registration (written and committed before any arm was rendered or labelled)

### Premise check (from 0034's own tables and a dry run; nothing rendered yet)

- **"Obi-Wan's keyframe itself is off the sheet (0.41 < 0.45)" holds only with the mixed-shot
  flag on.** 0034's 0.41 averages in 1294 (Obi-Wan's face merged into a trooper mask, 0.00) and
  1295, both mixed shots. With the flag off, which is the chosen setup, Obi-Wan's keyframe shots
  are 1290, 1292 and 1296, at 0.47 / 0.61 / 0.53 (mean 0.54). The Obi-Wan lever is therefore
  gated on the new baseline (rule O below), not assumed. **[Inferred from 0034's table]**
- **"Shorter clip lengths for faces" has nothing to shorten in this window.** Every keyframe
  shot here is 0.8-4.3 s, under LTX-2.5's one-clip limit (121 frames, 4.8 s), so each is already
  one clip of its own length (`--dry-run`: 14 keyframe shots, 24 text). Drift inside a 1.8 s
  clip (0034's 1287: 0.68 -> 0.27) is not a clip-length effect. This lever is not rendered; its
  film-wide reach (keyframe shots long enough to be split) is counted instead. **[Observed]**
- **Reference/identity conditioning:** the only option in the pipeline is the LTX-2.5
  Ingredients IC-LoRA (0029 arm I). Its weights are gated on Hugging Face and this job has no
  `HF_TOKEN`, so it cannot run here. **[Observed]**
- **The lever that can run: re-anchoring on the keyframe later in the clip.** ComfyUI's
  `LTXVAddGuide` (pinned v0.39.2) pins an image at any frame, `-1` being the last, and
  `LTXVCropGuides` cuts the guide frames out before decode. Added as
  `--keyframe-anchor-end S` and `--keyframe-anchor-mid S` (off by default; graph tests in
  `tests/test_keyframe_anchor.py`).

### Arms (same window, shots 1284-1321, Will's chosen setup: one-stage 640x352, keyframe-first,
klein on the pod, mixed-shot flag OFF, native audio, stitched at 1280x704)

| Arm | Change | What it tests |
|---|---|---|
| **B** base | none (0030/0034 setup) | the reference: fresh, so all arms share one set of sheets and keyframes |
| **E1** end100 | keyframe pinned as the last frame, strength 1.0 | strongest first+last anchoring |
| **E2** mid50end50 | keyframe guided at the middle and the last frame, strength 0.5 each | softer mid-shot re-anchor that should leave room for motion |

E1 and E2 are forks of B with only the keyframe shots re-rendered (`window_anchor.sh`), so their
text shots, sheets and keyframes are B's exactly and the anchor is the only difference.

### Measures

0034's scorer (`../0034-sw4-mixed-shot-keyframes/score_window.py`) and labelling rule, unchanged:
first, middle and last frame of each single-person shot (0034's 15: Han 6, Obi-Wan 6, Leia 3),
plus the drawn keyframe; every YuNet face labelled by eye from the crop sheet (the character's
own human face; never a helmet, droid, Wookiee or the other named person; equal fits take the
larger; under 60 px or turned away dropped). SFace cosine.

- **K-mid (primary):** mean middle-frame cosine to the sheet over the **9 single-person keyframe
  shots** (Han 1285/1287/1307, Obi-Wan 1290/1292/1296, Leia 1286/1297/1321). The anchors touch
  only keyframe shots, and the middle frame is the honest one: an end anchor makes the last
  frame look like the keyframe by construction.
- **Drift:** mean keyframe-to-sheet minus last-frame-to-sheet over the same 9 shots (and
  keyframe minus middle).
- **M** (0034's primary: middle-frame across-shot cosine, 15 shots, face-weighted) and the
  **pooled** across-shot view, as guard rails.
- **By eye**, all 14 keyframe shots of each anchor arm against B: a shot counts as damaged if
  the anchor makes it visibly frozen or static where B moves, snaps or pops back to the keyframe
  near its end, or doubles or merges a person.

### Decision rules

1. **A first-pass change clears the bar** if, against B: K-mid rises **>= +0.05**; drift
   (keyframe -> last) shrinks by **>= 0.10**; M and pooled across-shot fall by no more than
   **0.03**; and **<= 2 of 14** keyframe shots are damaged by eye. If both arms clear, the one
   with the higher K-mid; a tie goes to fewer damaged shots.
2. **Rule O (Obi-Wan):** if B's Obi-Wan keyframes average **< 0.45** to the sheet, redraw his
   sheet (new seed) and re-render his keyframe shots as one more arm. Otherwise the lever is
   closed: his keyframe is not the problem.
3. **Swap look (task step 2):** HyperSwap (0034's `face_swap_trial.py`, unchanged) on the best
   first pass (B if no arm clears). Side-by-side with and without the swap. Pick **8** labelled
   face crops where the hair or head shape visibly differs from the sheet, each shown as sheet |
   unswapped | swapped. A crop **looks wrong** if the swapped face visibly does not belong on
   that head: a skin-tone or lighting seam, a face too large/small or at a different angle from
   the head, or features that read as a mask. **The swap passes the look test if <= 2 of 8 look
   wrong.**
4. **Swap on top** is recommended only if it passes the look test **and** raises pooled
   across-shot similarity by **>= 0.08** on the best first pass.
5. **Recommendation:** an arm clears -> GO 30 Nov with that flag (cost change stated). No arm
   clears and the swap passes -> GO 30 Nov as 0034 said (render unchanged, swap afterwards). No
   arm clears and the swap looks wrong -> say so plainly and name the next lever and whether it
   needs a re-render; GO 30 Nov only if the next lever works on finished clips, otherwise not yet.
6. **Stop early:** if E1 and E2 both fail rule 1, no further first-pass arm is rendered beyond
   rule O.

**Spend cap:** **$6 RunPod + fal** for the whole task, from a RunPod balance of **$28.19** (0
pods) when the task started, via `tools/capped_run.py` with each arm's ownership journal; 0
pods and empty journals at the end. Everything runs from a scratch copy of the encode in the
job workspace (0030's photographic `characters.json`, shot fixes (108 stale tags), recast voices,
speaker fixes); the primary checkout is only read.

**Update 28 Nov 2026:** the swap's flicker (finding 9) is fixed; see [The flicker fix](#the-flicker-fix-28-nov-2026).

## Verdict (25 Nov 2026)

**No first-pass change clears the bar. Re-anchoring LTX on the keyframe does hold the face, but
only by freezing the shot: with the end anchor, every keyframe shot that moves in the baseline
(9 of 14) barely moves
(people running down a corridor stand still mid-stride), which is worse than the drift it fixes.
The swap passes the pre-registered still-frame look test (1 of 8 crops looks wrong) and lifts
across-shot similarity by +0.14, but as it runs today it flickers on and off inside 4 of 12
single-person shots. Recommendation: GO for 30 Nov on 0030's setup, unchanged (low res +
native audio, no mixed-shot flag, no anchor). Consistency work continues on the finished clips
and needs no re-render: fix the swap's flicker (decide per shot, not per frame) before any
swapped footage is published, and try the Ingredients reference LoRA, which is now unblocked, on
this window as the remaining first-pass lever.**

Spend: **$1.02 RunPod** ($28.19 -> $27.17; base $0.51, the two anchor arms $0.51 together) **+
cents of fal speech** (27 lines; one, "Blast them", refused by fal's content filter). **0 pods,
every ownership journal empty** (`evidence/*/hygiene.txt`, final balance read: 0 pods). The swap
ran on the Mac CPU ($0). Everything ran from the job workspace's scratch copy of the encode.

Confidence: **observed** = measured here; **inferred** = arithmetic from measured figures; **by
eye** = read off the filmstrips and crops named.

### How it ran

Resumed on 25 Nov with secrets from 1Password (`op run --env-file .env.tpl`). `op run` gives
the process a HOME with no ssh key, so the job made its own throwaway pod key in the workspace
(`LOSSY_RUNPOD_SSH_KEY`); it read no other credential. Base, then the two anchor arms in parallel
(each a fork of base with only the 14 keyframe shots re-rendered, so text shots, sheets and
keyframes are byte-identical across arms: `cmp` on `kf_1287.png`). All 38 shots completed in
every arm (one clip, 1307 in end100, took a second attempt).

### Numbers (labelled faces, 0034's scorer and rule; `summarise.py`, `evidence/summary.txt`)

| Arm | **K-mid** (9 keyframe shots, middle frame to sheet) | drift keyframe -> last | M | pooled across-shot |
|---|---|---|---|---|
| **B** base (0030 setup) | **0.445** | 0.153 | 0.292 | 0.391 |
| **E1** end anchor 1.0 | 0.507 (**+0.062**) | 0.032 (**-0.12**) | 0.355 (+0.06) | 0.400 (+0.01) |
| **E2** mid + end anchor 0.5 | 0.486 (+0.040) | 0.026 (-0.13) | 0.339 (+0.05) | 0.387 (-0.00) |
| B + HyperSwap | 0.591 (+0.145) | 0.040 | 0.342 (+0.05) | **0.528 (+0.137)** |

Base reproduces 0034's old window (M 0.29 vs 0.30) and its per-shot drift (1287 0.68 -> 0.26,
1297 0.52 -> -0.04 at mid-shot). **[Observed]**

Per shot, keyframe / middle / last cosine to the sheet:

| Shot | Who | Base | End anchor 1.0 | Mid+end 0.5 |
|---|---|---|---|---|
| 1285 | Han | 0.59 / 0.57 / 0.52 | 0.59 / 0.54 / 0.55 | 0.59 / 0.56 / 0.54 |
| 1287 | Han | 0.68 / 0.58 / **0.26** | 0.68 / 0.66 / 0.66 | 0.68 / 0.65 / 0.64 |
| 1307 | Han | 0.66 / 0.63 / 0.65 | 0.66 / 0.62 / 0.62 | 0.66 / 0.63 / 0.63 |
| 1290 | Obi-Wan | 0.47 / 0.62 / 0.63 | 0.47 / 0.58 / 0.59 | 0.47 / 0.60 / 0.56 |
| 1292 | Obi-Wan | 0.61 / 0.57 / 0.57 | 0.61 / 0.60 / 0.60 | 0.61 / 0.59 / 0.61 |
| 1296 | Obi-Wan | 0.53 / **0.32 / 0.24** | 0.53 / 0.55 / 0.50 | 0.53 / 0.49 / 0.53 |
| 1286 | Leia | 0.53 / 0.50 / 0.38 | 0.53 / 0.42 / 0.54 | 0.53 / 0.51 / 0.57 |
| 1297 | Leia | 0.52 / **-0.04** / 0.21 | 0.52 / 0.33 / 0.41 | 0.52 / 0.18 / - |
| 1321 | Leia | 0.43 / 0.26 / 0.17 | 0.43 / 0.27 / 0.25 | 0.43 / 0.17 / 0.20 |

### Findings

1. **Rule O: Obi-Wan's keyframe is not the problem.** Base keyframes 0.47 / 0.61 / 0.53 to the
   sheet, mean 0.54 >= 0.45. Lever closed, as the premise check predicted. **[Observed]**
2. **The anchor's face gain is a freeze, so E1 fails rule 1 on the by-eye part.** E1 passes
   every number (K-mid +0.06, drift -0.12, M +0.06, pooled flat), but the filmstrips
   (`evidence/eye_a.jpg`, `eye_b.jpg`) show the keyframe shots turned into near-stills: in 1315,
   1319 and 1320 Luke, Han and Leia sprint down the corridor in base and stand frozen in E1; Han's
   shout in 1307 becomes a held frame. Measured as mean frame-to-frame change (`anchor_eye_check.py`,
   `evidence/eye_check.txt`): base 0.5-12.5, E1 0.11-0.51 in 13 of 14 shots and 1.54 in 1297.
   The 9 shots that visibly move in base (1286, 1287, 1297, 1305, 1307, 1315, 1318, 1319, 1320:
   base motion 3.5-12.5) are frozen in E1; the other 5 were near-static already (base 0.5-1.3).
   Pinning the same image at both ends makes LTX's safest clip one that does not move. The face
   holds *because* nothing happens. **9 of 14 damaged against a limit of 2; fail.**
   **[Observed]**, **[by eye]**
3. **E2 fails both parts.** K-mid +0.04 (< +0.05), and of the same 9 moving shots it freezes 8
   (motion 0.46-2.21 against base 3.5-12.5) and keeps motion in 1315 only with a visible pop at
   its end (end-snap 19.9x the clip's median frame change): 9 of 14 damaged. **[Observed]**,
   **[by eye]**
4. **Drift and motion are the same thing here.** The shots that drift in base (1287, 1296, 1297,
   1321) are the ones where the person moves or turns; the ones that hold (1285, 1292, 1307 start)
   barely move. An anchor at strength 1.0 or 0.5 trades one for the other; there is no setting in
   these two that keeps both. **[Inferred]** from the per-shot table and motion numbers.
5. **Shorter clips: nothing to test in this window, and probably the same trade film-wide.**
   Every keyframe shot here is a single 0.8-4.3 s clip. Film-wide, 127 of 793 keyframe shots (16%)
   are long enough to be split into chained parts (median keyframe shot 2.3 s;
   `evidence/keyframe_shot_lengths.txt`). Re-starting each part from the keyframe would reset the
   face at every seam, which is a visible jump. Not rendered. **[Observed]** (counts),
   **[inferred]** (seams)
6. **Reference conditioning (Ingredients IC-LoRA) is now unblocked but was not run.** On resume
   the injected `HF_TOKEN` passes the Lightricks gate (`check_gated_access`: OK). The
   pre-registered stop rule (rule 6) and the task's "stop early" stopped further first-pass arms
   here, and arm I runs only through `--ab-manifest` on the two-stage graph, so it is its own
   experiment, not a flag on this window. It is the one first-pass lever left untested. Its guide
   is a static sheet repeated on every frame, so the freeze risk found here applies to it too.
7. **The swap, numbers: helps Han a lot, Leia a little, Obi-Wan not at all.** HyperSwap (0034's
   `face_swap_trial.py`, unchanged) on base: pooled across-shot +0.14 overall (Han 0.38 -> 0.67,
   Leia 0.36 -> 0.41, Obi-Wan 0.43 -> 0.40). On Obi-Wan's shots the swap fires but changes almost
   no pixels (`swap_flicker.py`: 0 frames with a visible change), so his face is already as close
   to the sheet as the swapper can make it. 242 s of Mac CPU for the window (3,046 frames).
   **[Observed]**
8. **The swap, look test (rule 3): 1 of 8 looks wrong on stills, so it passes.**
   `evidence/look_test_8.jpg`, sheet | unswapped | swapped, for 8 faces whose hair or head
   differs from the sheet (Han's sheet has short neat hair; the film draws him swept back, long
   and shaggy, and curly):
   - **1306/first (long shaggy hair): looks wrong.** The swapped face is smoother and paler than
     the neck and reads as a mask set inside a wig.
   - 1314/first and 1314/middle (big curly hair): look right; the features change, the head
     carries them.
   - 1285/middle, 1287/last, 1307/middle (swept-back and fuller hair): look right; Han's face is
     sharper and more like the sheet.
   - 1297/last (Leia in profile, hair back) and 1321/middle (small, soft face): barely changed;
     fine.

   So on stills, Will's worry is mostly not borne out: a different hairline does not make the
   inner face look pasted on, except on the one long-haired shot. **[By eye]**
9. **But in motion the swap flickers, and that does look weird.** The swap fires only on frames
   where the face scores >= 0.10 against the sheet, so on shots that hover near the line it
   switches on and off: 1314 switches 13 times in 3.4 s (10 runs under half a second), 1297 8 times,
   1321 8 times, 1306 6 times (`evidence/swap_flicker.txt`; `evidence/swap_flicker_1314.jpg`
   shows frames 2-8 swapped, 9-11 not, and so on). By eye at full size the eyes and brows twitch
   between two people. That is the real look problem, and it is in **4 of the 12** single-person
   shots. It is a gating bug, not a limit of the swapper: decide once per shot (swap every frame
   of the shot or none), or add hysteresis. Not fixed here. **[Observed]**, **[by eye]**
10. **Rule 4 (swap on top): yes by the pre-registered rule** (look test passed, pooled +0.14 >=
    0.08), **with the flicker fixed first**; as it runs today it should not go into a published
    cut. **[Inferred]** from 8 and 9.

### Cost change for the full film (2,070 shots)

| Change | RunPod | Other | Verdict |
|---|---|---|---|
| `--keyframe-anchor-end 1.0` | ~$0 (one guide frame per clip; the arm cost what base did per clip) | - | **no**: freezes keyframe shots |
| `--keyframe-anchor-mid/end 0.5` | ~$0 | - | **no**: under the bar and still freezes |
| HyperSwap post-pass, Mac CPU | $0 | **~4 CPU-hours** (inferred: 242 s for this 2-minute window x ~60 windows) | **yes, after the per-shot gate fix**; single-person shots only |
| Ingredients arm (untested) | ~$0.5 for this window (inferred: two-stage graph, 14 clips) | - | next experiment |

The swap's 4 CPU-hours replaces 0034's 50 CPU-hour estimate, which assumed 1.6 s per frame; this
run measured about 0.08 s per window frame (most frames hold no single person and are only
decoded and re-encoded). Licence as 0034: facefusion's research / non-commercial weights.

### Recommendation for 30 Nov

**GO on 0030's setup, unchanged** (low res + native audio, ~$23; `--keyframe-mixed-shots` off;
**no anchor flags**). Nothing tested here improves the render itself without breaking motion, and
both remaining levers work on the finished film without re-rendering it: the swap (after the
flicker fix) runs on the clips, and an Ingredients result could re-render just the keyframe shots
later. Cost change for the run: **$0**.

### Where things are (job workspace `evidence/`, outside git)

- `side_by_side_base_vs_swapped.mp4`: base | base + swap, 2 min, swapped audio (rule 3's side-by-side)
- `side_by_side_base_end100_mid50end50.mp4`: base | end anchor 1.0 | mid+end 0.5 (the freeze)
- `look_test_8.jpg` (the 8 crops), `look_candidates.jpg` (the 11 looked at), `swap_flicker_1314.jpg`
- `eye_a.jpg`, `eye_b.jpg`, `eye_check.txt` (filmstrips and motion/snap per keyframe shot)
- `scores/<arm>/{candidates,labels,score}.json` and `crops.jpg`; `summary.txt`; `swap_flicker.txt`
- `swap/base_swapped.mp4` (+ `.json`); `<arm>/{video,stitch,speech}.log`, `<arm>/hygiene.txt`
- Stitched windows: `runs/<arm>/<arm>_reconstructed_runpod-ltx25-keyframe+native.mp4`

The task asked for the side-by-sides in `~/projects/lossy/output/sw4-consistency-<date>/`; that
is the primary checkout, which this job may not write. From the repo root on the Mac:

```
D=~/projects/lossy/output/sw4-consistency-20261125; mkdir -p $D
cp /Users/will/workspaces/sw4-make-faces-consistent-in-the-first-cd09b7/evidence/{side_by_side_*.mp4,look_test_8.jpg,swap_flicker_1314.jpg,eye_b.jpg} $D/
```

### Code and tools added

- `decode.py --keyframe-anchor-end S / --keyframe-anchor-mid S` (with `runpod-ltx25-keyframe`):
  pins the shot's keyframe as a later guide frame (`LTXVAddGuide`, cropped out before decode;
  `strategies_video.add_anchor_guides`). Off by default and **not recommended** (finding 2); kept
  because it is small, tested (`tests/test_keyframe_anchor.py`) and the measurement for any later
  anchor experiment at lower strength.
- Research tools here: `window_anchor.sh` (the arms), `summarise.py` (pre-registered numbers),
  `anchor_eye_check.py` (filmstrips + motion/snap), `look_crops.py` (rule 3 crops),
  `swap_flicker.py` (swap on/off runs per shot).

## The flicker fix (28 Nov 2026)

Steady task `32fb7ca7`. Same window, same finished clips, Mac CPU only, **$0**, nothing published.

**Fixed: 0 of the 12 shots switch on and off (6 did before), and pooled across-shot similarity is
0.569, +0.178 over no swap (the per-frame swap gave 0.528, +0.137).** The swap is now
`tools/face_swap.py`, which decides once per shot. 0034's `face_swap_trial.py` is kept only as the
record of how the earlier videos were made.

### Cause

Measured by logging every face on every frame of the 15 one-person shots. **[Observed]**

1. **The per-frame gate.** 1297, 1314 and 1321 have a face on every frame, but its score crosses
   0.10 over and over.
2. **Two faces on screen.** "Most like the sheet on this frame" could move from one person to the
   other (1321 has two people on almost every frame).
3. **Shot edges were wrong (new).** The stitched video cuts up to 3 frames away from the
   `shots.json` times, so each shot was handed a few of its neighbour's frames. The swap and the
   flicker measure both did this. Deciding per shot on those times would still switch at the edges.

### The rule

- Shot edges come from the video: the biggest frame-to-frame change within 5 frames of each
  shot's start time. On this window the cut stands well clear of its neighbours at every boundary.
- A face counts only if it can be followed through **every** frame of the shot. The one most
  like the sheet (median over the shot) is the target.
- It is swapped on every frame if it reaches 0.10 on **any** frame; otherwise nothing is.
  These are the shots the old gate already touched, so no new threshold was tuned.

### Flicker, before and after

Switches inside each shot, `swap_flicker.py` with `--window 38 --thresh 5`:

| Shot | Before | After | What happens now |
|---|---|---|---|
| 1285, 1286, 1287, 1290, 1292, 1307 | 0 | 0 | swapped throughout |
| 1296 | 1 | 0 | swapped throughout |
| 1297 | 11 | 0 | swapped throughout |
| 1314 | 13 | 0 | swapped throughout |
| 1321 | 7 | 0 | swapped throughout |
| 1306 | 5 | 0 | **left unswapped** (the face is not there all shot) |
| 1308 | 1 | 0 | **left unswapped** (same) |
| 1291, 1294 (not in the 12) | 0 | 0 | left unswapped |
| 1295 (not in the 12) | 6 | 0 | left unswapped |

| Count over the 12 shots | Before | After |
|---|---|---|
| Any switch, true cuts, threshold 5 | 6 | **0** |
| Short runs under 0.5 s (0035's "4 of 12"), true cuts, threshold 5 | 4 | **0** |
| Any switch, true cuts, threshold 10 | 5 | 1 (1297) |

**Why the threshold moved from 10 to 5.** A frame left unswapped changes by at most 4.7 (re-encode
noise, all 2,396 of them); a swapped one by at least 5.5 (all 650). Faint swaps (Obi-Wan, and 1297 when the
head turns) sit between 5.5 and 10, so 10 reads them as off. 1297 is swapped on all 107 frames
(the tool's own log) and its change slides smoothly from 18 to 6 and back. **[Observed]**

Running 0035's original command (no flags) on the fixed video still shows switches in 1287,
1297, 1307, 1308, 1314 and 1321: it uses the `shots.json` times, so it counts neighbours' frames
(and 1297's faint stretch). With no flags the script reproduces 0035's original table exactly.

### Similarity

0034's scorer and 0035's by-eye labels, carried to each video by box position (`carry_labels.py`;
it reproduces 0035's hand labels for its swapped video on all 54 frames).

| Video | K-mid | M | **pooled across-shot** | Han | Leia | Obi-Wan |
|---|---|---|---|---|---|---|
| No swap | 0.445 | 0.292 | **0.391** | 0.39 | 0.36 | 0.43 |
| Per-frame swap (0035) | 0.591 | 0.342 | **0.528** | 0.67 | 0.41 | 0.40 |
| **Per-shot swap (shipped)** | 0.639 | 0.600 | **0.569** | 0.59 | 0.75 | 0.40 |
| Not shipped: median must reach 0.10 | 0.595 | 0.338 | **0.398** | 0.43 | 0.32 | 0.40 |

- Leia gains most, because 1297 and 1321 are now swapped on every frame.
- Han gains less than before, because 1306 and 1308 are now left alone. 1306 was the one crop
  that looked wrong in the look test.
- The stricter rule (swap only if the shot's typical frame reaches 0.10) was measured to check
  the choice. It leaves 1297, 1314 and 1321 alone and loses almost the whole gain.
- The scorer still picks its first and last frames from `shots.json` times, the same for every
  row. **[Observed]**

**By eye:** face crops along 1297, 1314 and 1321 (job `evidence/strips_flicker_shots.jpg`) show
the per-frame swap alternating between two faces and the per-shot swap holding one. In some 1321
frames the swapped eyes point a little differently from the unswapped ones. Not checked in
motion by a person yet.

### Limits

- One window. Film-wide, one noisy frame reaching 0.10 would now swap a stranger for a whole
  shot, where before it flickered. Check the per-shot log (`OUT.mp4.json`) for low-median swaps.
- A single missed face detection leaves the whole shot unswapped.
- Cut finding assumes cuts land within 5 frames of `shots.json` times; two near-identical shots
  side by side could be cut wrongly.
- Frames of one shot are held in memory: the film's longest one-person shot (66 s) needs about
  4.5 GB.
- Time is unchanged in practice: 255 s for the window (0035: 242 s).

### Clips and how to re-run

The clips are **not in git**: this repo is public and the task says publish nothing. They are in
`clips/` here (gitignored) and in the job workspace `evidence/clips/`:
`before_after_flicker_shots.mp4` (side by side: 1297, 1306, 1308, 1314, 1321),
`window_before_per_frame.mp4` and `window_after_per_shot.mp4`.

```
PYTHONDONTWRITEBYTECODE=1 /Users/will/projects/screenquote/.venv/bin/python tools/face_swap.py \
    BASE.mp4 RUN_DIR 1284 38 MODELS_DIR OUT.mp4
uv run python docs/research/0035-sw4-first-pass-face-consistency/swap_flicker.py \
    BASE.mp4 OUT.mp4 RUN_DIR 1284 1285 1286 1287 1290 1292 1296 1297 1306 1307 1308 1314 1321 \
    --window 38 --thresh 5
uv run python docs/research/0035-sw4-first-pass-face-consistency/before_after.py \
    BASE.mp4 BEFORE.mp4 OUT.mp4 RUN_DIR SIDE_BY_SIDE.mp4 1297 1306 1308 1314 1321
```
