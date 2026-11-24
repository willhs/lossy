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

## Status (24 Nov 2026): parked before any render; no spend

The pre-registration, the anchor flags and the arm script are committed. **Nothing has been
rendered and $0 has been spent** (RunPod balance read $28.19 with 0 pods at the start of the job;
no pod was rented and no ownership journal was written).

**Why it stopped.** The job could read `RUNPOD_API_KEY` and `FAL_KEY` through the worktree's
`.env` symlink when it started, but the symlink was gone by the time the baseline launched, so
the script's first balance read got an empty answer and it exited before renting anything. The
job is defined as having no secrets, so it does not go looking for keys elsewhere.

**Found so far (no render needed):**

- Premise corrections above: Obi-Wan's keyframes are only off the sheet with the mixed-shot flag
  on; the "shorter clips" lever has nothing to shorten in this window; Ingredients needs an
  `HF_TOKEN`. **[Inferred / observed]**
- **Film-wide, 127 of 793 keyframe shots (16%) are long enough (> 4.84 s) to be split into
  chained clips**; 457 run over 2 s; the median is 2.3 s (`decode --dry-run`, full film).
  So clip length can matter film-wide even though it does not in this window, and a split
  shot's later parts start from the previous part's last frame, carrying any drift with them.
  With `--keyframe-anchor-end`, every part of a split keyframe shot is pulled back to the
  keyframe at its end (`tests/test_keyframe_anchor.py::test_later_parts_keep_the_shots_anchor`).
  **[Observed]**

**To resume** (a job with RunPod + fal secrets, or Will on the Mac with `op run`), from the repo
root. `runs/base` is the workspace's scratch copy of the encode with 0030's fixes:

```
WS=<workspace with runs/base and face_models> BASELINE=<balance now>
ARM=base       bash docs/research/0035-sw4-first-pass-face-consistency/window_anchor.sh
ARM=end100     bash docs/research/0035-sw4-first-pass-face-consistency/window_anchor.sh &
ARM=mid50end50 bash docs/research/0035-sw4-first-pass-face-consistency/window_anchor.sh
```

Expected spend: about $0.6 for base (pod boot, klein sheets and keyframes, 38 clips, as 0034
measured) and about $0.3 for each anchor arm (boot plus 14 clips), so about $1.2 of the $6 cap,
plus cents of fal speech. Then score with 0034's `score_window.py` under the rules above, and run
the swap look test (rule 3).
