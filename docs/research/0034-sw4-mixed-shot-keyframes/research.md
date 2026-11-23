---
id: 0034-sw4-mixed-shot-keyframes
type: note
purpose: "Why SW4 characters were not consistent in the 2-minute window test, whether keyframing people who share a shot with troopers/droids fixes it, and a go/no-go for the 30 Nov run."
scope: ["research", "decode", "keyframes", "character-continuity", "star-wars"]
tags: ["research", "star-wars", "ltx-2.5", "keyframe", "klein", "face-consistency", "go-no-go"]
related:
  - "research/0030-sw4-full-run-preflight/research.md"
  - "research/0029-character-reference-continuity/research.md"
---

# SW4 character consistency: mixed-shot keyframes (23 Nov 2026)

Vision alignment: Have an online presence (goal: complete a movie or TV show reconstruction, Star Wars IV)

Steady task `82e4bf88`. Will watched research 0030's 2-minute window (shots 1284-1321) and "didn't
see the character consistency I was hoping for". The full run (task `0725e309`) is on hold until
this is answered.

## Pre-registration (written and committed before the new window was rendered or labelled)

**What was already known when this was written.** The flag below is built and unit-tested. The
old window (0030, low res + native audio, `output/sw4-window-test-20261123/`) and the source
film were run through the new candidate extractor, but neither was labelled or scored. One
thing was found on the way, and it is why the measure changes: 0030's `window_faces.py` embeds
"the first face YuNet finds". In this window that is often a stormtrooper helmet (YuNet scores
helmets up to 0.88), Chewbacca (0.82) or the other person in the shot (Luke in Han's 1285/1287),
and the shot map tags 1297 as Leia where the source shows Han from behind. Run unchanged on the
**real film**, 0030's measure gives Han 0.18, Obi-Wan 0.32, Leia 0.03 across shots (middle frame):
as low as the reconstruction. So 0030's 0.17-0.26 is mostly measuring the wrong faces.

**Measure** ([`score_window.py`](score_window.py)). Shots: the 15 whose shot map names exactly one
person (0030's set; Han 6, Obi-Wan 6, Leia 3). Frames: first, middle, last (0.06 s in from each
cut), scaled to 1280x704. Every YuNet face >= 40 px is a candidate, embedded with SFace. Whose face
it is gets decided by eye from the crop sheet under a fixed rule: the character's own human face,
never a helmet, droid or Wookiee, never the other named person (Luke blond young man, Han
dark-haired man, Obi-Wan old bearded man, Leia young woman); two that fit equally, the larger;
under 60 px or turned away, dropped. The automatic "largest face" number is reported beside it.

- **Primary (M):** mean pairwise SFace cosine across shots of each character's labelled middle-frame
  face, averaged over Han, Obi-Wan and Leia weighted by labelled faces. Low res + native audio arm
  (Will's chosen setup), old window vs new window. Ceiling: the source film, same labelling.
- **Secondary:** cosine to the character's sheet (both windows against the new run's sheets: the
  old run's sheets were not kept; Luke/Han/Obi-Wan have the same prompt and seed, Leia's block
  changed in `18e2855`), and the drift view on the new run's keyframe shots: drawn keyframe,
  first, middle and last frame, each to the sheet.

**Decision rules.**

1. Mixed-shot routing is worth turning on if **M rises by >= 0.10** over the old window, with
   troopers/droids still right on the contact sheet and no shot drawn with an extra or merged
   person that the text path did not have.
2. Where the loss is: keyframe-to-sheet **< 0.45** means klein/sheet (reference strength or sheet
   quality); keyframe-to-last-frame drop **>= 0.15** means LTX drifting from the first frame.
3. Face-swap post-pass (task step 4) is tried if the new **M < 0.40** (SFace's same-person line is
   about 0.36; Seinfeld 0029 had 0.55) **and** the source ceiling is clearly higher, so there is
   something to gain, and only if it can run inside the $6 cap with no new project dependency.
4. Go for 30 Nov with the flag on if rule 1 passes. If it fails, the 30 Nov recommendation is the
   0030 setup without the flag, and the consistency problem is written up as what it is.

Spend cap for the whole task: **$6 RunPod**, via `tools/capped_run.py` with this run's ownership
journal; 0 pods at the end.

**Addendum, still before the new window was scored (render in progress).** The source and old
window were labelled (`labels.json` beside each `candidates.json`, evidence dir) and scored. The
middle frame alone leaves the source with 3 Han, 3 Obi-Wan and 1 Leia face, too thin for a
ceiling, so a **pooled** view is added as a second headline: each shot's labelled frames averaged
into one face, then compared across shots, plus first-vs-last frame within a shot (drift). It
sits beside M; M stays the pre-registered primary. Because SFace may not separate real from
reconstructed faces at this size at all, a **visual identity grid** (each character's labelled
middle-frame face per shot: source, old, new) is also made and read by eye; where it and the
cosine disagree, both are reported.

## Verdict

**The mixed-shot flag fails its pre-registered bar; leave it off. The face-swap post-pass is the
only thing tried that measurably tightens faces, and only the inner face. GO for 30 Nov on 0030's
setup (low res + native audio, flag off), and treat consistency as a post-pass that does not need
the render to wait.** Spend: **$0.57 RunPod** ($28.76 -> $28.19) **+ $0.05 fal** speech, of the $6
cap. **0 pods, ownership journal empty** (`evidence/hygiene_final.txt`). Everything ran from a
scratch copy of the encode in the job workspace; the primary checkout was only read.

Confidence: **observed** = measured here; **inferred** = arithmetic or reading from measured
figures; **by eye** = read off the identity grid / crop sheets.

### Premise check (what the task assumed vs what is true)

- **"1,047 shots go to text because of a design character" is true, but only 266 of them hold a
  person.** The other 781 are troopers, droids or Chewbacca alone and stay text either way. The
  flag moves the film from **793 to 1,059 keyframe shots (+266, 13% of the film)**; in this window
  from 14 to 24 (`--dry-run`, `evidence/dryrun_mixed_window.txt`). **[Observed]**
- **0030's face numbers (Han 0.20, Obi-Wan 0.17, Leia 0.26) mostly measured the wrong faces.**
  `window_faces.py` took YuNet's first face: a trooper helmet in 1294/1295/1308, Chewbacca in
  1314, Luke in Han's 1285/1287. On the **real film** the same measure gives Han 0.18, Obi-Wan
  0.32, Leia 0.03. So "low compared with Seinfeld's 0.55" was never evidence about consistency.
  **[Observed]**
- **Even labelled, SFace cannot separate this reconstruction from the real film at this size.**
  Labelled pooled across-shot cosine: source Han 0.43 / Obi-Wan 0.45 / Leia 0.32, old window
  0.39 / 0.43 / 0.34. The cosine is a coarse instrument here; the identity grid carries the
  visual reading. **[Observed]**

### Numbers (labelled faces; low res + native audio)

Across = mean pairwise SFace cosine across shots; sheet = mean cosine to the run's sheet
(the swap uses the sheet as its source identity, so its sheet column rises partly by
construction; across-shots is the fair one). M is face-weighted over the three characters.

| Version | **M** (middle frame) | Han across / sheet (pooled) | Obi-Wan | Leia |
|---|---|---|---|---|
| Source film (ceiling) | 0.27 (Leia has 1 face) | 0.43 / - | 0.45 / - | 0.32 / - |
| Old window (0030) | **0.30** | 0.39 / 0.45 | 0.43 / 0.43 | 0.34 / 0.34 |
| New: mixed-shot keyframes | **0.24** | 0.41 / 0.44 | 0.48 / 0.45 | 0.36 / 0.37 |
| New + HyperSwap post-pass | **0.37** | **0.54 / 0.61** | **0.57 / 0.57** | **0.41 / 0.48** |

Full per-position tables: `evidence/scores/<version>/score.json`; labels beside them. **[Observed]**

Drift view, new window, cosine to the sheet at drawn keyframe -> first -> middle -> last frame
(per shot, `after/score.json`):

| Shot | Who | Keyframe | First | Middle | Last |
|---|---|---|---|---|---|
| 1285 | Han | 0.59 | 0.54 | 0.57 | 0.52 |
| 1287 | Han | 0.68 | 0.66 | 0.58 | **0.27** |
| 1307 | Han | 0.66 | 0.58 | 0.63 | 0.65 |
| 1308 | Han (mixed) | 0.58 | 0.57 | **0.02** | 0.11 |
| 1314 | Han (mixed) | 0.51 | 0.36 | 0.31 | 0.29 |
| 1290 | Obi-Wan | 0.47 | 0.64 | 0.62 | 0.63 |
| 1292 | Obi-Wan | 0.61 | 0.58 | 0.56 | 0.57 |
| 1294 | Obi-Wan (mixed) | **0.00** (merged) | -0.01 | 0.25 | 0.28 |
| 1295 | Obi-Wan (mixed) | 0.46 | 0.34 | 0.14 | - |
| 1296 | Obi-Wan | 0.53 | 0.51 | 0.32 | 0.25 |
| 1286 | Leia | 0.53 | 0.54 | 0.50 | 0.38 |
| 1297 | Leia | 0.52 | 0.48 | **-0.04** | 0.21 |
| 1321 | Leia | 0.44 | 0.11 | 0.26 | 0.17 |

Sheets themselves are self-consistent (panel-to-panel 0.65-0.72 for all four). **[Observed]**

### Findings

1. **Rule 1 (flag on?) fails.** M fell 0.30 -> 0.24; pooled rose only +0.02 to +0.05; and the
   flag drew **one merged person in 10 mixed shots**: 1294 is Obi-Wan's face inside a stormtrooper
   mask (keyframe to sheet 0.00). By eye it does help in places (Han in 1306 is no longer a
   long-haired stranger) and troopers, droids and three-person mixed shots (1288, 1302) render
   fine (`evidence/contact_sheet_new.jpg`). Net: no measurable gain, one new defect type.
   **[Observed]** (numbers), **[by eye]** (images)
2. **Where the identity goes (task step 2): mostly LTX drifting away from the first frame.**
   Sheets are fine (0.65-0.72 internally) and klein mostly lands the face (keyframe to sheet
   0.44-0.68, Obi-Wan's weakest). Inside the clip it drops: 7 of 13 shots lose >= 0.15 by the
   middle or last frame (1287, 1308, 1314, 1295, 1296, 1297, 1321), by up to 0.56. Rule 2 says
   LTX drift for Han and Leia (keyframe -> last drop 0.28, 0.24) and klein/sheet for Obi-Wan
   (keyframe to sheet 0.41 < 0.45). **[Observed]**
3. **The face swap works on what it can reach.** HyperSwap (screenquote's engine, sheet face as
   source, most sheet-like face per frame, ArcFace >= 0.10) swapped 869 of 1,414 single-person
   frames and touched no helmet, droid or Wookiee (crop audit, `scores/swapped/crops.jpg`).
   Across-shot similarity rises +0.12 to +0.15 pooled for Han and Obi-Wan, above the real film's.
   **By eye it is modest**: eyes, nose and mouth converge (1285 and 1287 read as one man), but
   hair and head shape stay as LTX drew them, so Han's hair still varies shot to shot. That is the
   limit screenquote 0028 §1 predicted for an inner-face swapper. It cannot fix the 1294 merge.
   **[Observed]**, **[by eye]**
4. **Rule 3 was overridden by the task.** The pre-registered rule said try the swap only if the
   source ceiling were clearly higher (it was not: 0.27). The task asked for it whenever the score
   was well under 0.55, and it was free, so it ran. Reported as a deviation.
5. **Chewbacca is still an ape** with `18e2855`'s rewritten block (every Chewbacca shot in the
   contact sheet). Leia's covered white gown did take. **[by eye]**
6. **Found on the way:** a pod whose second status read says `"runtime": null` crashed
   `create_pod` with `AttributeError` (attempt 1 of this run); `capped_run` restarted it. Fixed:
   it is now a retryable `PodSetupError` (`runpod_pod.py`, test in `tests/test_runpod_pod.py`).

### Cost change for the full film (2,070 shots)

| Change | RunPod | Other | Notes |
|---|---|---|---|
| `--keyframe-mixed-shots` | +266 klein images x ~10 s = ~44 A40-min, **~+$0.46** (inferred) | - | not recommended (rule 1) |
| HyperSwap post-pass, Mac CPU | $0 | **~50 CPU-hours** (inferred: 1.6 s per single-person frame here, x ~115k person frames) | single-person shots only; multi-person shots need a face-to-character assignment step |
| HyperSwap post-pass on a GPU pod | ~1-2 pod-hours, **~$1** (inferred, onnxruntime-gpu, not measured) | - | needs a small integration |

Licence: HyperSwap 1b's weights are facefusion's research / non-commercial ones (screenquote ADR
0012). An unlisted, unmonetised YouTube upload is non-commercial, but it is Will's call.

### Recommendation for 30 Nov

**GO on 0030's chosen setup (low res + native audio, ~$23), with `--keyframe-mixed-shots` off.**
Nothing here needs the render to change, and the one lever that helped (the swap) runs on the
finished clips afterwards. Cost change: **$0** for the run itself; the swap adds ~$1 of GPU or
~50 h of Mac CPU later. If Will wants consistency fixed *before* publishing, the next experiment
is the swap at film scale on a pod, then a whole-head swap (screenquote 0028's untested Klein +
BFS head LoRA) for hair and head shape. Lowering LTX's drift (stronger first-frame conditioning,
shorter clips) is the other lever, but it would mean re-rendering, so test it on this window
first (~$0.60), before 30 Nov if possible.

### Where things are (job workspace `evidence/`, outside git)

- `side_by_side_source_old_new.mp4`: source | old | new, 2 min, new window's audio
- `side_by_side_old_vs_new_swap.mp4`: old | new + face swap
- `new_window_lowres_native.mp4` (new window, 1280x704, native audio + recast speech);
  `swap/new_window_swapped.mp4` (+ `.json`, per-shot swap counts)
- `identity_grid.jpg` (each character's face per shot: source, old, new, new + swap);
  `contact_sheet_new.jpg`; `scores/<version>/{candidates,labels,score}.json` and `crops.jpg`
- `run/` logs (speech, video, stitch), `hygiene_final.txt`, `dryrun_mixed_window.txt`

Not copied to `output/sw4-window-test-20261123/` in the primary checkout: workspace rules forbid
writes there. One `cp` from the workspace evidence dir puts them beside 0030's clips.

### Code added

- `decode.py --keyframe-mixed-shots` (with `runpod-ltx25-keyframe`): `plan_route(..., mixed=True)`
  keyframes a shot holding people and design characters; `ShotRoute.designs` carries the design
  characters; the editor prompt names them as sheet-less and adds any block the identity cap
  dropped. Off by default; tests in `tests/test_keyframes.py::TestMixedShotRouting`.
- `runpod_pod.create_pod`: a null runtime after boot is a retryable setup failure.
- Research tools here: `score_window.py` (labelled face scorer), `identity_grid.py`,
  `window_mixed.sh` (the run), `face_swap_trial.py` (the swap, run with screenquote's venv).
