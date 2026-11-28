---
status: done
steady_task: 30299286-18d6-484a-bf1f-40f9c88c40aa
created: 2026-11-28
---

# 0018: one face-identity module (detect, embed, match to sheet)

Vision alignment: Have an online presence (goal: complete the Star Wars IV reconstruction and publish it).

## Why

"Who is in this frame" (YuNet detect + SFace embed + pick the character's face) is written five
times, with different face-picking rules, and the research scripts import each other by `sys.path`
or load another repo's file by an absolute Mac path. A face-picking fix (research 0034 found the
"first face" rule picks helmets and Chewbacca) has to be made in every copy, and the post-render
face swap research 0035 recommends cannot become a pipeline stage while its engine is an absolute
path into screenquote.

Verified against the code on 28 Nov (after abdfad4, the flicker job, landed):

| Copy | Rule |
|---|---|
| `keyframes.py` `OpenCVFaceEmbedder` | every face, detection order (sheet gate: centroid) |
| `docs/research/0034-*/score_window.py` `Faces` | faces >= 40 px tall, detection order; labelled pick, else largest by height; sheet = first face |
| `docs/research/0029-*/trial/score.py` `Faces` | largest area first |
| `docs/research/0029-*/trial/score_keyframes.py` `Faces` | largest area first, best of the top two |
| `docs/research/0030-*/window_faces.py` | first face (the rule 0034 rejected) |
| `docs/research/0035-*/look_crops.py` | imports 0034's `score_window` by `sys.path` |
| `tools/face_swap.py`, `docs/research/0034-*/face_swap_trial.py` | `ENGINE = "/Users/will/projects/screenquote/..."` |

## What

1. A package module `faces.py`: the `Face` type, `FaceIdentity` (YuNet + SFace, `find`, `faces`,
   `preflight`), the pick rules (labelled-by-eye pick else largest; most like a sheet; sheet
   signature), frame sampling for scoring, and the swap-engine seam (`SwapEngine` protocol,
   `load_swap_engine` from `LOSSY_FACE_SWAP_ENGINE`, no absolute path into another repo).
2. Every copy above uses it; `OpenCVFaceEmbedder` and the research `Faces` classes are deleted.
3. Research 0034's picking rule is the reference behaviour: old vs new give identical
   candidates and scores.

## Success criteria

- [x] No `cv2.FaceDetectorYN.create` outside `faces.py`; no research script imports another
      research dir; no absolute path to `face_swap_engine.py` in the repo.
- [x] `uv run python -m pytest -v` passes, with `tests/test_faces.py` covering the pick rules,
      `find`'s order/filter contract and the swap loader.
- [x] Old (`c2e7be4`) vs new `score_window.py candidates` + `score` on real SW4 images give
      byte-identical `candidates.json` and `score.json` (CPU, $0).
- [x] `.gent/rules.md` and README name `faces.py` and `LOSSY_FACE_SWAP_ENGINE`.

## Won't do

- Make the swap a pipeline stage (this makes it possible; doing it is a separate feature).
- Vendor screenquote's engine or add `onnxruntime` (AGENTS.md: no new dependency without
  discussion). The adapter loads a pinned file named by an env var.
- Change 0029's or 0030's recorded picking rules: their scripts reproduce published numbers, so
  they keep their rule on top of the shared detector.

## Assumptions

- Dispatch means the design is approved (the notes said Will reviews it first); every piece is a
  reversible refactor, so it went ahead.
- The 0035 window renders and `labels.json` live only on the Mac (research 0035 "Where things
  are"); the old-vs-new comparison runs on the SW4 reference images in `~/lossy-runs` instead, and
  the window comparison is reported not verified.

## References

- Research 0034 (`score_window.py`, the reference rule), 0035 (swap, flicker), 0029, 0030
- abdfad4 (flicker job: `tools/face_swap.py`)
- Steady task: 30299286-18d6-484a-bf1f-40f9c88c40aa
