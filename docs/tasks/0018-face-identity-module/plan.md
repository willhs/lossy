---
id: plan-0018
type: spec
purpose: "Implementation plan for one face-identity module replacing five research copies."
tags: ["plan", "faces", "refactor"]
related: ["./task.md"]
---

# 0018 face-identity module: implementation plan

Vision alignment: Have an online presence (goal: complete the Star Wars IV reconstruction and publish it).

## Overview

**Primary goal**: one place decides "which face in this frame is this character"; every caller
uses it; 0034's scores are unchanged.

**Approach** (refactor playbook, size L): pin 0034's behaviour with an old-vs-new run on real
images, add `faces.py`, migrate every caller and delete each old copy in the same phase, then
prove the pin on the real images again.

## Current state analysis

See task.md's table (verified 28 Nov against main after abdfad4). Key discoveries:

- `score_window.py` labels index candidates **in YuNet detection order after the 40 px filter**
  (`labels.json {"shot/pos": k}`), so `find` must keep detection order; sorting would silently
  re-point every existing label. 0029's area sort is a caller rule on top.
- 0034's `score` takes the sheet's `found[0]`; `look_crops.py` takes the sheet's largest by height;
  the swap takes largest by area (SCRFD faces, a different type). Head-and-shoulders sheets hold one
  face, so `sheet_face = largest` is the same pick; the pin checks it on real sheets.
- `strategies_video.py:2019` calls `embedder.preflight()`; `decode.py:873` catches
  `KeyframeSetupError` for it.
- `tools/face_swap.py` and `face_swap_trial.py` run under **screenquote's venv** (onnxruntime), so
  `faces.py` may import only cv2, numpy and the stdlib.

## Desired end state

```python
import faces
fi = faces.FaceIdentity.from_env()            # LOSSY_FACE_MODELS, default models/face
found = fi.find(img, min_h=40)                # list[Face], YuNet detection order
face = faces.pick(found, labels, "1306/first")  # labelled by eye, else largest; None = dropped
hit = faces.most_like(found, sheet_emb)       # (Face, cosine) or None
engine = faces.load_swap_engine(models_dir)   # LOSSY_FACE_SWAP_ENGINE -> screenquote OnnxEngine
```

## What we're NOT doing

- Swap as a pipeline stage; vendoring screenquote's engine; onnxruntime dependency.
- Changing 0029/0030 recorded rules (they keep them, on the shared detector).

## Assumptions

- Module is top-level `faces.py` (the repo is flat: `keyframes.py`, `ingredients.py`).
- `Face.box` is `(x, y, w, h)` ints, `Face.emb` a unit float32 vector; 0034's JSON keeps its
  rounding (score to 3 places) in the script, not the type.
- `FaceModelsMissing` is the module's error; the keyframe strategy turns it into
  `KeyframeSetupError` at its one preflight call, so `faces.py` does not import `keyframes.py`.
- The swap engine is "pinned" by recording the screenquote commit it was checked against
  (`1e09a04`) and checking the loaded module has `OnnxEngine` with `detect/embed/swap_aligned`;
  an unset env var is an error naming the variable (no default path into another repo).
- `frame_at` (ffmpeg grab at a time, scaled) moves into `faces.py` with `EDGE`/`POSITIONS` because
  `look_crops.py` needs it and must not import 0034.
- `window_faces.py` keeps "first face": it reproduces 0030's published table; AGENTS.md already
  points new scoring at `score_window.py`.
- The 0035 window data is on the Mac only: the pin runs on the 7 SW4 reference images in
  `~/lossy-runs/star_wars_iv_v2/characters/` and a synthetic window video made from them.

---

## Phase 1: Pin

- [ ] Write `evidence/pin/` script that runs `score_window.py` from `c2e7be4` (old) and the
      working tree (new) `candidates` + `score` on a synthetic window (fake run dir, video of the
      SW4 reference images, labels with picks and nulls) and byte-compares the JSON; plus old
      `OpenCVFaceEmbedder` / 0029 `Faces` vs new on the same images.
- [ ] Run it before any change: old == old (the harness is deterministic).

## Phase 2: `faces.py` + tests

- [ ] Create `faces.py`: `Face`, `FaceModelsMissing`, `FaceIdentity(find, faces, preflight,
      from_env)`, `largest`, `by_area`, `pick`, `most_like`, `signature`, `sheet_face`, `cosine`,
      `frame_at`/`EDGE`/`POSITIONS`/`WINDOW_SIZE`, `SwapEngine`, `SwapEngineUnavailable`,
      `load_swap_engine`.
- [ ] Create `tests/test_faces.py`: fake YuNet/SFace (order kept, min_h filter, unit length, none
      found, undecodable png); pick rules; most_like; signature; loader (unset, missing file, no
      OnnxEngine, ok); preflight; real-model test skipped without `LOSSY_FACE_MODELS`.

## Phase 3: migrate callers, delete copies

- [ ] `keyframes.py`: delete `OpenCVFaceEmbedder`; `_signature` uses `faces.signature`.
- [ ] `strategies_video.py`: `faces.FaceIdentity.from_env()`; preflight converts `FaceModelsMissing`.
- [ ] `tests/test_keyframes.py`: drop the moved test and import.
- [ ] `score_window.py` (0034): `faces.FaceIdentity`, `faces.pick`, `faces.sheet_face`, `faces.frame_at`.
- [ ] `look_crops.py` (0035): import `faces`, not 0034.
- [ ] 0029 `score.py`, `score_keyframes.py`: delete `Faces`; `faces.by_area(fi.find(img))`.
- [ ] 0030 `window_faces.py`: `faces.FaceIdentity`.
- [ ] `tools/face_swap.py`, 0034 `face_swap_trial.py`: `faces.load_swap_engine`.
- [ ] grep: no `FaceDetectorYN.create` outside `faces.py`, no `/Users/will/projects/screenquote`
      path in code, no research `sys.path` into another research dir.

## Phase 4: prove and document

- [ ] Run the pin: old == new for candidates.json, score.json, embedder and 0029 picks.
- [ ] `.gent/rules.md`: `faces.py` entry; face_swap run line with `LOSSY_FACE_SWAP_ENGINE`.
      README: `LOSSY_FACE_SWAP_ENGINE`.
- [ ] Run: `uv run python -m pytest -q` — all pass.

## Phase 5: Wrap

- [ ] Commit (job branch; Manager lands it).

## References

- Task: `docs/tasks/0018-face-identity-module/task.md`
- Research 0034 `score_window.py` (reference rule), 0035 research.md (data on the Mac), abdfad4.
