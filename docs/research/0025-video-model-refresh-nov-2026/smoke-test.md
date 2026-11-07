---
id: 0025-smoke-test
type: note
purpose: "Record the live smoke test of the new runpod-ltx25 strategy: setup, 8 shots, chaining seam ratio, spend, and the two bugs it caught."
scope: ["decode", "ltx-2.5", "runpod", "chaining", "smoke-test"]
tags: ["ltx-2.5", "runpod-ltx25", "i2v-chaining", "seam-ratio", "smoke-test"]
related:
  - "research/0025-video-model-refresh-nov-2026/research.md"
  - "research/0023-ltx2-dress-rehearsal/research.md"
---

# runpod-ltx25 smoke test

Vision alignment: On a new endeavour — Want to reach my potential with software

Follow-up to the "Work needed before the run" list in `research.md`. The strategy is
`RunPodLtx25Strategy` (`strategies_video.py`), selected with `decode.py --strategy runpod-ltx25`.

## Result

**Works end to end.** 8 Star Wars IV shots (5 single-part, 3 two-part), 11 clips, rendered through the real
`decode.py` path on a fresh pod. Spend **$0.68** of the $7 allowance (RunPod balance
$15.3464 -> $14.6630). `runpod.get_pods()` returned `[]` afterwards.

| Check | Result |
|---|---|
| ComfyUI upgrade | stock image v0.30.0 -> v0.39.2 by tag, verified by reading `comfyui_version.py` |
| Weights | 5 files from the `comfyicu/LTX-2.5` mirror at a pinned revision, each `sha256sum -c` verified before being moved into place |
| Output | every clip 1280x704, 25 fps, with a native audio stream |
| Frame counts | 57, 89, 89, 89, 65 on the single-part clips: all `% 8 == 1` |
| Pod | RTX A6000 had no capacity in either cloud; the chain fell through to RTX 6000 Ada at $0.74/hr |

## Chaining seam ratio (vs research 0023)

Same method as 0023, via the new `tools/seam_ratio.py`: mean abs gray difference between the true last
frame of part N and first frame of part N+1, divided by normal frame-to-frame motion. Shots 818, 1061, 1576.

| | seam | normal motion | ratio |
|---|---|---|---|
| 0023 LTX-2.3 unchained | 44.6 | 5.24 | 8.5x |
| 0023 LTX-2.3 chained | 10.4 | 4.67 | 2.2x |
| **LTX-2.5 unchained** (`LOSSY_CHAIN=0`) | 35.4 | 5.57 | **6.4x** |
| **LTX-2.5 chained** | 2.4 | 5.59 | **0.43x** |

Chaining works on 2.5, and the seam is below a normal frame step. Caveats:

- Only 3 seams, and different shots from 0023, so this is a smoke result, not a survey.
- A ratio under 1 means the boundary frames are nearly identical (the next part is conditioned on the last
  frame). It does not measure the motion ease-in/out hitch 0023 describes; that needs eyes on a long shot.
- Stage-2 re-pinning of the start frame (`REAPPLY_START_IMAGE_STAGE2`, on) was not A/B tested against off.

`evidence/contact_sheet_chained.png` (job workspace) shows one frame per clip.

## Bugs the smoke test caught

1. **Shared clip-saving bug.** `_attempt_one_clip` named the raw download `clip_path.replace(".mp4", ext)`.
   LTX-2.5's `SaveVideo` already returns `.mp4`, so the raw file *was* the clip; ffmpeg cannot convert a file
   onto itself, and the fallback then deleted the only copy. Every attempt failed, and retries hit ComfyUI's
   result cache. Fixed with a distinct `.raw` name; regression test in `tests/test_ltx25.py`.
2. **`LTXVImgToVideoInplace` needs `bypass`.** ComfyUI v0.39.2 rejects the node without the boolean. Only chained
   parts hit it, which is why single-part shots passed.

## Not covered

- Native audio quality (does it sync, vs the MMAudio ambience path): clips carry the audio, nobody has listened.
- Community-cloud pricing: no community stock during the test, so the cost per clip is the $0.74/hr Ada rate,
  and per-clip timings were not measured precisely (roughly 1-2 min per clip including download).
- The research-note cost-reporting bug (`RunPodSession` records the community rate after falling back to
  secure) is unfixed; the run's own "estimated cost" line under-reports.
- Nothing full-length was run.
