---
id: 0026-seinfeld-s01e03-first-end-to-end-run
type: note
purpose: "Record the first end-to-end run (Seinfeld S01E03, 1080p NAS source, runpod-ltx25): what was built to get it going, what it cost, how fast it ran, and what the output looks like."
scope: ["research", "encode", "decode", "stitch", "cost", "gemini", "vertex", "runpod"]
tags: ["research", "end-to-end", "seinfeld", "ltx-2.5", "vertex", "pgs", "spend-cap", "quality"]
related:
  - "research/0025-video-model-refresh-nov-2026/research.md"
  - "research/0024-full-run-checkpoint/research.md"
---

# Seinfeld S01E03: first end-to-end run

Vision alignment: Complete a movie or tv show Reconstruction (Star Wars IV headline run, with a control title)

Steady task `fb899084`. Will's go (7 Nov): overnight, hard US$8 cap on RunPod, no pod left
running, nothing published. Output stays local.

## Result

| | |
|---|---|
| Output | `output/seinfeld_s01e03/seinfeld_s01e03_reconstructed_runpod-ltx25.mp4` (gitignored, local only) |
| Length | **1382.48 s (23.04 min)** vs the 1382.42 s source: all 315 shots. Largest single-shot drift 0.087 s (about 2 frames at 25 fps), sum of drifts 0.02 s. 1280x704, 25 fps, h264 + aac. |
| Audio | per-character speech (419 lines, 4 voices + narrator) + MMAudio ambience (281 shots generated, 34 speech-only shots skipped, 0 failed) |
| RunPod spend | **$9.21 in total**: $7.2776 for the first run (shots 0-298, no ambience) + **$1.9325** for the finish job (video 299-314 $0.69, ambience $1.24; cap $2.50). Balance $7.3799 -> $5.4474. |
| Other spend | fal.ai speech $0.75, Gemini encode about $0.17 |
| Pods left | none: `myself{pods{id}}` is `[]` and `currentSpendPerHr` is 0 after the finish job |
| Pods left | none: `myself{pods{id}}` returned `[]` after the final terminate |
| Pace | about 2.3 min per shot over the run (299 shots in about 11.2 h incl. 20 min boot), about $0.024 per shot |

Stills: `contact_sheet.jpg` (5 frames from the first, partial cut, at 45, 330, 640, 940 and 1210 s) and
`contact_sheet_final.jpg` (8 frames from the finished episode, at 45, 330, 640, 940, 1210, 1270, 1330 and
1380 s; the last one is the ending, a lighthouse at dusk).

## Quality: style drifts between shots

The contact sheet shows the biggest problem. Within one episode the look changes shot to shot:
photoreal live action in some shots, **3D-cartoon animation in others** (2 of the 5 stills). The
describe pass has no field that pins the rendering style, so the model picks one per shot.
One still also lands on a recognisable look from a different well-known show. Both are prompt-level
problems, not model-capacity ones; a fixed style clause in `prompt_format.py` is the obvious
first fix. [measured on 5 stills; not a full-episode survey]

## What it took (and what got built)

1. **Gemini was 429ing.** Every call on both encode models failed for 15+ min (project-wide
   `api_requests` per-minute/region limit on `gen-lang-client-0146943418`; it carries a
   Google-side quota override, so model choice does not help). Vertex AI on the same project has
   its own quota and works. Added `LOSSY_GEMINI_BACKEND=vertex` (+ `LOSSY_ENCODE_MODEL`);
   Vertex serves `gemini-3.1-flash-lite` and `gemini-3.5-flash-lite`, **not** the pinned
   `gemini-2.5-flash-lite` or the `*-preview` ids. This run used `gemini-3.1-flash-lite`, a model
   change from the July pin, so descriptions were not compared to the pinned model. The 315
   descriptions contain no character names.
2. **Shot detection differs by source.** The default adaptive detector gives 212 shots on the
   1080p BluRay (24 fps); the April 343-shot count came from the 480x360 copy. `-d content`
   gives 315 shots, mean 4.39 s, 84 over 4.84 s. That is what this run used.
3. **BluRay subtitles are bitmap (PGS).** `ffmpeg -map 0:s:0 out.srt` cannot convert them, so
   stage 2 would have had no dialogue (the April encode had none). `tools/pgs_to_srt.py` renders
   the track over black and OCRs each distinct subtitle with tesseract: 419 cues, timings good,
   text mostly clean. Tesseract reads capital "I" as "|"; mapped back. Some residual OCR noise is
   likely.
4. **Voice casting.** The proposal cast Kramer with a female voice and Elaine with a British
   accent; both were hand-changed in `voice_map.json` (Kramer -> Callum, Elaine -> Matilda).
   **Casting is still Will's call**; nobody has heard the samples.
5. **Spend cap.** `decode.py` has no spend limit. `tools/capped_run.py` measures spend from the
   RunPod balance, stops at a hard line (cap minus 0.3), retries after a pod failure and
   terminates any pod it did not find at start. A first version projected total spend from shot
   count and a second from video-seconds; both over-estimate early on because they ignore the
   fixed cost per shot (about 46 s + 16 s per video-second, fitted from clip timestamps), so the
   run used `--no-project` and the spend line.

## The stall at the end

At 12:41 the last clip landed. The decode then sat for 20 minutes polling one ComfyUI job on
the first pod (A6000, $0.59/h) while **also provisioning a second pod** (L40, $0.69/h, same
session name) and leaving the first running. Spend ran at about twice the normal rate with no
output, so the run was stopped at 13:02 and both pods were terminated. The progress file also
lagged: it is written in batches, so shots 295-298 had finished clips on disk but were not
recorded, and the first stitch mis-assigned their parts (shots 296 and 298 came out 5.6 s and
10.6 s long). The four shots were added to the progress file from the clips on disk and the
re-stitch came out clean (largest drift 0.045 s). [measured]

**Open bug:** a hung ComfyUI poll has no timeout, and re-provisioning does not terminate the
pod it replaces.

## Cost, and why there is no ambience

(Written after the first run; the finish job below did run the ambience pass.)
Video alone took $7.25 of the $8 cap at 94.8% of shots. The ambience pass (RunPod MMAudio,
about $0.8 in the research 0025 projection) would have crossed the cap, so it was not run.
The 0025 projection of about $4.50-7 all-in under-called this run by roughly $1-2: the fit
above says the A6000 secure rate and 2.3 min/shot, not the 8 GPU-hour figure, are the real
numbers for LTX-2.5 at 1280x704 on this episode.

## Finish job (8 Nov): shots 299-314, ambience, re-stitch

Will approved about $2 more, hard cap $2.50 from a $7.3799 balance, run under `capped_run`
(`--no-project`, `--margin 0.3`, `--baseline` fixed at the starting balance so both stages count
against one cap).

| Stage | Result | Spend |
|---|---|---|
| Video, shots 299-314 | 16 shots, 1 pod (L40 at $0.69/h; the A6000 was unavailable), 1.0 h in all: about 14 min boot, then 49 min of clips | $0.69 |
| Ambience (RunPod MMAudio) | 281 clips, 34 speech-only shots skipped; two pod sessions on an L40S at $0.54/h (the first one aborted, see below) | $1.24 |
| **Total** | cap $2.50, stop line $2.20 | **$1.9325** |

- **Resume worked once the progress file's mtime was right.** The first run's folder was copied into
  this worktree; `cp -R` reset mtimes, so `check_encode_fingerprint` saw a progress file older than
  `shots.json` and refused it. Restoring mtimes (`touch -r`) fixed it. The progress file lagged by
  nothing this time (it already held 0-298, including the four shots added by hand).
- **The ambience pass aborted at shot 199 with "Too many errors"** after 21 speech-only shots. A
  shot whose sound description is all speech makes `RunPodMMAudioStrategy` / `MMAudioStrategy`
  return `[]` on purpose; `run_audio` counted that as a failure, slept 10 s, retried, and stopped
  after 20. Fixed: strategies that filter speech set `filters_speech = True` and `run_audio` records
  those shots as skipped before calling them (test `TestRunAudioSpeechOnlyShots`). The pod was
  terminated on exit; the resume paid a second boot (the two sessions billed about $0.45 and $0.36 for 179 and 102 clips).
  This would have hit any episode with 21+ dialogue-only shots, so Star Wars IV would have hit it too.
- **Per-shot length check.** All 315 shots have an adjusted clip; the largest deviation from
  `shots.json` is 0.087 s, and the stitch's own drift warning fired for 102 shots, all within about 2
  frames. The muxed audio is 0.044 s shorter than the video (1382.436 s vs 1382.480 s).
- **Pace.** Video 16 shots in 49 min after boot (about 3 min per shot, a little slower than
  the first run's 2.3; these tail shots are longer on average). Ambience about 4 shots per minute.
- **Quality.** The style drift reported above continues in the tail: `contact_sheet_final.jpg` has 3D
  cartoon shots (330 and 940 s) next to photoreal ones, and the stills at 1210-1330 s look like a
  different well-known drama rather than the source show. Not fixed here; the style clause is still
  the first thing to try before a second episode. [8 stills only, not a survey]
- **Ambience level.** Mean loudness of the finished mix is -18.5 dB with peaks at 0 dB; nobody has
  listened to it yet.
- **Not run:** a hung ComfyUI poll was not seen this time (no stall), so the open bug above is
  untouched.

The partial first cut is kept as `partial_shots0-298_no_ambience.mp4` next to the finished file.
