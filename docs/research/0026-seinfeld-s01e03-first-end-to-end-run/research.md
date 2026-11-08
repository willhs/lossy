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
| Length | 1262.96 s (21.05 min) of the 1382.4 s source: shots 0-298 of 315 (91% of the runtime) |
| Missing | shots 299-314, the last ~2 min (the tail of the episode). Stopped at the cap, not by failure. |
| Audio | per-character speech (419 lines, 4 voices + narrator). **No ambience/music** (see Cost) |
| RunPod spend | **$7.2776** (balance $14.6575 -> $7.3799, settled a few minutes after the terminate), cap $8 |
| Other spend | fal.ai speech $0.75, Gemini encode about $0.17 |
| Pods left | none: `myself{pods{id}}` returned `[]` after the final terminate |
| Pace | about 2.3 min per shot over the run (299 shots in about 11.2 h incl. 20 min boot), about $0.024 per shot |

Stills: `contact_sheet.jpg` (5 frames of the reconstruction, at 45, 330, 640, 940 and 1210 s).

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

Video alone took $7.25 of the $8 cap at 94.8% of shots. The ambience pass (RunPod MMAudio,
about $0.8 in the research 0025 projection) would have crossed the cap, so it was not run.
The 0025 projection of about $4.50-7 all-in under-called this run by roughly $1-2: the fit
above says the A6000 secure rate and 2.3 min/shot, not the 8 GPU-hour figure, are the real
numbers for LTX-2.5 at 1280x704 on this episode.

## To finish the episode

- Resume decode for shots 299-314 (about $0.5-1.0 incl. a pod boot) and re-stitch.
- Decide whether to add a fixed style clause before spending more on the same prompts.
- Optionally run the ambience pass.
