---
id: 0032-rate-distortion-protocol
type: note
purpose: "Pre-registered protocol for the words-per-second vs fidelity chart (research 0031 item 2): the fixed scene, budgets, seeds, baselines, the bit-rate measure and the scoring, plus the script that prepares it and a one-line run command. Desk work: no render, no paid call."
scope: ["research", "rate-distortion", "evaluation", "protocol", "ltx-2.5", "star-wars"]
tags: ["research", "pre-registration", "rate-distortion", "fidelity", "av1", "bits-per-second"]
related:
  - "research/0031-strategic-overview-after-sw4/research.md"
  - "research/0005-compression-analysis/research.md"
  - "research/0026-seinfeld-s01e03-first-end-to-end-run/research.md"
  - "research/0028-style-drift/research.md"
  - "design/adr/0010-pod-ownership-journal.md"
---

# The rate-distortion curve of language: protocol and script (22 Nov 2026)

Vision alignment: Have an online presence (goal: complete a movie or TV show reconstruction, Star
Wars IV headline run)

Steady task `439d83cd`. Spec: research 0031 item 2. **Nothing has been rendered and nothing paid
for.** This note fixes every choice before any clip exists, so the chart cannot be tuned to the
result. The code is [`tools/rd_curve.py`](../../../tools/rd_curve.py), tested in
`tests/test_rd_curve.py`.

## Answer

**Ready to run.** The pilot (30 clips, projected $0.72, capped at $2) needs only Will's go and a
funded RunPod account; the full chart is 300 clips, projected $7.20 (cap $11.50). Nothing has been
decoded, so there is no curve yet.

Run it (pilot only; the Gemini rewrite and the cheap offline steps come first, see "Run it"):

```bash
uv run python tools/capped_run.py --cap 2 --margin 0.5 --no-project --output-dir output/rd_curve -- uv run python decode.py output/rd_curve --strategy runpod-ltx25 --ab-manifest output/rd_curve/manifests/pilot.json
```

Two numbers are already real, because neither endpoint needs a rewrite (measured on SW4 shots
430-469, 99.85 s; see "What the stream is"):

| Variant | Words per shot | After xz | Bits per second |
|---|---|---|---|
| 0 words (timing plus the generic line) | 7 (the constant) | 476 B | **38** |
| Full (as encoded) | 124 | 8,192 B | **656** |

**Premise corrections.**
- **"Full" is about 124 words in this scene (143 across the film), not ~200.** So the spec's
  128-word budget would be "full" for most shots. The protocol uses **64** instead (see "Budgets").
- **The scene alone costs more bits per second than the film (656 vs the film's ~330-375)**,
  because 40 shots give xz little repetition to find; 0005's ~11x came from 2,070 shots. The
  x-axis is a within-experiment comparison, not a film-scale rate (see "Limits").
- **Identity blocks do not ride on the words.** `characters.json` assigns up to two verbatim
  identity blocks (about 50-70 words each) to a shot by shot index, whatever the text says. A
  "0-word" render with them on is not zero words, so the primary arms switch them off (see
  "Design choices").

## Fixed in advance

| Item | Choice |
|---|---|
| Film and encode | SW4, `output/star_wars_iv_v2` (2,070 shots, identity blocks regenerated 22 Nov, ea7b459) |
| Scene | Shots **430-469**: 40 consecutive shots, 99.85 s (source 1688.645-1788.495 s), the Tatooine desert scene: Luke, Obi-Wan, C-3PO, R2-D2, Chewbacca, a creature. 13 shots carry dialogue, 4 are longer than one clip. Chosen by reading the encode only, before any render, for: contiguous, one setting, people and two named droid designs, outside the tested 684-707. |
| Core subset | Every 4th scene shot: 430, 434, ..., 466 (10 shots). Pilot, seeds 2-3 and the identity arm use it. |
| Decoder | `runpod-ltx25` text-to-video, no keyframes (the keyframe arm adds an image channel that would break the words axis), style prefix on every arm |
| Style prefix | "Live-action 1977 science-fiction film footage of real actors and practical sets, photographed on 35mm film." (Will's 22 Nov choice for the full run). It is a decoder-side constant, identical on every arm and not counted in the bit rate. |
| Budgets | 0, 8, 32, 64 and full words (below) |
| Seeds | Seed 1 = the shot index (as a normal decode). Seeds 2 and 3 = index + 10,000 and + 20,000, on the 10 core shots only. |
| Baseline | AV1 of the same 99.85 s at 10 and 30 kbps, scored the same way |
| Fidelity | The item-1 tool (below). Primary measure: shot-level top-1 among N = 10. |

### Budgets

A budget is the **maximum total words across the eight fields the video prompt uses**
(`shot_type`, `camera_movement`, `subjects`, `action`, `lighting`, `color_palette`, `mood`,
`setting`). `sound`, `music` and `dialogue` feed the audio channels and are left as encoded in
every variant.

- **0:** every shot's description is the same constant, "a shot from a 1977 science-fiction film".
  What differs per shot is the timing, which sets the clip length. The constant is not counted as
  words but is in the stream, where xz makes it almost free.
- **8, 32, 64:** one Gemini rewrite of the *existing* description to at most N words, so the
  encoder's perception is held fixed and only the budget changes. Each budget is rewritten from the
  full text, never from another budget. The model must keep who and what is in frame, the shot
  type, the main action and the place before mood, palette and lighting. A rewrite that does not
  fit is retried three times with the count fed back, then **fails** (never truncated); the shot is
  left unwritten and the run exits non-zero.
- **Full:** the description as encoded. A description already at or under a budget is kept as
  encoded, so budgets saturate; the script reports the effective words per shot of each variant.
- **Why 64, not 128:** this scene averages 124 words, so 128 ~ full. 64 keeps five distinct points
  on a roughly log axis (0, 8, 32, 64, 124). This is the one change from the approved spec.

### What the stream is (bits per second)

For each variant the stream is, per shot, `{index, start_s, end_s, description: {the eight video
fields that are non-empty}}`, as minified JSON with sorted keys, compressed with
`xz -9e --check=none -T1` (Python `lzma`, one block, no integrity check; a test pins it to the
`xz` binary's byte count). Bits per second = 8 x compressed bytes / scene seconds.

- Timing is in every variant, so the 0-word point is the cost of the cut list itself.
- Not counted, same on every arm: `characters.json` (a per-film registry), the style prefix,
  `sound`/`music`, `dialogue` and the decoder's own weights.
- **AV1** is the measured size of the encoded file x 8 / 99.85 s, video only, never the target rate.
  The 10 kbps size can differ from its target.

### The AV1 baseline

Two-pass libaom of the scene, 360p, `-cpu-used 4`, no audio (commands from `plan --src`). Its
frames are then cut at the same source spans as the reconstructions and scored the same way.
The July post called 10 kbps the codec floor. 360p is a pre-registered choice (rate control at
10 kbps is stable there and the scorer resizes frames anyway); it is not tuned afterwards.

### Design choices

| Choice | Alternatives | Why |
|---|---|---|
| Identity blocks **off** for the primary arms | On for every arm (the real codec) | With them on, the 0-word arm still carries about 100 words per shot. Off isolates the words. The `identity` step re-runs the core at 0, 32 and full with them on, so the real configuration is also on the chart. Personal names are still stripped from the text (a decode strips them, and it keeps arms comparable), and the design names that `keep_name` keeps (Chewbacca, droids) stay in any text that has them. |
| Render through `--ab-manifest` | Rewritten `shots.json` in a decode dir | The manifest carries each prompt verbatim at its own seed, so arms cannot be reworded, and a resume skips finished arm/shot pairs. The rewritten `variants/<budget>/shots.json` are the bit-rate measure and the record, not decode inputs. |
| Seeds 2-3 on the core only | 3 seeds on all 40 | 300 clips ($7.20) instead of 600 ($14.40). Seed noise is read off 10 shots x 3 seeds at every budget. |
| Text-to-video only | Keyframe-first | One channel varies at a time. |
| One clip per shot | Chain parts for long shots | 4 shots are over 4.84 s (9.2, 7.1, 6.3 and 6.4 s). Each renders its first 121 frames only. The job's `span_s` is the source span the clip covers, and the score uses that span's midpoint. The bit rate still counts all 40 shots. |

## The fidelity interface (not duplicated)

Scoring is the tool from task `51077770` ("Lossy: fidelity score", item 1), planned as
`tools/fidelity.py`. It had not landed when this was written, so this section records only what
the chart needs from it; **check its real interface before the chart runs**.

- **Input:** for each clip, the reconstruction's midpoint frame and the source frame at the
  midpoint of the clip's source span, cut fresh from the video. Every manifest job carries
  `span_s: [start, end]` for that, because a long shot's clip covers less than `start_s`-`end_s`.
- **Output used:** top-1 among N = 10 at the three levels (same-scene neighbours = shot level;
  random shots from the same film = scene level; other films = film level), chance 10% each.
- **Same tool for AV1:** its frames go in as another set of 40 reconstructions.
- If the tool only reads a shots.json plus a clips dir, wrap it, do not fork it: build the scene's
  `shots.json` with `end_s` set to `span_s[1]`.

## Pre-registered analysis

- **Primary chart:** x = bits per second (log), y = shot-level top-1 (N = 10), one point per
  budget at seed 1 on all 40 shots, 95% interval by bootstrapping shots. AV1 10 and 30 kbps on the
  same axes. A second panel with the scene and film levels. A third with words per second on x.
- **Seed noise:** on the 10 core shots, the spread of the 3 seeds per budget, to say which gaps
  are real.
- **Identity arm:** the core at 0, 32 and full with identity on, plotted beside the off arms.
- **Reported whatever the shape.** If text never reaches AV1's fidelity at any budget, or the
  curve is flat above 32 words, that is the finding. No budget, shot or seed is dropped after
  seeing a score. A clip that fails to render is listed as failed, not replaced.
- **Guesses, written down now** [Speculative]: film level near ceiling at every budget (the style
  prefix alone does it); shot level near chance at 0 and rising steeply by 32; flat from 64 to
  full; AV1 above every text point on shot-level fidelity, but at 15-260 times the rate.

## Cost

Cost model from 0026 (46 s fixed + 16 s per video-second per clip, A6000 secure $0.59/h, 20 min
boot) and 0026's measured ~$0.024 a shot; a projection quotes the larger. The script reads the real
frame counts.

| Step | Clips | GPU-min | Model | At $0.024/clip | Projected | `capped_run` cap |
|---|---|---|---|---|---|---|
| pilot (core x 8, 32, full x seed 1) | 30 | 44 | $0.63 | $0.72 | **$0.72** | $2 |
| chart (40 x 5 x seed 1, core x 5 x seeds 2-3) | 300 | 423 | $4.35 | $7.20 | **$7.20** | $11.50 |
| identity (core x 0, 32, full, identity on) | 30 | 44 | $0.63 | $0.72 | **$0.72** | $2 |

The chart contains the pilot's 30 clips, which a resume skips, so doing the pilot first costs
nothing extra: pilot + chart + identity is about $8.64 projected for the 360 distinct clips. The
Gemini rewrite is 120 calls on this scene (40 shots x 8, 32 and 64 words; a shot already under a
budget would be kept as encoded), a few cents. The AV1 encodes are local ffmpeg, free. The cap is 1.5 x the projection + $0.50, rounded up
to $0.50; past forecasts have run 1.4-2.7x low (0031), so the cap is a ceiling, not a target.
Spend must go through `capped_run.py` with `--output-dir` the run dir. The account needs the pilot
amount free and nothing else using it (0030, RunPod sweep hazard).

## Run it

All paths relative to the repo root. Steps 1-4 are free or cents; step 5 is the paid pilot.

```bash
SRC=/Users/will/projects/lossy/output/star_wars_iv_v2; RUN=output/rd_curve
# 1. Gemini rewrite: 120 calls, cents. Drop --go to see the count first.
op-scope lossy -- env LOSSY_GEMINI_BACKEND=vertex LOSSY_ENCODE_MODEL=gemini-3.1-flash-lite \
  uv run python tools/rd_curve.py rewrite $SRC $RUN --go
# 2. variants, bit rates, manifests (offline, free)
uv run python tools/rd_curve.py build $SRC $RUN
# 3. print the decode commands, projected costs and the 4 AV1 ffmpeg commands; run the ffmpeg ones
#    (local, free), then re-run step 2 so bitrate.json lists AV1
uv run python tools/rd_curve.py plan $RUN --src $SRC
# 4. check the manifest builds real graphs (no pod, no spend)
uv run python decode.py $RUN --strategy runpod-ltx25 --ab-manifest $RUN/manifests/pilot.json --dry-run
# 5. the paid pilot (30 clips, projected $0.72)
uv run python tools/capped_run.py --cap 2 --margin 0.5 --no-project --output-dir $RUN -- uv run python decode.py $RUN --strategy runpod-ltx25 --ab-manifest $RUN/manifests/pilot.json
```

`rewrite` resumes (finished rewrites are skipped, one file write per shot) and `build` rewrites its
outputs from scratch, so both are safe to re-run. The chart and identity steps are the same step 5
command with `chart.json` / `identity.json` and the caps `plan` prints.

## Verified without a pod

On the real SW4 encode (read only), with the Gemini step replaced by a truncating stand-in:
`build` wrote all five variants and the three manifests; **`decode.py --ab-manifest --dry-run`
accepted all 360 jobs** (30 + 300 + 30) and built a valid ComfyUI graph for each. The word counts
and bit rates of the 8, 32 and 64 variants from that run are not results (truncation is not a
rewrite); only the 0-word and full endpoints above are.

## Limits

- **The measure is video-channel text only.** Dialogue and sound text are excluded, so these
  rates are not the 330-375 bit/s film figure of 0005, which included them.
- **Short-scene compression.** xz over 40 shots finds less repetition than over a film (656 vs
  ~330 bit/s at full). If the headline needs a film-scale rate, measure each variant's marginal
  bytes inside the full film's stream; not done here.
- **One scene.** The curve is for a desert dialogue scene. Another scene (action, space) could
  differ. A second scene is a follow-up, not part of this protocol.
- **Gemini rewrite quality is part of the result.** A different rewrite model or prompt moves the
  curve; the model id is recorded in each `rewrites.json`.
- **The fidelity tool is the other half.** It has its own limits (frame embedding at a midpoint
  of a moving shot) and is not fixed here.
- **Names in the text.** Design names (Chewbacca, C-3PO) survive in the text where the
  description has them, so some of the "words" are lookup keys for the decoder's memory
  (research 0021, item 3 of 0031 tests that).
