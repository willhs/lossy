---
id: "0022"
type: research
purpose: "Verify whether lossy's decode/stitch/eval path works for hand-authored (non-source-film) manifests, before writing ~20 shots for the lumpy-blue-men short film against a guessed format."
scope: ["decode", "stitch", "eval", "authored-manifests"]
non_goals: ["source-film reconstruction quality", "video generation model comparison"]
tags: ["research", "decode", "stitch", "eval", "authored-manifest", "lumpy-blue-men"]
related: ["design/adr/009-authored-manifest-retiming.md", "design/adr/002-stateless-cli-pipeline.md"]
---

# Authored Manifest Verification

## Question

lossy was built to reconstruct an existing film: `encode.py` turns a source video into `shots.json`/`characters.json`, then `decode.py`/`stitch.py` turn those back into video. The lumpy-blue-men short film has no source film -- `shots.json` and `characters.json` are hand-authored from a screenplay, and `encode.py` is never run. Does the decode/stitch/eval path actually work for a manifest that never went through encode, or does it silently assume encode-only invariants that break?

## Context

This is the first task of the lumpy-blue-men film (Steady goal `e2f6faab`), split out specifically so the ~20-shot manifest for the real film is written against a format we've *proven*, not guessed. `lumpy-blue-men/docs/design/architecture.md` already listed four suspected leaks based on reading the code, unverified. This doc verifies each by hand-writing a throwaway 2-3 shot manifest and actually running it through the pipeline.

## Scope

- **Investigate**: whether `decode.py`, `stitch.py`, and `eval.py` work correctly against a hand-authored `shots.json`/`characters.json` with no corresponding encode output.
- **Out of scope**: video generation quality, the two-cut soundtrack mux (ADR-0002, a lumpy-blue-men-side concern), the real lumpy-blue-men shot content.

## Findings

### Setup

Hand-wrote a 3-shot v2 `shots.json` + 1-character `characters.json` (gnome-warrior entering a cave -- content is a format probe, not film material) at `tests/fixtures/authored_example/`. Copied into a fresh scratch output dir and ran the real pipeline, no mocks:

```bash
python decode.py /tmp/lossy-authored-probe --strategy fal-seedance
python decode.py /tmp/lossy-authored-probe --strategy fal-seedance --stitch
```

`fal-seedance` chosen as the cheapest strategy with no pod-lifecycle overhead (~$0.02/s, hosted API, no RunPod bring-up). Total spend: **$0.22** for 3 clips (10s of combined `duration_s`, fal-seedance's 2s-minimum-per-clip billing floor is most of that -- a bit more than "a few cents" but still trivial).

### Result: decode and stitch both worked with zero errors, first try

```
Generating 3 clips via fal-seedance (0 already done)...
  Shot 0 (1/3 remaining)...
  Shot 1 (2/2 remaining)...
  Shot 2 (3/1 remaining)...

Done. Generated 3 clips.
  Total: 3 completed, 0 failed
  Estimated cost: $0.22
```

```
Stitching 3 clips (3 shots) for full reconstruction...
  Saved to /tmp/lossy-authored-probe/lossy-authored-probe_reconstructed_fal-seedance.mp4
  Original duration: 10.0s (0.2min)
  Clips used: 3
```

`ffprobe` confirms a valid 10.0s h264 mp4, exactly matching the manifest's total `duration_s` -- retiming (see below) hit the target precisely. No diagnostic tripwire warnings fired (`stitch.py:529-544`, `:637-641` -- these only print if corrected duration deviates from `shots.json` by >1 frame).

### 1. Encode-derived fields (`camera_motion_detected`, `audio_detected`, `temporal_segments`) -- REFUTED, not a leak

Grepped every read site in `decode.py`/`stitch.py`/`strategies_video.py`/`prompt_format.py`:

- `camera_motion_detected` -- **never read** outside `manifest.py`'s doc comment and `encode.py` (where it's written). Decorative on the decode side.
- `audio_detected` -- **never read** by decode/stitch. Only reader anywhere is `tools/eval_audio_captioning.py:154`, via `shot.get("audio_detected", {}).get("bucket", "")` -- already tolerant of absence.
- `temporal_segments` -- read via `.get()` in `strategies_video.py:214,746`, and only consulted for multi-part (split) shots. Absence falls back to `vary_prompt_for_part`'s generic per-part prompt variation cues (`prompt_format.py:152-166`). No error path.

The throwaway fixture omits all three fields entirely and decode ran clean. Added `tests/test_decode.py::TestFalSeedanceGenerateMissingEncodeFields` to lock this in: one test exercises a multi-part split (`duration_s=15.0`, forces `[12, 3]`) with `temporal_segments` absent, confirming the `vary_prompt_for_part` fallback path runs without error; another exercises a single-part shot with none of the three fields present at all.

**Conclusion: no code change needed.** This suspected leak doesn't exist -- decode/stitch were already field-absence-tolerant by construction (every read of an encode-derived field uses `.get()` or isn't read at all).

### 2. `eval.py` -- CONFIRMED, real bug, fixed

`eval.py`'s `video` subcommand assumes the *original* encode artifacts exist:

```python
# eval.py:788-789 (before fix)
shot_manifest = load_json(manifest.shot_index_path(str(output_dir)))
prompts, _ = manifest.load_shots(str(output_dir))
```

Running `eval.py video` against the authored-manifest probe dir (which has `shots.json` but never had a `shot_index.json`, since that's an encode-stage-1-only artifact) reproduced the predicted crash exactly:

```
Traceback (most recent call last):
  File "eval.py", line 849, in <module>
    main()
  File "eval.py", line 788, in main
    shot_manifest = load_json(manifest.shot_index_path(str(output_dir)))
  File "eval.py", line 46, in load_json
    with open(path) as f:
FileNotFoundError: [Errno 2] No such file or directory: '/tmp/lossy-authored-probe/shot_index.json'
```

**Fix** (`eval.py:786-798`): wrap the two original-artifact loads in `try/except (FileNotFoundError, ValueError)`, print a message explaining this is expected for an authored manifest, and `sys.exit(0)` -- a clean skip, not a failure exit code.

```
No original encode artifacts found (shot_index.json/shots.json) -- skipping eval.
This is expected for an authored (non-source-film) manifest, which has nothing
to compare against.
```

Verified: `eval.py video /tmp/lossy-authored-probe --strategy fal-seedance` now exits 0 with that message instead of a traceback. Added `tests/test_eval.py::TestVideoEvalNoOriginal` (2 cases: `shot_index.json` missing with `shots.json` present, and both missing).

### 3. No first-class entry point -- CONFIRMED as a documentation gap, not a code gap

`manifest.load_shots` (`manifest.py:187-205`) only validates the top-level envelope (`{"format": "v2", "shots": [...], "dialog": [...]}`); nothing checks that an encode step ever ran. The authored-manifest path already worked today -- it was an undocumented side effect of the stateless-CLI-stages design (ADR-002 already names "easy to manually intervene between stages" as an intended consequence), not a missing capability.

**Fix**: no new `decode.py` flags (would add risk to the source-film path for zero behavioral gain -- decode doesn't need to know the difference). Instead: a "Authored manifests (no source film)" section in `README.md` documenting the minimum shape, plus `tests/fixtures/authored_example/` as a committed worked example other than a `/tmp` scratch dir.

### 4. Timing/retiming -- real design question; decision: keep as-is

`stitch.py`'s `_retime_shot_clips`/`_retime_clip` (`stitch.py:285-412`) boundary-locks each shot's generated clip to `entry["end_s"] - running_pos`, using ffmpeg `setpts`(slow down)/trim(speed up) to force the clip to match. For source-film mode, `duration_s` was *observed* from the original edit. For an authored manifest, `duration_s` is *chosen* by the screenwriter/author.

The probe run's stitched output landed at exactly 10.0s (the sum of the three shots' `duration_s`) with the tripwire warnings silent, confirming retiming works identically regardless of where `duration_s` came from.

**Decision** (see ADR-009): keep retiming as-is for authored manifests. An authored duration still represents the *intended* on-screen length -- for lumpy-blue-men specifically, shot timing needs to hit story beats and sync to the two-cut soundtrack (ADR-0002 in the lumpy-blue-men repo). Forcing generated clips toward that duration is the same job retiming already does for observed durations; there's no meaningful distinction between "observed" and "authored" at stitch time, only where the number came from upstream. The alternative (honor the clip's native generated length, let total runtime drift) trades sync accuracy for avoiding `setpts` motion-speed artifacts, and was rejected -- see ADR-009 for the full alternatives analysis.

## Conclusions

Of the four suspected leaks from the task brief:

| Suspected leak | Verdict | Action taken |
|---|---|---|
| Timing/retiming | Real, but current behavior is correct for authored manifests too | Documented decision; ADR-009 |
| `eval.py` errors with no original | Confirmed bug | Fixed: clean skip, `sys.exit(0)` |
| Encode-derived fields (`camera_motion_detected`/`audio_detected`/`temporal_segments`) | Refuted -- never an issue | No code change; regression tests added |
| No first-class entry point | Real gap, but documentation-only fix | README section + fixture example |

**The authored-manifest path works today with a single code fix (`eval.py`).** decode.py and stitch.py needed zero changes -- they were already tolerant of everything an authored manifest omits. The lumpy-blue-men shot list can be written directly against the `shots.json` v2 schema documented in `README.md` and exercised by `tests/fixtures/authored_example/`.

## Recommendations

- Write the lumpy-blue-men shot list against the schema in `README.md`'s "Authored manifests" section; use `tests/fixtures/authored_example/` as a structural template.
- No further lossy code changes are needed before authoring begins.
- If lumpy-blue-men later wants clip motion untouched by `setpts` retiming (e.g. a shot where speed-adjustment artifacts are visually unacceptable), revisit ADR-009's rejected alternative rather than silently special-casing one shot.
