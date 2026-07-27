---
id: adr-009-authored-manifest-retiming
type: decision
purpose: "Record the decision to keep stitch's clip retiming behavior unchanged for hand-authored (non-source-film) manifests."
scope: ["design", "architecture", "decode", "stitch"]
non_goals: []
tags: ["adr", "stitch", "authored-manifest"]
related: ["design/adr/002-stateless-cli-pipeline.md", "research/0022-authored-manifest-verification/research.md"]
---

# Context

`stitch.py` boundary-locks each shot's generated clip to `entry["end_s"] - running_pos` from `shots.json`, using ffmpeg `setpts` (slow down) or trimming (speed up) to force the generated clip to match (`stitch.py:285-412`, `_retime_shot_clips`/`_retime_clip`). This exists because source-film reconstruction needs generated clips to land on the *observed* shot durations of the original edit, so the reconstructed film's total runtime and cut timing match the source.

lumpy-blue-men has no source film. `shots.json`'s `duration_s` per shot is *chosen* by the author when writing the screenplay-derived shot list, not measured from anything. The question: should authored `duration_s` still drive retiming, or should stitch instead honor whatever length the video-generation model naturally produces (letting total runtime drift from the authored plan)?

# Decision

Keep retiming unchanged. Authored `duration_s` is treated identically to observed `duration_s` -- both are read the same way by `stitch.py`, and no code distinguishes "this number came from an author" vs "this number came from PySceneDetect."

Verified in `docs/research/0022-authored-manifest-verification/research.md`: a 3-shot authored manifest run through `decode.py --strategy fal-seedance --stitch` produced a 10.0s output matching the sum of authored `duration_s` values exactly, no retiming-tripwire warnings fired.

# Consequences

**Positive:**
- Shot timing in the authored manifest is a reliable contract -- an author who writes `duration_s: 3.5` for a shot gets a 3.5s shot in the output, which lumpy-blue-men needs for beat-by-beat pacing and the two-cut soundtrack sync (lumpy-blue-men ADR-0002).
- Zero code change -- no new branch, no authored-vs-observed flag to thread through decode/stitch, no added risk to the source-film path (goal `b33ea4e0`, Star Wars IV reconstruction).
- decode.py already passes `duration_s` straight through to the generation strategy as `target_duration_s` regardless of provenance (`decode.py:199,210,333,343`); retiming at stitch time is just a second correction pass for whatever gap remains between the generated clip's actual length and the target. That gap exists identically whether the target came from an author or an encoder.

**Negative:**
- `setpts` speed-adjustment can introduce visible motion-speed artifacts (unnaturally fast/slow movement) when the generated clip's native length differs significantly from the authored target -- most noticeable on close-up character motion. This risk is unchanged from the source-film path, where it's an accepted tradeoff already.
- If a specific lumpy-blue-men shot's retiming artifact turns out to be visually unacceptable, there's currently no per-shot escape hatch (e.g. "use native length for shot 4") -- the alternative below would need to be revisited for that shot specifically, not the whole pipeline.

# Alternatives Considered

- **Honor the clip's native generated length; let total runtime drift.** Would avoid `setpts`/trim artifacts entirely, generated motion would always look natural. Rejected because it breaks the timing contract the authored manifest is written against -- lumpy-blue-men's beats and the two-cut soundtrack mux depend on shots landing at their planned `start_s`/`end_s`. Drifting runtime would need a second pass to re-derive absolute timing from actual clip lengths before the soundtrack mux could sync to it, adding complexity for a benefit (avoiding retiming artifacts) that's not yet been observed as a real problem in this pipeline.
- **A per-manifest or per-shot flag to opt out of retiming.** Not implemented -- no concrete need has surfaced yet (the probe run showed no retiming artifacts at all for a 3-shot test). Adding the flag speculatively would be exactly the kind of unrequested abstraction to avoid; revisit if a specific lumpy-blue-men shot needs it.

# Links

- [Authored Manifest Verification research](../../research/0022-authored-manifest-verification/research.md)
- [ADR-002: Stateless CLI Pipeline](002-stateless-cli-pipeline.md)
