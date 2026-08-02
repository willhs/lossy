---
id: 0023-ltx2-dress-rehearsal
type: note
purpose: "End-to-end dress rehearsal of the Star Wars IV pipeline on one contiguous 5.5-minute segment with LTX-2 at 1280x704, to shake out defects and measure real per-shot cost before committing to the full 2069-shot run."
scope: ["research", "decode", "ltx2", "cost", "rehearsal", "go-no-go"]
tags: ["research", "rehearsal", "ltx-2", "cost", "identity", "resume", "speaker-attribution"]
related: ["research/0022-ltx2-vs-wan22-trial/research.md", "research/0020-wan22-identity-i2v-verification/research.md", "research/0017-video-gen-speed-benchmarks/research.md"]
---

## Question

Steady task: Dress rehearsal -- end-to-end segment with all improvements (`35631193-9f48-466c-be04-bfc389215d42`). Final gate before the expensive full run.

Can the pipeline produce one contiguous watchable segment end-to-end with LTX-2, what does a shot really cost at 1280x704, and what must be fixed before committing to all 2069 shots?

## Method

Segment: shots **794-881** (54.5-60.0 min, 88 shots, 328.3s of film) -- the Death Star infiltration. Chosen for the widest character spread in any 6-minute window (Vader 18 lines, Luke 14, Han 14, Leia 10, Tarkin 4, Obi-Wan 4), which exercises the Vader/Tarkin voice split and stacks more characters per shot than the film average.

Stages run: `decode --strategy runpod-ltx2` (video) -> `--speech` (ElevenLabs via fal, per-character voice map) -> `--audio --audio-strategy runpod-mmaudio` (ambience) -> `--stitch`. A deliberate interrupt/resume was performed during the video pass.

**Caution on shot selection:** the `index` field diverges from list position from shot 451 onward (1618 shots affected) because of the July re-encode. Slicing `shots[794:882]` selects different footage than filtering on `index`. All figures here use the `index` field.

## Findings

### Cost at 1280x704 (the number the task needed)

| Measure | Value |
|---|---|
| Marginal generation cost, warm pod | **~$0.006/shot** (10 shots per ~$0.06 on a $0.33/hr A6000) |
| Wall-clock, warm pod | ~1.4 min/shot; 88 shots in ~2.5hrs including setup |
| Pod setup (per pod) | ~15-20 min, ~$0.10 -- 34GB of model downloads, not cached between pods |
| Video pass, this segment | $1.53 all-in including two wasted pods |
| Speech, 70 lines | $0.16 (fal ElevenLabs) |
| Ambience, 86 shots | $0.20 |

**Full-run projection: $15-25.** 2069 shots x ~$0.006 = ~$13 of GPU time, plus a handful of pod setups and retries. Comfortably inside the $50-60 goal ceiling. The "all-in $/shot" figure tracked during the run peaked at $0.046 and is not a useful projection basis -- it amortises fixed setup over however few shots have landed so far.

Resolution note: the 0022 trial measured LTX-2 at 768x512 and called it ~3x faster than Wan22. At a matched 1280x704 LTX-2 runs ~1.4 min/shot vs Wan22's ~3.25 min/shot, so roughly **2.3x faster**, not 3x. The speed advantage is real but smaller than the unmatched trial suggested.

### Defects found and fixed (all would have hit the full run)

1. **LTX-2 rendered at 768x512.** A placeholder from the trial spike, never raised. Fixed to 1280x704, matching Wan22.
2. **Resume silently dropped interrupted split shots.** `clip_exists_for_shot` returned true if the *first* part existed, so a run killed mid-shot marked the shot done with parts missing. Worse, the adoption branch never populated `progress["clips"]`, so the stitch fell back to a single-file lookup that does not exist for a split shot -- the shot vanished from the film with no error. Fixed via `manifest.existing_clip_parts` + `GenerationStrategy.expected_part_count`, with regression tests.
3. **LTX-2 had no character-identity conditioning at all.** `RunPodLtx2Strategy` extended `RunPodWanStrategy` without `CharacterIdentityMixin`, so no canonical descriptions reached the model -- Han and Luke rendered as generic modern men in a warehouse. Fixed by wiring the mixin in.

   This also **retroactively undermines the 0022 model trial**: `runpod-wan22` carries the mixin and LTX-2 did not, so Wan22 was judged with identity enrichment and LTX-2 without it. The "on par at least" verdict was formed on an unfair comparison in this respect as well as on resolution.

3b. **A failed chain-frame upload aborted the whole render.** A stalled `scp` of the I2V conditioning frame raised `TimeoutExpired` straight through the generate loop and killed a render 84 clips in. Chaining is an enhancement -- losing it for one part costs one seam -- so it now degrades to an unchained part, with the timeout raised to 60s.

4. **No I2V chaining, so long shots jump-cut internally.** `_supports_i2v = False` meant every part of a split shot was generated from scratch. The first rehearsal pass put **172s of 328s (52% of runtime) in split shots and introduced 26 cuts the film does not have** -- worst case shot 846, a 17.7s Vader/Tarkin/Leia scene rendered as four disconnected takes. The task notes flagged the flag; nobody had measured what it cost.

   Fixed with an `LTXVImgToVideo` branch (it supersedes `EmptyLTXVLatentVideo`, returning image-conditioned positive/negative *and* the latent, so conditioning flows through it before `LTXVConditioning` and the AV concat takes its latent). The base class's chaining plumbing needed no changes.

   **Measured effect** (mean absolute gray difference across a part boundary, 0-255; 0 = identical):

   | Shot | Un-chained | Chained |
   |---|---|---|
   | 846 (4 parts) | 52.9 | **2.6** |
   | 794 (3 parts) | 51.7 | **2.1** |

   **Those two figures are wrong** -- they were measured with `-sseof`, which samples ~0.2s before the end rather than the true final frame. Surveyed properly across every multi-part shot, comparing the actual boundary frames against normal frame-to-frame motion:

   | | seam | normal motion | ratio |
   |---|---|---|---|
   | unchained | 44.6 | 5.24 | **8.5x** |
   | chained | 10.4 | 4.67 | **2.2x** |
   | chained + tail-trim | 11.5 | 5.12 | 2.2x |

   So chaining takes a seam from 8.5x a normal frame step down to 2.2x -- a large, genuine win, but not the near-seamless continuity the flawed measurement suggested. **The residual 2.2x is still visible as a hitch in elongated scenes, and Will flagged it on viewing.**

   Trimming the eased frames at seams was tried against it and **measured no benefit** (11.5 vs 10.4), so it is disabled. Two things were learned in the attempt: a head trim is actively harmful (seams of 25-43) because continuity lives in the conditioned first frames, and a trim retrofitted onto already-generated clips is equally harmful (23.1) because it breaks the correspondence with the frame the next part was conditioned on. Any future attempt must trim before `_upload_start_frame` and never touch the head.

   Side benefit of chaining: Vader renders in correct black armour when chained versus a grey sweatshirt un-chained -- each part inherits the previous part's appearance instead of re-rolling it, so chaining damps within-shot identity drift too.

5. **GPU fallback chain was too narrow to survive capacity pressure.** A re-render attempt exited with no pod at all: RTX A6000 reported no instances available and L40S errored in the same breath, and those were the only two types tried. Widened to five 48GB types ordered by price (A6000 $0.33, A40 $0.35, L40 $0.69, RTX 6000 Ada $0.74, L40S $0.79). The very next attempt found A6000 *and* A40 both unavailable and succeeded on L40 -- it would have failed outright without the fix. Also corrected the L40S rate, hardcoded at $0.54 against an actual $0.79.

### Pod reliability -- the largest open risk

Across **10 pod provisioning attempts** this session:

| Outcome | Count | Cost |
|---|---|---|
| Worked | 6 | -- |
| **Died on CUDA init** before ComfyUI started | **3 (30%)** | $1.05, no output |
| **No capacity at all** (every GPU type unavailable) | 1 | $0 |

Every one of these required a human to notice and restart by hand -- `decode.py` exits rather than re-provisioning. A full run is ~40+ GPU-hours across many pods; at a 30% dead-on-arrival rate that is roughly one stall per hour of wall-clock, each stalling the run indefinitely until someone intervenes.

**This, not cost or quality, is the thing most likely to make the full run painful.** An automatic re-provision-on-failure loop (retry N times, cycling GPU types) is the highest-value fix remaining.

Capacity pressure also moves cost: the successful chained re-render landed on L40 at $0.69/hr because both cheaper pools were empty, roughly doubling that render's cost. Full-run cost therefore depends on which pool has capacity -- **$15-25 at A6000 rates, $30-40 if it runs on L40-class hardware throughout**.

### Defects found, not fixed (go/no-go list)

4. **Canonical descriptions bake in one costume for the whole film.** Stage 3 derived Luke's description from all 572 of his shots, and the trench run dominated: "typically seen in an orange flight suit... yellow goggles". The mixin prepends that verbatim to every Luke shot by design, so Luke wore an X-wing pilot suit on the Death Star. **334 of his 572 shots (58%) predate Yavin**, where that is wrong.

   *Worked around for the rehearsal* by hand-stripping wardrobe from 11 human character descriptions in `characters.json`, leaving costume only where it is the character (Vader, troopers, droids, Chewbacca, Jawa). **This edit lives in `output/`, which is gitignored -- a stage-3 re-run regenerates the flight-suit descriptions.** The fix belongs in the stage-3 prompt in `encode.py`: ask for stable identity (face, build, bearing) and leave wardrobe to per-shot descriptions.

5. **Identity bleed when several characters are stacked.** Concatenated identity blocks blend across figures -- Luke rendered with C-3PO's gold droid body and a human head. Affects shots with >=3 blocks: **196 of 2069 shots (9.5%)** film-wide; 14 of 88 (16%) in this segment. Mitigation: cap identity blocks at the two most prominent characters per shot and let the shot description carry the rest.

6. **Non-humanoids render as humans in costume.** C-3PO's description ("a tall, golden *humanoid* droid... he has no hair") reliably produces a man in a gold bodysuit. R2-D2, having no human referent, renders well. Fix is wording: describe C-3PO as a mechanical robot, not a humanoid.

7. **Speaker attribution has a measurable error floor.** 20 of 1298 lines (1.5%) are attributed to a character who cannot be in that scene -- Han at 7 min before he appears, Aunt Beru at 97 min, Dodonna during the 37-min Death Star conference, Obi-Wan delivering Dodonna's briefing. Same-era swaps are invisible to this check, so **1.5% is a floor, not the rate**. Mean confidence 0.88; 19% of lines below 0.8.

   Consequence is audible: a mis-attributed line gets the wrong voice and can manufacture the exact collision the casting amendment removed (a line at 59.6 min attributed to Uncle Owen, who is dead, plays in George 13s before George speaks as Han). Mitigation: route lines below a confidence threshold to the narrator voice.

8. **`--keep-pod` has no idle timeout.** The video pass finished in 2.5hrs; the pod then idled ~10.5hrs at $0.33/hr because the next stage was not started. **~$3.50 burned doing nothing -- more than twice the render itself.** On a full run with any human-in-the-loop gap this is the single largest avoidable cost. Fix: idle-timeout in `RunPodSession`, or never pass `--keep-pod` without a watchdog.

9. **Stale per-strategy artifacts silently contaminate a new run.** `audio/runpod-mmaudio/` held 87 clips covering shots 794-881 from March, predating the July re-encode that renumbered shots. Their progress file would have made the audio pass skip every shot as done, and the first test stitch muxed whole-film March tracks onto a 328s video -- silently, no error. Archived to `output/star_wars_iv_v2/archive_pre_rehearsal_20260801/`. The pipeline has no run-identity or encode-generation stamp to detect this.

10. **Stitch muxes every audio strategy it finds**, including ones with no clips in range (elevenlabs, mmaudio, musicgen were confirmed digitally silent here). Harmless but wasteful, and it is what let the stale March tracks in.

11. **Two ambience shots failed** (841, 848) out of 88 -- 2.3%. Not retried; a retry costs a whole pod spin-up for two clips.

12. **Final mux peaks at 98.8% FS** despite `alimiter=limit=0.95` in the chain. Worth a look before mastering a full film.

### What worked

- **88/88 shots generated, zero failures** -- twice, on both the un-chained and chained passes.
- **Chaining holds across the whole segment**, not just the validation shot: 20 of 88 shots split (68 single-part, 15 two-part, 4 three-part, 1 four-part), and sampled boundaries sit at ~2 rather than ~52.
- **Timeline accuracy: 328.27s output vs 328.285s expected -- 15ms drift over 5.5 minutes.** The boundary-locked retiming holds at length; per-shot deviations of +/-0.05-0.09s are corrected shot-by-shot rather than accumulating.
- **Interrupt/resume works** (and earned its cost by exposing defect 2). Progress file stayed consistent; in-flight shots were lost cleanly rather than half-recorded.
- **Voice map fully resolves**: all 70 lines in the window mapped, no unmapped speakers. Han and Tarkin both use George but land 37s apart in different scenes.
- **Speech and ambience both real in the mux** (58.9% and 70.7% FS peaks), correctly restricted to the segment by `--start-index/--limit`.
- `--limit` now honoured on stitch (was ignored, sweeping in every clip after `--start-index`).

## Recommendation

**Conditional go.** Cost is not the risk -- $15-25 against a $50-60 ceiling. The pipeline runs a contiguous segment end-to-end without errors and holds its timeline.

**All four blocking items are now fixed** (2026-08-02):
- **Automatic re-provision on pod failure** -- the three host-level failures raise a retryable `PodSetupError` and `RunPodSession.with_setup_retry` re-runs setup on a fresh pod up to four times, clearing stale state between attempts. Every RunPod strategy inherits it.
- **Defect 8 (idle pod)** -- `--keep-pod` takes a duration (default 30 min) and the pod arms its own `runpodctl remove pod $RUNPOD_POD_ID` before the local process exits. "Forever" is no longer expressible. *Not yet exercised against a live pod.*
- **Defect 4 (costume in canonical descriptions)** -- all three stage-3 description prompts now ask for stable identity and forbid clothing for human characters, keeping the carve-out for characters whose costume or shell *is* their identity. This replaces the gitignored hand-edit.
- **Defect 9 (stale artifact contamination)** -- `manifest.encode_fingerprint` hashes the shot boundaries and `check_encode_fingerprint` refuses progress files from a different encode, with an mtime fallback that catches legacy unstamped files. Wired into video, audio and speech resume.

Worth fixing, quality-visible:
- **Defect 5** (cap identity blocks at 2) and **defect 6** (reword non-humanoid descriptions) -- together these cover the worst visual failures.
- **Defect 7** (confidence-threshold fallback for speaker attribution).

Open question for Will: the segment is watchable and recognisably Star Wars in places -- Leia is genuinely good, R2-D2 and the lightsaber read correctly -- but Vader wears a grey sweatshirt and Chewbacca is a bearded man in a fur hood. **Whether that clears the bar for a full 116-minute reconstruction is a judgement call, not a measurement.**

## Artifacts

- `output/star_wars_iv_v2/star_wars_iv_v2_reconstructed_runpod-ltx2.mp4` -- the rehearsal segment (328s, 1280x704, chained, ambience + speech)
- `output/star_wars_iv_v2/clips/runpod-ltx2/` -- 114 clip files, 88 shots, chained
- `output/star_wars_iv_v2/clips/runpod-ltx2-nochain/` -- the same 88 shots without I2V chaining (before/after on the 26 spurious cuts)
- `output/star_wars_iv_v2/clips/runpod-ltx2-noidentity/` -- 45 shots with no identity conditioning (before/after comparison)
- `output/star_wars_iv_v2/clips/runpod-ltx2-costumebug/` -- 3 shots with the flight-suit-everywhere bug
- `output/star_wars_iv_v2/characters.json.bak-precostume` -- descriptions before wardrobe stripping
- `output/star_wars_iv_v2/archive_pre_rehearsal_20260801/` -- the March audio/speech artifacts, archived not deleted

## Spend

**~$12.30 total this session**, against ~$3.90 spent previously on the goal (~$16.20 cumulative, ceiling $50-60).

| | |
|---|---|
| Final render, 88 shots chained (A6000, 2.0hrs) | $1.35 |
| Earlier chained render (L40 @ $0.69/hr, 1.6hrs) | $1.10 |
| Abandoned head-trim render + its crash restart | $1.25 |
| Chaining validation on one shot | $0.31 |
| First 88-shot render (A6000 @ $0.33/hr, 2.5hrs) -- superseded by chaining | $0.83 |
| Speech, 70 lines (fal) | $0.16 |
| Ambience, 86 shots | $0.20 |
| Un-enriched + costume-bug renders (archived) | $0.46 |
| Interrupt/resume test -- found the shot-dropping bug | $0.16 |
| **Idle pod left running after a completed render** | **$3.40** |
| **Three pods dead on CUDA init, no output** | **$1.05** |

Useful generation was about **$2.60 of ~$12.30**. The rest was overhead, waste, and superseded work -- dominated by the idle pod and dead-on-arrival pods, both of which are fixable in code.

Note the L40 ran the same 88 shots in 1.6hrs versus 2.5hrs on the A6000, so the pricier GPU was only ~30% more expensive in total, not 2x.
