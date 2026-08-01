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

- **88/88 shots generated, zero failures.**
- **Timeline accuracy: 328.27s output vs 328.285s expected -- 15ms drift over 5.5 minutes.** The boundary-locked retiming holds at length; per-shot deviations of +/-0.05-0.09s are corrected shot-by-shot rather than accumulating.
- **Interrupt/resume works** (and earned its cost by exposing defect 2). Progress file stayed consistent; in-flight shots were lost cleanly rather than half-recorded.
- **Voice map fully resolves**: all 70 lines in the window mapped, no unmapped speakers. Han and Tarkin both use George but land 37s apart in different scenes.
- **Speech and ambience both real in the mux** (58.9% and 70.7% FS peaks), correctly restricted to the segment by `--start-index/--limit`.
- `--limit` now honoured on stitch (was ignored, sweeping in every clip after `--start-index`).

## Recommendation

**Conditional go.** Cost is not the risk -- $15-25 against a $50-60 ceiling. The pipeline runs a contiguous segment end-to-end without errors and holds its timeline.

Fix before the full run (cheap, all code-side):
- **Defect 8 (idle pod)** -- the only one that costs real money, and it cost more than the rehearsal render.
- **Defect 4 (costume in canonical descriptions)** -- currently a hand-edit in gitignored output that a re-encode silently reverts. Move it into the stage-3 prompt.
- **Defect 9 (stale artifact contamination)** -- an encode-generation stamp, or the full run starts from a clean output dir.

Worth fixing, quality-visible:
- **Defect 5** (cap identity blocks at 2) and **defect 6** (reword non-humanoid descriptions) -- together these cover the worst visual failures.
- **Defect 7** (confidence-threshold fallback for speaker attribution).

Open question for Will: the segment is watchable and recognisably Star Wars in places -- Leia is genuinely good, R2-D2 and the lightsaber read correctly -- but Vader wears a grey sweatshirt and Chewbacca is a bearded man in a fur hood. **Whether that clears the bar for a full 116-minute reconstruction is a judgement call, not a measurement.**

## Artifacts

- `output/star_wars_iv_v2/star_wars_iv_v2_reconstructed_runpod-ltx2.mp4` -- the rehearsal segment (328s, 1280x704, ambience + speech)
- `output/star_wars_iv_v2/clips/runpod-ltx2/` -- 114 clip files, 88 shots
- `output/star_wars_iv_v2/clips/runpod-ltx2-noidentity/` -- 45 shots with no identity conditioning (before/after comparison)
- `output/star_wars_iv_v2/clips/runpod-ltx2-costumebug/` -- 3 shots with the flight-suit-everywhere bug
- `output/star_wars_iv_v2/characters.json.bak-precostume` -- descriptions before wardrobe stripping
- `output/star_wars_iv_v2/archive_pre_rehearsal_20260801/` -- the March audio/speech artifacts, archived not deleted

## Spend

**$5.21 total this session**, against ~$3.90 spent previously on the goal (~$9.11 cumulative, ceiling $50-60).

Of that $5.21, only ~$1.70 was useful generation: $1.02 video + $0.16 speech + $0.20 ambience, plus ~$0.30 of productive pod setup. The remainder was **$3.50 idle pod** (defect 8), **$0.25** a pod that died on CUDA init before doing any work, and **$0.16** the interrupt/resume test (which paid for itself by finding defect 2).
