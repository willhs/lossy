---
id: 0024-full-run-checkpoint
type: note
purpose: "First checkpoint of the full Star Wars IV run: 50 shots rendered end-to-end on live hardware to measure real cost/rate at scale, validate pod self-destruct and the stray sweep against a live pod, and surface anything that changes the go/no-go on the remaining ~2000 shots."
scope: ["research", "decode", "ltx2", "cost", "full-run", "pod-hygiene", "go-no-go"]
tags: ["research", "full-run", "checkpoint", "ltx-2", "cost", "runpod", "letterbox", "go-no-go"]
related:
  - "research/0023-ltx2-dress-rehearsal/research.md"
  - "research/0021-training-data-contamination/research.md"
  - "research/0020-wan22-identity-i2v-verification/research.md"
---

# Full-run checkpoint: first 50 shots

## Question

Steady task: *Full run: 2069 shots, Star Wars IV end-to-end* (`0725e309-8fd7-4a92-bfe6-7aac4822449d`).

Will green-lit **starting** the run, not completing it. Render roughly the first 50 shots, stop, terminate the pod, and report. Does the pipeline hold at scale, what does a shot really cost, does pod self-destruct actually work on live hardware, and does anything here change the go/no-go on the remaining ~2000 shots?

## Method

Config taken from the dress rehearsal (0023), unchanged:

```
python decode.py output/star_wars_iv_v2 --strategy runpod-ltx2 --start-index 10
```

`--start-index 10` skips the logo/crawl shots (0-9), which were already rendered. No `--keep-pod`: this checkpoint has no following audio stage, so the pod should die with the process.

Ran 19:02-20:03 NZST on one RTX A6000 @ $0.33/hr. Stopped with SIGINT at 50 new shots. Shots 10-60 inclusive, minus shot 56 (see below).

**Note on where this ran:** the render was executed in the main checkout (`/Users/will/projects/lossy`), not the job worktree, because the 2069-shot encode lives in the gitignored `output/star_wars_iv_v2/` there and is not reproducible in a fresh worktree. Only code changes were made in the worktree, on the job branch.

## Findings

### Cost and rate: faster than the rehearsal, but do not extrapolate per shot

| Measure | Checkpoint | Rehearsal (0023) |
|---|---|---|
| Wall-clock, warm pod | **0.84 min/shot** | 1.4 min/shot |
| Marginal cost, warm pod | **$0.0046/shot** | $0.006/shot |
| All-in incl. this pod's setup | $0.0066/shot | -- |
| Pod setup | 18.2 min | 15-20 min |
| Total spend, this checkpoint | **$0.33** (1 pod, 1.0 hr) | -- |

Setup was 18.2 min of the 60.3-min pod lifetime; generation was 42.1 min for 50 shots.

**The per-shot figures flatter the full run and must not be extrapolated directly.** Shots 10-59 average **2.49s** against a film mean of **3.62s** — the opening reel is short-shot-heavy, so this slice is **1.45x shorter per shot** than the film. Cost tracks *duration* (parts generated), not shot count.

Extrapolating on film-seconds instead (2.95 s of film per GPU-minute):

| | |
|---|---|
| Film runtime | 124.7 min across 2069 shots |
| **GPU hours needed** | **~42 hr** |
| At A6000/A40 ($0.33-0.35/hr) | **~$14** |
| At L40-class ($0.69-0.79/hr) | **~$31** |

A naive per-shot extrapolation would have said $9.59 — a ~35% underestimate. The duration-weighted figure lands squarely on the rehearsal's $15-25 / $30-40 range, so **the rehearsal's projection holds and the $50-60 ceiling is not at risk**.

### Pod hygiene: all three mechanisms validated on live hardware

This was the headline unknown — self-destruct and the stray sweep were unit-tested only. All now exercised against real pods.

1. **Self-destruct fires. Confirmed end-to-end.** On a throwaway pod, armed a 2-minute TTL through the real `arm_self_destruct` code path, then *dropped the local reference entirely* so nothing local could intervene. The pod removed itself after **127s against a 120s TTL**, verified gone via the RunPod API. Cost of the test: ~$0.03.

   The two preconditions that could have silently broken it both hold: `RUNPOD_POD_ID` **is** exported to non-interactive SSH sessions (the most likely silent failure), and `runpodctl` is present at `/usr/bin/runpodctl` on the stock image.

2. **Stray sweep ran clean.** SIGINT drove the normal cleanup path: pod terminated, sweep ran, nothing reported. **RunPod API confirms 0 pods remaining** — no pod from this job survives.

3. **GPU fallback chain earned its keep, unprompted.** The self-destruct test hit real capacity pressure: RTX 4090 "does not have the resources", RTX A4000 "no longer any instances available", succeeded on A6000. Without the widened chain from 0023 that attempt would simply have failed.

Minor, cosmetic: `disarm_self_destruct` kills the `sh -c` wrapper and removes the PID file, but orphans the child `sleep`. Disarm is still *effective* — the dead parent can never run the removal — it just leaves a harmless sleeping process.

### NEW DEFECT: 26% of shots render with baked-in letterbox bars

**13 of 50 shots (26%)** come back with ~23% of frame height as black bars top and bottom, baked into the pixels. The rest are full-frame. They are scattered, not clustered:

```
shots 10-60:  ...L.LL............LL.....L.....L.L.....L..L..x.LLL
              (L = letterboxed, . = full frame, x = shot 56, lost)
```

Consecutive shots therefore flip between ~2.35:1 and the native 1.82:1, which will read as **aspect-ratio flicker throughout the film**, plus a ~23% vertical resolution loss on the affected quarter.

The rehearsal could not have caught this: its segment (794-881) was Death Star interiors, and this is triggered by wide space/vista and corridor content.

I looked for a prompt-level cause and **did not find one**. Letterboxed and clean shots have indistinguishable prompt structure — both open "Cinematic \<shot type\>, \<camera move\>". Shots 13/15/16 letterbox while 14/17/18 do not, with near-identical phrasing. It reads as a stochastic model association with "cinematic wide space shot", not a fixable token.

Mitigations, cheapest first — none requires re-rendering what exists:
- **Detect and crop in the stitch.** Bars are trivially detectable (row luminance < 8). Crop and rescale affected clips to fill. Costs some framing, removes the flicker entirely.
- Or letterbox *everything* consistently for a uniform 2.35:1 film.
- Or accept it.

### NEW DEFECT: a transient local socket error permanently drops a shot

Shot 56 is missing from the film. Cause:

```
Error generating 0056.mp4: [Errno 49] Can't assign requested address
0056.mp4: attempt 1 failed, retrying in 20s [attempt 2/2]...
No output file found for 0056.mp4
Retrying shot 56 in 10s...          <- outer retry, also 2 attempts
No output file found for 0056.mp4
```

`Errno 49 (EADDRNOTAVAIL)` is a **local** failure — the client could not get a socket — not a pod fault. All four attempts fell inside ~40s, too tight for socket pressure to clear, so the shot was abandoned and the run moved on.

The likely mechanism is connection churn: **every** ComfyUI call opens a fresh TCP+TLS connection. `runpod_pod.py` uses bare `httpx.get`/`httpx.post` (lines 555, 639, 653, 674, 694) with no shared `httpx.Client`, so nothing is pooled. This run made **1,249 connections for 50 shots (~25/shot)**, 1,071 of them 2-second history polls. Extrapolated: **~48,000 connections across the full run.**

**Observed loss: 1 shot in 50 (2%).** If that rate holds, roughly **40 shots go missing across the full run**, each a silent gap, recorded only in the `failed` list.

Recommended before the full run (all cheap):
- Reuse a single `httpx.Client` in `RunPodSession` — removes ~48,000 connection setups and most of the exposure.
- Lengthen the retry backoff; 20s/10s is too short for socket-level pressure.
- Add a re-sweep pass at the end of a run that retries everything in `failed`.

### Fixed here: the `failed` list was untrustworthy across resumes

The checkpoint showed shots 10-14 sitting in `completed` **and** `failed` simultaneously. `_record_success` never cleared a stale failure from an *earlier* run — only the in-run retry path did. On a resumed 2069-shot run, that makes `failed` useless as a QC signal, which matters exactly when shots are being silently dropped (above).

Fixed in `decode.py`: `_record_success` now clears the index from `failed` on every success path, covering both the video and audio loops (whose duplicated clears are now removed). Three regression tests added; full suite 458 passed.

Note the fix landed *after* this run, so the committed progress file still carries the stale entries `[5..25]` alongside the genuine failure at 56.

### Encode still carries character names — the re-encode was never done

`CLAUDE.md` lists item 5, *"Re-encode the film with the fixed describe pass -- BLOCKS the rehearsal"*, as outstanding. It still is. `shots.json` is dated 29 Jul, from the old naming-inviting prompt.

Measured on the encode being rendered right now: **390 of 2069 shots (18.8%) carry explicit character names in their `subjects` field** — e.g. shot 19, *"A golden humanoid droid (C-3PO) and a blue and white astromech droid (R2-D2)"*. Nine of the 50 checkpoint shots are affected.

This is the exact contamination vector from 0021: naming a character invites the model to reconstruct from world knowledge rather than from the compressed description, which is the property that whole research note argues undermines the reconstruction claim.

**This is the single most consequential decision to make now, because it is destructive later.** A re-encode changes `manifest.encode_fingerprint`, which by design invalidates every existing progress file and clip. Doing it after 2000 shots throws away ~40 GPU-hours; doing it now throws away 50 shots and ~$0.33. Cost of the re-encode itself: ~$1.30.

### What held up well

- **50/50 attempted shots generated** aside from the socket loss; **zero pod failures**, against the rehearsal's 30% dead-on-arrival rate. One pod, first GPU type, no re-provisioning needed.
- **Chaining is better than the rehearsal.** Seam ratio **1.2x** normal frame motion (mean seam 2.7 against motion 2.28), versus 2.2x chained in 0023 and 8.5x unchained. 5 of 50 shots split (4 two-part, 1 four-part).
- **Resolution correct**: 1280x704 @ 25fps confirmed on disk, parts capped at 121 frames.
- **Stormtroopers render genuinely well** — recognisable armour, correct white corridor, red blaster fire.
- **Rehearsal defect 6 did NOT reproduce.** C-3PO rendered as an actual gold *metallic droid*, not a man in a gold bodysuit, and R2-D2 read correctly. The stage-3 description rewording appears to have worked.

### Not exercised at this checkpoint

- **Identity bleed (3+ characters)** — the slice contains **zero** such shots, so the 9.5%-of-film defect is untested here. It remains an accepted limit going in.
- **Speech, ambience, stitch** — video pass only.
- **`--keep-pod` in anger** — the TTL was validated on a throwaway pod, not across a real video-to-audio handoff.
- **Character-shot-map misattribution** (noticed, low impact): shots 40 and 48 on the Tantive IV are attributed to Uncle Owen and Luke Skywalker, neither of whom can be there — the same class of error as the 1.5% speaker-attribution floor. Both rendered as plausible rebel troopers, so no visible harm this time, but the mixin will happily prepend the wrong canonical face.

## Conclusions

The pipeline holds at scale. Cost is not the risk: ~42 GPU-hours and **~$14-31**, comfortably inside the $50-60 ceiling. Pod hygiene — the thing the task flagged as the money risk — is now **validated on live hardware for the first time**, all three mechanisms, and nothing survives this job.

Two new defects surfaced that only scale could have shown, both because the rehearsal segment was interiors and this one is not: **letterbox flicker on 26% of shots**, and **silent shot loss from local socket exhaustion at ~2%**.

The genuinely awkward finding is not a defect at all: the film is being rendered from an encode that the project's own notes say should have been replaced first, and 18.8% of its shots name characters outright.

## Recommendations

Ordered by what has to be decided before more GPU time is spent.

1. **[HUMAN] Decide the re-encode before resuming.** Re-encoding invalidates all clips by fingerprint. Doing it now costs 50 shots and $0.33; doing it after the full run costs ~40 GPU-hours. Given 0021's argument, my recommendation is to **re-encode now** (~$1.30) and restart the run from shot 0.
2. **Pool the HTTP connections** (single `httpx.Client` in `RunPodSession`) and lengthen the retry backoff, before committing 40 GPU-hours to a path that loses ~2% of shots. Cheap, and the loss is otherwise silent.
3. **Add an end-of-run retry sweep** over the `failed` list, now that the list is trustworthy.
4. **Decide the letterbox policy** — crop-and-fill in the stitch is the cheapest fix and needs no re-render. This is a judgement call about how the film should look, so it is Will's, not mine.
5. Optional: reap the orphaned `sleep` in `disarm_self_destruct`. Cosmetic.

**Go/no-go on the remaining ~2000 shots: technically GO — cost, reliability and pod hygiene all check out.** The blocker is editorial, not technical: settle the re-encode question first, because that decision is cheap today and expensive after 40 GPU-hours.

## Artifacts

- `output/star_wars_iv_v2/clips/runpod-ltx2/` — 57 clip parts, shots 10-60 (56 missing)
- `output/star_wars_iv_v2/decode_progress_runpod-ltx2.json` — 60 completed, `failed` carries stale `[5..25]` plus the genuine 56
- `/tmp/lossy_fullrun_logs/decode_full.log` — full run log including the shot 56 failure

## Spend

| | |
|---|---|
| Checkpoint render, 50 shots (A6000, 1.0 hr) | $0.33 |
| Live self-destruct validation (throwaway pod) | ~$0.03 |
| **This job total** | **~$0.36** |

Against the task's "well under $1" allowance. Cumulative on the goal: ~$16.60 of the $50-60 ceiling.
