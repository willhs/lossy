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

**Both fixed after the checkpoint** (2026-08-04):
- `RunPodSession.http` is now one pooled `httpx.Client` (8 connections, 4 keepalive), shared by all five ComfyUI call sites and closed on cleanup. This collapses ~48,000 connection setups to a handful held open per host.
- `CLIP_ATTEMPTS` is 3 with an escalating backoff (20s then 40s), so a transient local failure has room to clear instead of burning four attempts inside 40s.

Still worth doing: a re-sweep pass at the end of a run that retries everything in `failed`.

### Fixed here: the `failed` list was untrustworthy across resumes

The checkpoint showed shots 10-14 sitting in `completed` **and** `failed` simultaneously. `_record_success` never cleared a stale failure from an *earlier* run — only the in-run retry path did. On a resumed 2069-shot run, that makes `failed` useless as a QC signal, which matters exactly when shots are being silently dropped (above).

Fixed in `decode.py`: `_record_success` now clears the index from `failed` on every success path, covering both the video and audio loops (whose duplicated clears are now removed). Three regression tests added; full suite 458 passed.

Note the fix landed *after* this run, so the committed progress file still carries the stale entries `[5..25]` alongside the genuine failure at 56.

### Encode still carries character names — the re-encode was never done

`CLAUDE.md` lists item 5, *"Re-encode the film with the fixed describe pass -- BLOCKS the rehearsal"*, as outstanding. It still is. `shots.json` is dated 29 Jul, from the old naming-inviting prompt.

Measured on the encode being rendered right now: **390 of 2069 shots (18.8%) carry explicit character names in their `subjects` field** — e.g. shot 19, *"A golden humanoid droid (C-3PO) and a blue and white astromech droid (R2-D2)"*. Nine of the 50 checkpoint shots are affected.

This is the exact contamination vector from 0021: naming a character invites the model to reconstruct from world knowledge rather than from the compressed description, which is the property that whole research note argues undermines the reconstruction claim.

**Correction to an earlier claim in this note:** re-encoding does *not* automatically invalidate the clips. `manifest.encode_fingerprint` hashes only shot count, index and start/end times — not descriptions. A stage-2 re-run rebuilds descriptions from the unchanged stage-1 boundaries, so the fingerprint is identical and a resumed decode would **silently skip** the 50 old clips, mixing contaminated renders with clean ones. Restarting cleanly requires explicitly clearing the progress file and moving the old clips aside.

### The re-encode does not fix the naming, and the names are load-bearing

Attempted on 2026-08-04, and **stopped after ~$0.006** when the premise failed.

Ran stage 2 over the first 21 shots with the current `ENCODE_MODEL` (`gemini-2.5-flash-lite`) and the supposedly fixed prompt. Naming rate was **identical to the old encode: 1 of 21 shots, the same shot**:

```
OLD (gemini-3.1-flash-lite-preview):
  "A golden humanoid droid (C-3PO) and a blue and white astromech droid (R2-D2)..."
NEW (gemini-2.5-flash-lite, "fixed" prompt):
  "A golden humanoid droid (C-3PO) and a blue and white astromech droid (R2-D2)
   are in the foreground, walking down a sterile, white corridor..."
```

The reason is in `SYSTEM_PROMPT` itself. The anti-naming clause is scoped entirely to the *continuity context*:

> "It deliberately excludes character identity — do not infer, reuse, or guess who a person is from it. Identify who/what is in THIS shot from these frames alone; **if you don't recognize someone, describe them by visible appearance rather than guessing a name.**"

That forbids *guessing* a name for an unrecognised person. It explicitly permits naming someone the model *does* recognise — and Gemini recognises C-3PO. The fix addressed the Leia-hologram failure (identity carried across a cut) and never addressed naming outright, so `CLAUDE.md`'s "root cause was the prompt inviting naming at all" is only half true.

**And the names cannot simply be banned, because the pipeline consumes them.** `_text_match_cast` in `encode.py` matches TMDB cast names against each shot's `subjects` text; that text match is the primary, cheap path for assigning characters to shots, and the resulting `characters.json` `shots` lists are exactly what `build_character_shot_map` feeds to the identity mixin. Strip the names and the text-match path yields nothing: every shot falls through to the supervised Gemini batch, which can still match on appearance ("a golden humanoid droid") but costs more and is less certain.

So there is a genuine tension: **0021 wants names out of the descriptions; the character pipeline uses those names to know who is in a shot.**

### Resolved: strip at the decode side (2026-08-05)

Will chose to keep names in the encode for stage 3 and strip them when composing the prompt. Implementing it turned up a much larger problem than the descriptions.

**The real exposure was 88.4%, not 18.8%.** `CharacterIdentityMixin._prepend_identity` prepended `"{display_name}: {description}"` for every mapped character, so a name reached the model on **1828 of 2069 shots** regardless of what `subjects` said. The 18.8% figure reported earlier in this note measured the descriptions, not the prompt the model actually receives.

| | shots | % |
|---|---|---|
| Name in `subjects` | 389 | 18.8% |
| Name reaching the model, before | **1828** | **88.4%** |
| Name reaching the model, after | **3** | **0.14%** |

The fix drops the display-name label (the description alone carries the identity) and runs `prompt_format.strip_character_names` over the composed prompt. Stripping keeps the text readable rather than leaving holes: a parenthetical gloss goes whole, an appositive takes both its commas, a governing preposition leaves with its object, and the stranded copula in "C-3PO is a tall golden droid" is repaired to "A tall golden droid". Text with no name in it is returned untouched, so nothing is silently reflowed.

The 3 residual shots name **Greedo** (757) and **Jabba** (791, 793), neither of whom is in `characters.json` — stage 3 keeps only characters appearing in ≥2 shots, and `speakers.json` doesn't list them either. There is no on-disk source for those names, so stripping them would need a hardcoded list for 0.14%. Left as a documented limit.

**This reverses `SPEC-220/REQ-020`**, which required the canonical name in the prompt. The spec is amended in place with the reasoning. `docs/design` is approval-gated, so that edit rides on the naming decision rather than standing on its own.

Note the fix works **on the existing encode** — it needs no re-encode at all. That changes what the re-encode is for: model consistency and shot 451, not naming.

The separate reason to re-encode still stands and is untouched by this: `config.py` records that this encode used `EVAL_MODEL` rather than `ENCODE_MODEL` and "must be re-encoded with ENCODE_MODEL before the real full run so the film is encoded with the same model throughout."

Measured cost of a full re-encode, from the 21-shot sample: **~$0.64**, not the ~$1.30 estimated.

One more thing a re-encode would change: shot **451** is present in `shot_index.json` with all four keyframes on disk, but missing from `shots.json` — it errored during the old describe pass. That single dropped shot is the cause of the known "index diverges from list position from 451 onward" defect.

### Shot 451: one poisoned keyframe, not an undescribable shot

451 failed *again* on the re-encode, reproducibly — so it was never bad luck. It is the Tusken Raider standing over Luke at 29:00.

My first hypothesis was a safety-filter block, and it was **wrong**. The block is `BlockedReason.OTHER`, and setting `BLOCK_NONE` on every harm category does not lift it. Testing that before building on it is what surfaced the real cause: **only the first of its four keyframes trips the filter.** Frames 2, 3 and 4 each describe fine alone, and all four pass under `EVAL_MODEL`.

So a blocked describe call now retries with one frame dropped, largest subsets first. Verified live: 451 recovers for **$0.00016**, described from three frames instead of four — strictly better than losing it and silently renumbering the 1618 shots after it.

### Re-encode result (2026-08-06)

| | |
|---|---|
| Shots described | **2070 — no gaps, the complete film for the first time** |
| Model | `gemini-2.5-flash-lite` (`ENCODE_MODEL`, as `config.py` required) |
| Cost | **$0.3777** (vs ~$0.64 projected, ~$1.30 originally estimated) |
| Wall-clock | ~2.5hrs of describing, plus a 21hr hang (below) |
| Fingerprint | `805f0b0bd0ea` → **`8855b97c449f`** |
| Runtime | 124.7 min, mean shot 3.62s |

Three calls failed across ~2070; two recovered on the existing retry, one was 451.

**The fingerprint change invalidates the 50 checkpoint clips**, as expected under a restart from shot 0. They are archived to `archive_pre_reencode_20260806/` rather than deleted, along with their progress file, so nothing can adopt them off disk.

Naming after the strip, measured on the new encode:

| | shots | % |
|---|---|---|
| Names in `subjects` (kept for stage 3) | 394 | 19.0% |
| Names reaching the video model | **4** | **0.19%** |

The 4 are shots 757, 788, 791, 793 — Greedo and Jabba, still absent from `characters.json`.

### The encode hung for 21 hours

The first re-encode attempt stopped at shot ~400 and sat there: process alive, 0% CPU, 5s of CPU across 23hrs, nothing written, no error. `genai.Client` was built without `http_options`, so `generate_content` had no timeout and blocked forever on a socket the far end had stopped answering — the log shows "Server disconnected without sending a response" 66 shots earlier.

No work was lost (`save_shots` checkpoints every 10 shots) and no money (Gemini bills per call). But it cost a night, and **nothing detected it**: a hang keeps the process alive so liveness checks pass, spends nothing so cost alarms stay quiet, and simply stops changing — which to a progress watcher is indistinguishable from a slow shot. The monitor watching it only reported on count *change*, so it reported nothing at all.

Fixed with a 180s deadline on every Gemini call, and a replacement monitor that alerts explicitly on no progress for 12 minutes.

This is the **third instance of one pattern**: unbounded network calls with no deadline. Shot 56 died to local socket exhaustion, the ComfyUI polling opened ~48,000 connections, and the encode hung on a dead socket. Worth carrying into the 40-hour render, where a silent hang costs far more than a night.

Minor: `encode_costs.json` is rewritten rather than merged on a resumed stage 2, so it now records only the final follow-up call. The real total is in the run log.

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

1. ~~Decide how naming and character-assignment should coexist~~ — **decided and implemented**: names stay in the encode for stage 3, stripped at composition. 88.4% → 0.14%. See above.
2. ~~Pool the HTTP connections and lengthen the retry backoff~~ — **done**, see above. Untested against a live pod; the next run exercises it.
3. **Add an end-of-run retry sweep** over the `failed` list, now that the list is trustworthy. Not done.
4. ~~Decide the letterbox policy~~ — Will chose crop-and-fill; **implemented** in `stitch.py`. `_detect_letterbox` runs ffmpeg `cropdetect` over 60 frames and only treats a 6-40% vertical trim as a matte, so dark shots and pillarboxing are left alone. `_strip_letterbox_group` detects once per shot and applies the same crop to every part — detecting per part would move the jump into the middle of a shot. Verified on the real clips: shot 0058 goes from 23% bars to 0% at unchanged 1280x704 and frame count, and clean shots are correctly untouched. Off via `LOSSY_STRIP_LETTERBOX=0` to compare stitches.
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
