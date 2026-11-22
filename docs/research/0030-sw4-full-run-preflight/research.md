---
id: 0030-sw4-full-run-preflight
type: note
purpose: "No-spend preflight rehearsal of the 30 Nov Star Wars IV full run (job 08db840b): every item passed or failed with evidence, the exact command, and what must change before the run."
scope: ["research", "decode", "stitch", "full-run", "runpod", "capped-run", "preflight"]
tags: ["research", "preflight", "star-wars", "ltx-2.5", "keyframe", "resume", "spend-cap", "go-no-go"]
related:
  - "research/0026-seinfeld-s01e03-first-end-to-end-run/research.md"
  - "research/0024-full-run-checkpoint/research.md"
  - "research/0029-character-reference-continuity/research.md"
  - "design/adr/0010-pod-ownership-journal.md"
---

# Star Wars IV full run: no-spend preflight (21 Nov 2026)

Vision alignment: Have an online presence (goal: complete a movie or TV show reconstruction, Star Wars IV)

Steady task `e42fb813`. The run is job `08db840b` (Steady task `0725e309`), scheduled for
**30 Nov 13:00 NZDT on `budgie`**. This rehearsal used mocks, dry runs and local ffmpeg only:
no paid call, no pod, $0. The RunPod and fal edges were faked by
[`rehearse.py`](rehearse.py) (a fake pod session and a fake account; the real `decode.main`,
`run_decode`, `RunPodLtx25Strategy.generate`, `stitch_clips` and `tools/capped_run.main` ran
unchanged) and [`cap_scenario.sh`](cap_scenario.sh).

Confidence: **observed** = run here against the real SW4 encode (copied read-only into a scratch
dir); **observed/mock** = run here with a faked paid edge; **source** = read in code;
**inferred** = arithmetic from measured figures; **unproved/live** = needs a key or a pod.
Evidence files are in the job workspace `evidence/` directory (not committed).

## Verdict

**NO-GO for 30 Nov as the job stands. The code is close; the job is not.** Four things must change
first, and none of them needs code: where the run happens (B1), what the cap buys (B2), the
job's own instructions (B3), and two decisions that are Will's (B4). The code defects this
rehearsal found (shots out of film order in the stitch, a truncated progress file blocking every
resume) are fixed here with tests.

| # | Blocker | Why | Fix |
|---|---|---|---|
| B1 | **`budgie` has no encode.** Job 08db840b is targeted at `budgie`, and `~/projects/lossy/output/` does not exist there (no `shots.json`, no `characters.json`, no speech). `output/` is gitignored, so a fresh job worktree has none either. | observed (`ssh budgie ls ~/projects/lossy/output`: no such directory) | Copy the minimal input set (below) to a persistent dir on budgie with `rsync -a` (keeps mtimes; `cp -R` broke resume in 0026) and point the command at that absolute path. |
| B2 | **$27 does not finish the film on LTX-2.5.** The plan is 2,070 shots / 2,687 clips. At the Seinfeld pace (about 46 s + 16 s per video-second per shot) that is **~60 GPU-hours: $35 at the A6000 rate 0026 paid ($0.59/h), $41 on an L40 ($0.69/h)**, plus ambience (~$8 at Seinfeld's $0.0044/clip). The older $15-25 projection was LTX-2 (0.84 min/shot); LTX-2.5 runs ~2.3 min/shot. With $27 the video pass stops around 65-75% of the film at a resumable point. | inferred from observed plan + 0026 figures | Will decides: raise the cap (and the account balance; it was ~$5.45 on 8 Nov before ~$3 of research 0029 spend, unproved/live), or accept a planned partial run that resumes later. |
| B3 | **The job's notes are stale and contradict each other.** They still scope "stop at ~50 shots", list `runpod-wan + runpod-mmaudio --concurrent-audio` as defaults (`--concurrent-audio` is silently ignored by every LTX strategy: `RunPodLtx2Strategy.__init__` forces it off), cite `bbe9506` and "self-destruct never validated" (0024 validated it; ADR 0010's journal replaced the sweep), say 2,069 shots (the encode has 2,070, index 0-2069), and never give the exact command. A headless agent reading them may stop at 50 shots or pick the wrong strategy. | observed (notes vs code) | Done in this job: a "PREFLIGHT 21 Nov" section appended to Steady task `0725e309`'s notes supersedes the stale parts and gives the command below. |
| B4 | **Two product calls are open.** (a) Keyframe-first or text: the verdict comes from task `236284e2` (job `2ed396f1`, running now). (b) Style prefix: SW4 `shots.json` has none, so no prompt says "live-action"; 0026/0028 found cartoon drift without one, and the director note says to stop if the Seinfeld report says "not ready". | observed | Will picks both before the run. Recommended defaults below. |

Filed under goal `b33ea4e0`: `f91deccb` (B1, copy the encode to budgie, headless, $0) and
`bdebe712` (B2 + B4, Will's calls).

### SW4 cost with keyframes (updated 22 Nov 2026, task 0017)

Keyframe-first passed its paid check (research 0029, "SW4 paid check, finish"). On fal it added
$15-19 and took the run over the ceiling. Task 0017 moved klein onto the LTX pod
(`--keyframe-generator pod`, now the default) and measured it on SW4 684-710: about 10 s of A40 time per
keyframe, the same face cosine as fal or better (research 0029, "Self-hosted klein on the pod").

| Item | Text only | Keyframe on fal | Keyframe on the pod |
|---|---|---|---|
| Video, LTX-2.5 (~60 GPU-h, B2) | $35-41 | $35-41 | $35-41 |
| Ambience (MMAudio) | ~$8 | ~$8 | ~$8 |
| Sheets (12 x up to 3 draws) | - | ~$0.19 | ~6 min of pod time, ~$0.06 |
| Keyframes (795 shots) | - | $15-19 (upper bound $18.55) | ~795 x 10 s = 2.2 GPU-h, **$1.30-1.50** at $0.59-0.69/h |
| klein download at boot (12.5 GB) | - | - | ~2 min per pod, under $0.03 each |
| **SW4 total** | **$43-49** | **$58-68** | **$44.5-50.5** |

So keyframe-first on the pod fits the $50-60 ceiling, where fal did not. The fal cap
(`--keyframe-fal-cap`) and `FAL_KEY` are no longer needed; the pod time is RunPod spend, so
`capped_run` now sees it. One more pod-time item: a replacement pod re-downloads klein (~2 min)
even though the keyframes are already cached locally.

Two "MUST LAND BEFORE 30 NOV" tasks from the 0026 review are still in the backlog and not
dispatched: `fa27c947` (ssh boot-race fatality, F6) and `4189b5dc` (refuse-to-rent escape
hatch, F5). The run is not safe unattended without F6 (one unlucky replacement pod ends the
decode process; capped_run restarts it up to 6 times). Filed as a blocker reminder, not redone here
(they edit `runpod_pod.py`, which this job avoided).

## Preflight checklist

| Item | Result | Evidence |
|---|---|---|
| **Job description matches today's code** | **FAIL** (B3) | Notes vs `decode.py` flags, `strategies_video.py:1382`, `docs/research/0024`, ADR 0010 |
| **Encode on the run machine** | **FAIL** on budgie (B1); PASS on the Mac (`/Users/will/projects/lossy/output/star_wars_iv_v2`, fingerprint `8855b97c449f`, 2,070 shots, 124.7 min) | observed |
| **Plan generation, keyframe OFF** (`runpod-ltx25`) | **PASS**: 2,070 shots, 2,687 clips (360 split shots), every ComfyUI graph builds (frames % 8 == 1), 126.3 min generated, 0 personal names in any prompt (designs such as Vader and C-3PO keep theirs on purpose via `keep_name`), longest prompt 1,915 chars | observed: `rehearse.py plan`, `decode.py --strategy runpod-ltx25 --dry-run` (new, below); `evidence/plan_ltx25_full.json` |
| **Plan generation, keyframe ON** (`runpod-ltx25-keyframe`) | **PASS with two conditions**: 795 keyframe shots, 1,275 text (1,047 design characters, 228 no characters), 12 sheets. (1) The default `--keyframe-fal-cap` is **$1.00** against an upper bound of **$18.55** (keyframes) + $0.19 (sheets): with the default only ~40 shots get a keyframe and the rest silently fall back to text, so the flag must be set. (2) The face models (`models/face`: `yunet.onnx`, `sface.onnx`) exist on neither the Mac primary checkout nor budgie; the run stops at `prepare()` before any pod (fail closed, observed) until `LOSSY_FACE_MODELS` points at them. | observed: `evidence/keyframe_dryrun_full.txt`, `evidence/keyframe_on_preflight.log` |
| Keyframe ON first prompt sanity | **Note**: the first keyframe shot is 48 (Tantive IV), whose shot map says Luke Skywalker; 0024 already flagged 40 and 48 as misattributed. A keyframe there draws Luke's face onto a rebel trooper. | observed |
| **Crash mid-run, resume** (SIGKILL inside a split shot, under the real `capped_run`) | **PASS**: capped_run saw exit 137, swept the orphaned pod, restarted; the resume adopted the 3 shots whose clips were on disk, regenerated the partial split shot 69 (all 3 parts), finished 120/120, 0 failed, 0 pods left | observed/mock: `evidence/crash_resume_capped.log` |
| **Kill during a progress save** | **FAIL, fixed here.** A signal inside `json.dump` left half a progress file; every resume then died with `JSONDecodeError` before doing anything, so capped_run would burn its 6 retries and exit 4 with the run dead until a human repaired the file. Now written atomically (`manifest.write_json_atomic`); the same kill leaves the previous save and the resume finishes. | observed/mock: `evidence/kill_in_save_before_fix.log`, `evidence/kill_in_save_after_fix.log` |
| **Stitch order** | **FAIL, fixed here.** `shots.json` stores shots out of film order (97 after 401, 334 after 97, 451 last: a retried stage 2 appended them), and `load_shots` did not sort. The stitch walked the stored order: shot 97 was placed 20 minutes late, shot 98 (1.79 s) was stretched to 8.95 s over 97's slot, and 334 and 402 were squashed to 0.08 s. On the full film 451 would land after the end credits and 452 would stretch over its gap. `load_shots` now returns index order; `encode_fingerprint` still hashes file order so stamped progress files (`8855b97c449f`, the reusable speech) keep matching. After the fix the same stitch puts 97 between 96 and 98, and 98, 334 and 402 come out at 1.80 s, 1.64 s and 3.44 s. Side effect: `--start-index`/`--limit` now count in index order, not file order. | observed: real stitch over 460 shots, `evidence/order_concat_before_fix.txt`, `order_concat_after_fix.txt`, `order_durations_after_fix.txt` |
| **capped_run limits** | **PASS**: `--cap 27 --margin 0.5` stops at $26.50 spent; the cap firing ended the child, swept, rc 3, and a rerun resumed and finished (rc 0, 300/300). Exit codes 0/3/4/5 as documented. Overshoot past the line is poll interval x burn rate plus balance lag: at $0.69/h and `--poll 90` about $0.02, plus RunPod's balance lag (unproved/live). The cap is RunPod only; fal (speech, keyframes) is not seen. | observed/mock: `evidence/cap_stop_and_resume.log`; `tests/test_capped_run.py` |
| capped_run sweeps every new pod on the account | **Fixed after this note (job `0443bd59`, mock-tested)**: the sweep now ends only pods the run's ownership journal claims, so `--output-dir $RUN` is required in the commands below (or `--sweep-account` for the old rule). Account exclusivity is still unconfirmed by Will, but the run no longer depends on it for the sweep. Original rule: nothing else may use the RunPod account during the run (0026 F7, task `29a4bc50`). Job `2ed396f1` (paid keyframe check) is running today and must be finished before 30 Nov. | source, `tools/capped_run.py` `sweep` |
| **Pod journal and exit sweep** (b7f7977, d61666c) | **PASS on mocks**, as the 0026 review found. The journal lives in `<output>/runpod_journal/`: the output dir must be persistent (not inside a job worktree that gets reaped), or a crash's claims are lost with it. | observed/mock: `tests/test_runpod_pod.py`, `tests/test_pod_review_scenarios.py`; source |
| **SSH key** (13cbaf1, task 76498e88) | **PASS locally**: `resolve_ssh_identity()` returns `~/.ssh/id_ed25519` on the Mac; on budgie `~/.ssh/id_ed25519` exists and is passphrase-less. Live behaviour **unproved/live** (job `2ed396f1` is the first real test). F6 (`fa27c947`) not landed. | observed; `tests/test_pod_ssh.py` |
| **Disk** | **PASS**: budgie 96 GB free, Mac 114 GB free. Expected output ~10 GB: clips ~2 GB (Seinfeld final ran 2.1 Mbit/s; SW4 LTX-2 clips were 0.66 MB/shot), adjusted ~2 GB, final ~2 GB, audio wavs ~0.5 GB each. Pod disk is 120 GB for 40 GB of weights. | observed (`df`), inferred (sizes) |
| **Speech reuse** | **PASS with a caveat**: `speech_progress.json` (530 lines done, 9 failed) carries fingerprint `8855b97c449f` and will be reused. The casting in `voice_map.json` has still not been heard by Will. | observed |
| **Stale audio in the SW4 dir** | **Risk**: the stitch auto-mixes every `audio/<strategy>/` dir that passes `_audio_matches_encode`; the Mac SW4 dir holds five from older runs. Copying only the input set to a fresh dir avoids it. | source, `stitch.py` `_mux_audio` |
| **What the job report will contain** | Defined below ("Job report"). The old list (first-50-shot checkpoint, live self-destruct verdict) no longer applies. | |

Looked for and not found: a duplicate or missing shot index (0-2069 all present once); a
frame count LTX-2.5 rejects; a prompt that fails to build; a personal name reaching the model;
a crash point in a split shot that resume mishandles; a pod surviving a cap stop or a crash
under capped_run (fake account).

## The command (keyframe OFF; the default until 236284e2 says otherwise)

On budgie, with keys from the proj-lossy vault (job `secrets=true`), and `RUN` an absolute
path outside any job worktree:

```bash
RUN=~/lossy-runs/star_wars_iv_v2

# 0. Free plan check (no pod, no network): 2070 shots, 2687 clips expected
uv run python decode.py $RUN --strategy runpod-ltx25 --dry-run

# 1. Video under the cap (exit 0 finished, 3 cap reached, 4 retries exhausted, 5 pod unconfirmed)
uv run python tools/capped_run.py --cap 27 --margin 0.5 --no-project --output-dir $RUN \
  --progress $RUN/decode_progress_runpod-ltx25.json --total-shots 2070 --shots $RUN/shots.json \
  -- uv run python decode.py $RUN --strategy runpod-ltx25 [--style-prefix "<B4b>"]

# 2. Ambience only if money is left: same cap, measured from step 1's starting balance
uv run python tools/capped_run.py --cap 27 --margin 0.5 --no-project --output-dir $RUN --baseline <step-1 baseline> \
  -- uv run python decode.py $RUN --audio --audio-strategy runpod-mmaudio

# 3. Stitch whatever exists (local ffmpeg, $0)
uv run python decode.py $RUN --strategy runpod-ltx25 --stitch
```

Keyframe ON instead: `--strategy runpod-ltx25-keyframe` (klein runs on the pod by default since task
0017: ~2.2 GPU-h, ~$1.4, inside capped_run; `--keyframe-generator fal --keyframe-fal-cap <USD>` is the
old fal route, about $19 outside capped_run), `LOSSY_FACE_MODELS=<dir with yunet.onnx and sface.onnx>`,
and the progress file `decode_progress_runpod-ltx25-keyframe.json`. Pass the matching progress file
to capped_run's `--progress`. Run its `--dry-run` first: it prints the routing and the pod-time estimate.

Input set to copy (from the Mac, `rsync -a` so mtimes survive):
`shots.json characters.json characters/ speakers.json voice_map.json scenes.json
speech/ speech_progress.json`. Leave everything else (old clips, audio, progress files from
other encodes) behind.

Style prefix recommendation (B4b, Will's call, not verified on SW4): `"Live-action 1977
science-fiction film footage of real actors and practical sets, photographed on 35mm film."`
It costs nothing and 0028 found no prompt carried any live-action anchor.

## Job report (what job 08db840b should hand back)

- Shots completed / failed / not reached, from `decode_progress_runpod-ltx25.json`; the
  `INCOMPLETE:` line decode prints; capped_run's exit code and why.
- Spend: capped_run's `FINAL` line (baseline, spent, balance); $/shot and min/shot against this
  note's ~$0.017-0.020/shot and ~1.7 min/shot (60 GPU-h over 2,070 shots); fal spend separately.
- Pods: `pods []` in capped_run's FINAL line, `runpod_journal/` empty, and a console check.
- The stitched (possibly partial) film path, the stitch's drift warnings count, and how many
  shots the letterbox strip cropped.
- The exact resume command if the cap stopped it.
- Anything that changes the go/no-go for the rest.

## Fixed here (mock-tested, no spend)

- `manifest.load_shots` returns shots in index order; `encode_fingerprint` keeps hashing file
  order. Tests: `tests/test_manifest.py::TestShotOrder`,
  `tests/test_decode.py::test_stitch_concatenates_out_of_order_shots_in_film_order`.
- Progress files are written atomically (`manifest.write_json_atomic`, all eight progress writes
  in `decode.py`). Tests: `TestWriteJsonAtomic`,
  `test_a_kill_during_a_progress_save_does_not_block_the_resume`.
- `decode.py --dry-run` now works for every LTX strategy (it was keyframe-only): builds each
  remaining shot's prompts and graphs and prints the clip count. Test:
  `test_ltx25_dry_run_plans_every_clip_without_a_pod`.

Each new test fails on main (except the fingerprint guard, which pins the old value).

## Not verified

- Anything a real pod or account decides: the account balance, real burn rate, balance lag,
  sshd boot race, ComfyUI behaviour (unproved/live; job `2ed396f1` is the next live evidence).
- A 60-hour run inside one Manager job phase (Seinfeld's 11 h is the longest so far).
- Stitch on the full 2,070 shots (the rehearsal stitched 460; the fixed defect is per-position,
  and the full-film case follows from the same code path).

## 22 Nov 2026: photographic identity blocks for SW4

Will's decision: the SW4 run uses the style prefix and identity blocks regenerated with the
photographic `IDENTITY_DESCRIPTION_SPEC` (research 0028).

**Premise check.** The spec is in the encoder (`encode.py`, `IDENTITY_DESCRIPTION_SPEC`, commits
e54fa16 / 8ed62c1: "A real man/woman", caricature intensifiers banned, droids as machines). The
SW4 `characters.json` on disk is dated 8 Aug, so it was written before that spec. [Direct]

**Why not re-run stage 3.** `encode.py stage3` rewrites the whole registry: roster, names and
shot lists, not only descriptions. The unsupervised path caps at 5 characters. SW4 has 26
hand-merged characters (`keep_name`, `red_two_wedge_antilles`, and so on). Re-running it would
change who is in which shot, not just how they read. So the job added
`tools/regen_identity.py`: it rewrites only `description`, with the same prompt shape stage 3
uses for stubs and the same spec. Roster, names, shot lists and `keep_name` stay as they are
(tests: `tests/test_regen_identity.py`). [Direct]

**Run.** The AI Studio key has no quota from the job (429, `quota_limit_value` 0, every model).
The same calls through Vertex worked:
`LOSSY_GEMINI_BACKEND=vertex LOSSY_ENCODE_MODEL=gemini-3.1-flash-lite`, 26 calls, cents. It ran
on a scratch copy (the job cannot write the primary checkout). All 26 roster entries, names,
shot lists and `keep_name` flags are identical to the backup. [Direct]

**Hand fixes** (`data/sw4_hand_fixes.py`; the raw model output regressed these):

| Block | What the model did | Fix |
|---|---|---|
| `darth_vader` | "A mechanical robot consisting of a ... helmet" | Armoured masked figure, no skin visible, cape kept |
| `stormtrooper` | "A Stormtrooper is a mechanical-looking soldier" (old block said "human soldier") | "A masked soldier in glossy white plastic armour", no skin or face visible |
| `c_3po` | Invented "protruding circular sensor array on the forehead" and an "asymmetrical faceplate" | Removed both |
| `chief_jawa` | "hunched posture" (a banned word) | Removed |
| `han_solo` | "light, natural stubble" (old block and shots say clean-shaven) | Clean-shaven |
| 8 pilots (`red_two/three/four/six`, `gold_leader/two/five`, `red_two_wedge_antilles`) | Each got an invented scar, mole or ear asymmetry, mostly the same "faint horizontal scar on his left cheekbone" | Removed. Seven pilots sharing one scar would render as one face |

After the fixes no block contains a banned intensifier (`very`, `sharp`, `angular`, `hunched`,
`mop`, `humanoid`, and so on), and every human block opens "A real man/woman". R2-D2, Chewbacca
and the Jawa read as machine and creature, not "a man in a costume". Known weakness: the eight
pilots are still near-identical (late-20s, light-brown hair, square face), as before. They are
told apart by shot, not by block. [Direct: blocks read; not rendered]

**Before and after** (full blocks in `data/sw4_characters_photographic.json`; the old ones are in
the backup):

| Character | Before | After |
|---|---|---|
| Luke | "young Caucasian male ... somewhat angular face with a prominent brow" | "A real man, age 20 ... light blonde hair cut in a short, layered style ... oval-shaped face, straight narrow nose, light blue eyes" |
| Han | "rugged, somewhat angular face ... often a slight smirk or determined expression" | "A real man in his mid-30s ... thick dark brown hair, slightly shaggy ... heavy dark brows, deep-set dark eyes, clean-shaven" |
| Obi-Wan | "kind, deep-set blue eyes and a generally gentle, wise expression ... light-colored robes" | "A real man, age 60-65 ... pale weathered complexion ... full neatly trimmed white beard, deep-set pale blue eyes, natural pores" |
| C-3PO | "A humanoid robot ... slender, anthropomorphic build" | "A mechanical robot with a polished gold-plated exterior ... round glowing yellow optical lenses ... exposed wiring, mechanical joints" |
| Darth Vader | "tall, imposing figure encased in a black, life-support suit" | "A tall figure in glossy black armour and a full black helmet ... triangular grille, dark angled lenses ... no skin visible" |
| Stormtrooper | "A human soldier wearing a distinctive white armored suit" | "A masked soldier in glossy white plastic armour ... black T-shaped visor, snout-like vent ... no skin or face visible" |

**Fingerprint.** `manifest.encode_fingerprint` hashes only the shot boundaries in `shots.json`,
and the tool only reads `shots.json`, so it stays **8855b97c449f** (computed on the current
`shots.json`). `decode.py <scratch dir with the new characters.json> --strategy runpod-ltx25
--dry-run` prints 2070 shots, 2687 clips, 26 characters. Budgie copy task f91deccb: the value to
check is still 8855b97c449f. [Direct]

**To put the new blocks in place** (the job could not write the primary checkout; Will runs this
on the Mac):

```
D=/Users/will/projects/lossy/output/star_wars_iv_v2
mkdir -p $D/characters.pre-photo-20261122 && cp $D/characters.json $D/characters.pre-photo-20261122/
cp docs/research/0030-sw4-full-run-preflight/data/sw4_characters_photographic.json $D/characters.json
```

Run it from the repo root. No clips, audio or progress files are touched.

## 23 Nov 2026: recast voices and a 2-minute mid-film test (task 462c92a3)

**Verdict: GO for 30 Nov on the planned look, with five changes first (below). The render is
photoreal and on-model for troopers and droids. The weak parts are not the video settings: speaker
attribution is wrong on a quarter to half of the lines, the character registry puts Uncle Owen and
Aunt Beru in ~100 shots after they die, and Chewbacca renders as an ape.** Spend: **$1.63 RunPod**
(balance $30.39 -> $28.76) **+ $0.05 fal speech + cents of Vertex**, under the $5 cap. **0 pods
left, both ownership journals empty** (`hygiene.txt`). Everything ran from a scratch copy of the
encode in the job workspace; the primary checkout was only read.

### Premise check (what the task assumed vs what is true)

- **R2-D2 and Chewbacca do not "speak English" because of casting; they speak because of
  misattribution.** Neither has a single subtitle line in the film. All 25 lines attributed to
  `r2_d2` and all 10 to `chewbacca` in `speakers.json` are other characters' lines (C-3PO's "At
  last. Where have you been?", Han's "Not a bad bit of rescuing, huh?", Dodonna's briefing).
  Giving them beeps or growls would have replaced real dialogue with noise. [Direct: every line read]
- **Arm A: LTX-2.5 already keeps its audio; the stitch threw it away.** `runpod-ltx25` decodes the
  audio latent into every clip (`strategies_video.py`, node 31), but `stitch._retime_clip` re-encodes
  each clip with `-an`, so the film never heard it. Fixed: `--audio-strategy native` (below).
  [Direct]
- **The first window I picked (1040-1056, the detention-block plan) could not test keyframes:**
  every shot holds a design character (troopers, droids, Chewbacca), so all 17 route to text
  (`--dry-run`). Moved to 1284-1321. [Direct: `dryrun_*.txt`]
- The encode has **2,070 shots**, not 2,012 (`shots.json`; fingerprint 8855b97c449f unchanged).

### The window: shots 1284-1321 (86:00-88:02, 122 s, 38 shots, 46 clips)

Chosen by scoring every 2-minute window between 30% and 70% of the film (skipping 684-707) for: a
C-3PO/R2 presence, a human two-hander, a design character, speaker count, and keyframe shots (by
dry-run). This one has the Han/Leia "walking carpet" argument, C-3PO on the comlink with R2 on
screen, Obi-Wan at the tractor beam, stormtroopers in most shots and Chewbacca. 14 keyframe shots
(Luke, Han, Leia, Obi-Wan sheets), 24 text shots (all design-character).

### Voices: old -> new

| Character | Old | New | Why |
|---|---|---|---|
| C-3PO | Sarah (woman) | **Daniel** (British male, precise, formal) | fussy, precise male |
| R2-D2 | Matilda | **null: non-speaking** | no subtitle lines exist; see below |
| Chewbacca | Callum | **null: non-speaking** | same |
| Chief Jawa | Charlotte | **null: non-speaking** | Jawas speak untranslated Jawaese |
| Luke | Charlie (shared with Red Leader, Red Two) | **Liam** (young American male) | split from Red Leader |
| Red Leader | Charlie | **Eric** (middle-aged American male) | split from Luke |
| Red Two | Charlie | **Will** (young American male) | split from Luke |
| Han | George (shared with Tarkin, Owen) | **Chris** (American male, casual) | split from Tarkin |
| Uncle Owen | George | **Bill** (older American male) | split from Tarkin |
| General Dodonna | Roger (shared with Obi-Wan) | **Brian** (deep American male) | split from Obi-Wan |
| General Willard | Adam (shared with Vader) | **Charlie** | split from Vader |
| Leia, Vader, Obi-Wan, Tarkin, Stormtrooper, Beru, narrator | Alice, Adam, Roger, George, Callum, Laura, Roger | unchanged | |

`voice_map.json` now allows `null` for a character who never speaks a human language. A line
attributed to one is voiced with the fallback voice (`--speech-voice`, Roger), never dropped.
R2-D2's and Chewbacca's own sounds come from the ambience track: the describe pass's `sound` field
mentions R2's beeps/whistles in **132 of his 269 shots**, but Chewbacca's roars/growls in only
**21 of 163**. So **R2 beeps in about half his shots (MMAudio or LTX audio); Chewbacca is mostly
silent**. In this window neither is asked for, so both are silent here. All voices are ElevenLabs
premade (bank widened in `voice_casting.py`), with no cloning and no plan change. The file is
`data/sw4_voice_map_recast.json`. Old map and `speech_progress.json` are backed up in the scratch
copy (`backup/`).

**Speech regenerated:** the window's 28 lines, 27 made ($0.05). **Line 982, "It's them. Blast
them.", was refused by fal's content checker** (`content_policy_violation`, category violence).
The full film will lose a few such lines unless they are retried with a reworded prompt.
**Full regeneration needed before 30 Nov: 728 of 1,298 lines** (every line of Luke 276, Han 187,
C-3PO 179, R2 25, Red Leader 20, Dodonna 12, Chewbacca 10, Willard 8, Owen 5, Red Two 5, Jawa 1).
That is about $1.50 of fal, and only after the attribution fix below, or the new voices land on
the wrong lines.

### Two encode defects found on the way (both fixed for the window, data committed)

1. **Speaker attribution is wrong on 25-57% of lines.** Hand-checked against the film: lines
   797-824 **21/28 right (75%)**, lines 959-986 **12/28 right (43%)**. Typical errors are swapped
   turns in a two-hander and troopers' banter given to Obi-Wan. A recast cannot fix this: Han's
   new voice still says Leia's lines. Fixes for both windows: `data/sw4_speaker_fixes.json`.
   Stage 4 now has a rule against attributing subtitles to non-speaking characters; a full fix
   needs a re-run of stage 4 (cents) and a spot-check of accuracy before the speech pass.
2. **The registry puts dead characters in later shots.** Uncle Owen is tagged in 75 shots after
   his last living appearance (shot 402, 25.5 min), and Aunt Beru in 32. In this window that
   includes Han's shots 1285, 1307 and 1314. Keyframe-first would draw them as an extra person.
   Tarkin was tagged at the tractor-beam panel (1291, Obi-Wan's shot). Removed via
   `data/sw4_character_shot_fixes.json`; apply it to the full-run copy.

### Measured (planned arm unless noted)

| Measure | Result |
|---|---|
| Pod | **secure A40 @ $0.63/h** for both video arms; secure 4090 @ $0.61/h for MMAudio. **No community A6000/A40/L40 was free** on any of 3 rents. |
| GPU time per clip | full res **91.8 s**; one-stage 640x352 **47.0 s (0.51x)**; klein 7.3 s per image |
| $/shot (this window, incl. pod boot + 40 GB download) | full res **$0.023**; low res ~$0.016 (2 boots); MMAudio $0.004 |
| min/shot | full res **1.85 GPU-min** (2.2 wall-min with boot); low res 0.97 GPU-min |
| Photoreal (0028 `classify_style.py`, 1 frame/clip) | full res **43/45** (1 unparsed); low res **43/46**. Every miss is a Chewbacca shot ("3d_animation", lookalike "Smallfoot") |
| Cartoon drift | none on humans, troopers or droids. Chewbacca only |
| Face cosine across shots / to sheet (`window_faces.py`, single-person shots) | Han 0.20 / 0.36 (5 shots), Obi-Wan 0.17 / 0.26 (6), Leia 0.26 / 0.44 (3). Low res: same within 0.04 across shots, 0.04-0.06 lower to sheet. Below 0029's Seinfeld 0.55 because most Obi-Wan/Han shots here share the frame with troopers and route to text |
| Wardrobe | **Leia bare-shouldered in her close-up (1286)** and in a different outfit in most shots (white, black dress, dark jacket). Han's vest mostly right. Troopers correct |
| Droids / creatures | Troopers excellent; C-3PO gold with a grille "mouth"; R2-D2 right. **Chewbacca is a shaggy ape with a human-ish face** |
| Low res vs full | same composition (same seed); softer at full size after Lanczos; near-indistinguishable at thumbnail (`res_compare.jpg`) |
| Audio | all four versions carry audio: native -14.8/-15.1 LUFS, MMAudio -16.0 LUFS. Not judged by ear |

### Cost options for the full film (2,070 shots, 2,687 clips)

Extrapolated from this window's GPU seconds per clip (clip length here 2.74 s; film 2.78 s), plus
klein for 795 keyframe shots. MMAudio is $8.0 from this window's $0.147 for 38 shots. Secure A40
at $0.63/h, measured; community A40 at $0.35/h in brackets, never available today.

| Setup | $/shot | Full film | Notes |
|---|---|---|---|
| Planned (full res + MMAudio) | $0.025 ($0.014) | **$52** ($29) | video $43 (68.5 GPU-h) + klein $1 + MMAudio $8. Does not fit the $28.76 balance on secure |
| Native audio (full res, no MMAudio) | $0.021 ($0.012) | **$44** ($25) | saves MMAudio's $8 and ~13 pod-hours |
| Low res + MMAudio | $0.015 ($0.008) | **$31** ($17.5) | video $22 (35 GPU-h) |
| Low res + native audio | $0.011 ($0.006) | **$23** ($13) | cheapest; same photoreal rate |

### What should change before 30 Nov

1. **Re-run stage 4 and spot-check it** (cents), then regenerate the 728 recast lines (~$1.50).
   Without it the recast is wasted on misattributed lines.
2. **Apply `sw4_character_shot_fixes.json`** to the run's `characters.json` (Owen/Beru/Tarkin).
3. **Pick the money option: the planned setup does not fit the balance on secure cloud.** Low res +
   native audio (~$23) or low res + MMAudio (~$31) fit; full res fits only on community cloud,
   which was never available today. Will's call; both versions of the window are there to compare.
4. **Chewbacca:** his block reads as "ape" to LTX. Rewrite it toward "a 7-foot Wookiee, tall
   upright fur-covered alien, long brown hair over the whole body, blue eyes, bandolier" with no
   ape words, or accept it.
5. **Leia's wardrobe:** her block or her shots' prompts need "white high-necked long-sleeved gown"
   so the close-up is not bare-shouldered.

### To put the new casting and fixes in place (Will, on the Mac, from the repo root)

```
D=/Users/will/projects/lossy/output/star_wars_iv_v2; R=docs/research/0030-sw4-full-run-preflight/data
mkdir -p $D/backup-20261123 && cp $D/voice_map.json $D/speakers.json $D/characters.json $D/speech_progress.json $D/backup-20261123/
cp $R/sw4_voice_map_recast.json $D/voice_map.json
uv run python -c "
import json; s=json.load(open('$D/speakers.json')); f=json.load(open('$R/sw4_speaker_fixes.json'))['fixes']
for k,v in f.items(): s['assignments'][k]={'character':v,'confidence':1.0,'source':'hand'}
json.dump(s,open('$D/speakers.json','w'),indent=2)
c=json.load(open('$D/characters.json')); r=json.load(open('$R/sw4_character_shot_fixes.json'))['remove']
for ch in c['characters']:
    ch['shots']=[x for x in ch.get('shots',[]) if x not in set(r.get(ch['name'],[]))]
json.dump(c,open('$D/characters.json','w'),indent=2)"
```

Then drop the recast characters' lines from `speech_progress.json` (or move `speech/` aside) so the
next speech pass regenerates them.

### Where things are (job workspace, outside git)

- Test clip (planned): `evidence/test_clips/star_wars_iv_v2_reconstructed_runpod-ltx25-keyframe+runpod-mmaudio.mp4`
- Arm A (LTX audio + recast speech): `...+native.mp4`; Arm B: `star_wars_iv_v2_lowres_...` (both audios)
- Side-by-side with the source: `evidence/side_by_side.mp4`; contact sheet `evidence/contact_sheet.jpg`;
  resolution stills `evidence/res_compare.jpg`; logs, `faces.txt`, `style_*.jsonl`, `hygiene.txt`

### Code added (tested, `tests/test_stitch_native_audio.py`, `test_ltx25.py`, `test_decode.py`)

- `decode.py --stitch --audio-strategy native`: each clip's LTX soundtrack becomes the ambience track
  (`stitch.extract_native_audio`, idempotent, never auto-mixed into a plain stitch).
- `--ltx-one-stage` (0025's 640x352 graph, dry-run aware) and `--stitch --scale WxH` (Lanczos).
- `voice_map.json` `null` = non-speaking; wider voice bank; stage 4 non-speaker rule.
- `window_test.sh` runs the whole test (balance gate, speech, both arms, MMAudio, four stitches,
  comparisons, hygiene) under one cap and resumes after a stop; `window_faces.py` measures faces.

## 23 Nov 2026: SW4 run dir on budgie (task f91deccb, blocker B1 closed)

`budgie:~/lossy-runs/star_wars_iv_v2/` now holds `shots.json`, `characters.json`, `characters/`,
`speakers.json`, `voice_map.json`, `scenes.json`, `speech/` (530 files), `speech_progress.json`
(rsync -a, mtimes kept; contents byte-identical to the staged copy). Face models are at
`budgie:~/lossy-runs/face_models/` (`yunet.onnx`, `sface.onnx`, from opencv_zoo): set
`LOSSY_FACE_MODELS=$HOME/lossy-runs/face_models` for the run. [Direct]

- `characters.json` is the **photographic** one (`data/sw4_characters_photographic.json`) with
  `sw4_character_shot_fixes.json` applied (108 stale shot tags removed: Owen, Beru, Tarkin). The Mac
  primary still has the old one (Aug 8). Not applied: the recast `voice_map.json`, the speaker fixes
  and the speech regeneration, so `voice_map.json` / `speakers.json` / `speech*` are the old Mac
  copies. The "To put the new casting..." recipe above still has to run on the run dir before speech.
- Fingerprint on budgie (`manifest.encode_fingerprint`): **8855b97c449f**, as expected.
- `decode.py --strategy runpod-ltx25 --dry-run` on the staged copy: 2070 shots, 2687 clips, 26
  characters. Run from this branch's code: budgie's checkout (`dd4cd9d`, 6 behind main) was not
  updated, since only `~/lossy-runs` was approved for writes. Pull it before the run.

## 23 Nov 2026: Chewbacca and Leia identity blocks rewritten (task 12b72ad4)

Items 4 and 5 of "What should change before 30 Nov" are done in `data/sw4_characters_photographic.json`
(and reapplied by `data/sw4_hand_fixes.py`). Chewbacca: "seven-foot-tall Wookiee, an upright
fur-covered alien", long brown hair over the whole body, blue eyes, bandolier; the old "primate-like",
"mane", "clawed" and "prominent teeth" wording is gone. Leia: the block now says a plain white,
floor-length, long-sleeved gown with a high rounded neckline, neck, shoulders and arms covered.
Guarded by `tests/test_sw4_identity_blocks.py` (word checks only). **Not rendered**: whether LTX now
draws them right needs a pod; the run dir on budgie still carries the old blocks until the spec is
copied over its `characters.json`. [Direct: text only]
