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

## 22 Nov 2026: photographic identity blocks for SW4 (not finished)

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

**Fingerprint.** `manifest.encode_fingerprint` hashes only the shot boundaries in `shots.json`,
and the tool only reads `shots.json`, so it stays **8855b97c449f**. Computed on a copy of the
current `shots.json`. [Direct]

**Blocked: no Gemini quota.** From the job, every Gemini model (`gemini-2.5-flash-lite`,
`gemini-3.1-flash-lite-preview`, `gemini-2.5-flash`, `gemini-2.0-flash`) returns 429
RESOURCE_EXHAUSTED with `quota_limit_value: 0` (`asia-southeast1`). The key is present in `.env`
but this project has no quota from the job's network location. The regeneration, the
before/after diff of six characters and the droid/creature/armour check have therefore not been
done. To finish, from the Mac:

```
uv run python tools/regen_identity.py /Users/will/projects/lossy/output/star_wars_iv_v2
```

It backs up `characters.json` to `characters.pre-photo-YYYYMMDD/` first. `--dry-run` prints the
prompts without calling Gemini. Cost is about 26 flash-lite calls (cents).
