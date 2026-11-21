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
| Length | **1382.48 s (23.04 min)** vs the 1382.42 s source: all 315 shots. Largest single-shot drift 0.087 s (about 2 frames at 25 fps), sum of drifts 0.02 s. 1280x704, 25 fps, h264 + aac. |
| Audio | per-character speech (419 lines, 4 voices + narrator) + MMAudio ambience (281 shots generated, 34 speech-only shots skipped, 0 failed) |
| RunPod spend | **$9.21 in total**: $7.2776 for the first run (shots 0-298, no ambience) + **$1.9325** for the finish job (video 299-314 $0.69, ambience $1.24; cap $2.50). Balance $7.3799 -> $5.4474. |
| Other spend | fal.ai speech $0.75, Gemini encode about $0.17 |
| Pods left | none: `myself{pods{id}}` is `[]` and `currentSpendPerHr` is 0 after the finish job |
| Pods left | none: `myself{pods{id}}` returned `[]` after the final terminate |
| Pace | about 2.3 min per shot over the run (299 shots in about 11.2 h incl. 20 min boot), about $0.024 per shot |

Stills: `contact_sheet.jpg` (5 frames from the first, partial cut, at 45, 330, 640, 940 and 1210 s) and
`contact_sheet_final.jpg` (8 frames from the finished episode, at 45, 330, 640, 940, 1210, 1270, 1330 and
1380 s; the last one is the ending, a lighthouse at dusk).

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

**Bug, closed 15 Nov 2026** (both halves of the original line):

- *Re-provisioning did not terminate the pod it replaces.* Fixed earlier by 05059f3:
  `forget_pod`, setup retry and re-create all terminate first. A mock fleet test
  (`test_the_wedged_pod_is_terminated_before_its_replacement_is_created`) records the
  order `create pod-1, terminate pod-1, create pod-2`.
- *A hung poll has no timeout / the hung job is never stopped.* The 300 s timeout already
  existed, but on timeout nothing cancelled the ComfyUI job. ComfyUI runs one job at a time,
  so every retry queued behind the dead job and could only time out too. Measured with
  `TestHungComfyUIJob` (mocked RunPod API, ComfyUI and clock; no pod, no spend):
  before the fix a never-finishing job was abandoned after **16.0 min** (3 x 300 s + 20 s + 40 s
  backoff), on the same pod, with the stuck job still in the queue.
  Now: `submit_workflow` drops the job from the queue and sends `/interrupt` on timeout (or on
  a poll error) and checks `/queue`. If the job will not stop, the pod is marked `wedged`, and
  the clip loop replaces it (old pod terminated first). Bounds, all asserted in tests:
  - interrupt works but the clip hangs every time: abandoned in **<= 17 min** (3 attempts), 1 pod;
  - interrupt does not free the worker: pod replaced after **one** 5 min timeout, the clip then
    succeeds on the new pod;
  - every pod wedges: clip abandoned in **<= 11 min** after 2 pods, both terminated, zero live pods
    (the next shot provisions a fresh one).

The audio strategies (`RunPodMMAudioStrategy`, `RunPodMusicGenStrategy`) now act on `wedged`
too (16 Nov 2026): `WedgedPodReplacementMixin` in `strategies_audio.py` applies the video
policy to each audio clip: the first wedge ends the pod and provisions one replacement, the clip
is retried there, and a second wedge on the same clip (`CLIP_WEDGE_STRIKES = 2`) abandons it with
the pod dropped. Before this, a wedged audio pod was only caught by `run_audio`'s "Too many
errors" stop and the exit sweep. Mock tests: `TestWedgedPodInAudioStrategies`
(`tests/test_runpod_pod.py`, both strategies). Ownership, overlap and a failed terminate are
unchanged (tasks 243f1be5 and 9f7ea2a6). The mock cannot show a real ComfyUI's `/interrupt` and
`/queue` payload handling (written from the ComfyUI API as documented; not run against a pod).

## Cost, and why there is no ambience

(Written after the first run; the finish job below did run the ambience pass.)
Video alone took $7.25 of the $8 cap at 94.8% of shots. The ambience pass (RunPod MMAudio,
about $0.8 in the research 0025 projection) would have crossed the cap, so it was not run.
The 0025 projection of about $4.50-7 all-in under-called this run by roughly $1-2: the fit
above says the A6000 secure rate and 2.3 min/shot, not the 8 GPU-hour figure, are the real
numbers for LTX-2.5 at 1280x704 on this episode.

## Finish job (8 Nov): shots 299-314, ambience, re-stitch

Will approved about $2 more, hard cap $2.50 from a $7.3799 balance, run under `capped_run`
(`--no-project`, `--margin 0.3`, `--baseline` fixed at the starting balance so both stages count
against one cap).

| Stage | Result | Spend |
|---|---|---|
| Video, shots 299-314 | 16 shots, 1 pod (L40 at $0.69/h; the A6000 was unavailable), 1.0 h in all: about 14 min boot, then 49 min of clips | $0.69 |
| Ambience (RunPod MMAudio) | 281 clips, 34 speech-only shots skipped; two pod sessions on an L40S at $0.54/h (the first one aborted, see below) | $1.24 |
| **Total** | cap $2.50, stop line $2.20 | **$1.9325** |

- **Resume worked once the progress file's mtime was right.** The first run's folder was copied into
  this worktree; `cp -R` reset mtimes, so `check_encode_fingerprint` saw a progress file older than
  `shots.json` and refused it. Restoring mtimes (`touch -r`) fixed it. The progress file lagged by
  nothing this time (it already held 0-298, including the four shots added by hand).
- **The ambience pass aborted at shot 199 with "Too many errors"** after 21 speech-only shots. A
  shot whose sound description is all speech makes `RunPodMMAudioStrategy` / `MMAudioStrategy`
  return `[]` on purpose; `run_audio` counted that as a failure, slept 10 s, retried, and stopped
  after 20. Fixed: strategies that filter speech set `filters_speech = True` and `run_audio` records
  those shots as skipped before calling them (test `TestRunAudioSpeechOnlyShots`). The pod was
  terminated on exit; the resume paid a second boot (the two sessions billed about $0.45 and $0.36 for 179 and 102 clips).
  This would have hit any episode with 21+ dialogue-only shots, so Star Wars IV would have hit it too.
- **Per-shot length check.** All 315 shots have an adjusted clip; the largest deviation from
  `shots.json` is 0.087 s, and the stitch's own drift warning fired for 102 shots, all within about 2
  frames. The muxed audio is 0.044 s shorter than the video (1382.436 s vs 1382.480 s).
- **Pace.** Video 16 shots in 49 min after boot (about 3 min per shot, a little slower than
  the first run's 2.3; these tail shots are longer on average). Ambience about 4 shots per minute.
- **Quality.** The style drift reported above continues in the tail: `contact_sheet_final.jpg` has 3D
  cartoon shots (330 and 940 s) next to photoreal ones, and the stills at 1210-1330 s look like a
  different well-known drama rather than the source show. Not fixed here; the style clause is still
  the first thing to try before a second episode. [8 stills only, not a survey]
- **Ambience level.** Mean loudness of the finished mix is -18.5 dB with peaks at 0 dB; nobody has
  listened to it yet.
- **Not run:** a hung ComfyUI poll was not seen this time (no stall); the bug above was
  closed afterwards with mock tests only, not observed on a live pod.

The partial first cut is kept as `partial_shots0-298_no_ambience.mp4` next to the finished file.

## Addendum (16 Nov 2026): final-mix peak fix; re-stitch not run

The "peaks at 0 dB" finding is a code defect in `stitch.py`, now fixed (research 0023 defect 12):

- The single-track path (the one Seinfeld took) applied `volume={vol}dB` (speech is +6 dB) with **no limiter at all**.
- The multi-track path's `alimiter=limit=0.95` did not hold either: `alimiter` defaults to `level=1` (auto-level), which re-normalises the output back up after limiting. On a synthetic hot pink-noise mix the old chain measured +0.6 dBFS before AAC and +0.2 dBFS after, with the limit at 0.95.
- Both paths now end in one chain (`_true_peak_limiter`): resample to 4x, `alimiter=limit=0.8:level=disabled`, resample back. The same synthetic mix measures about -1.6 dBFS after AAC (ebur128 peak=true), and `tests/test_stitch_limiter.py` asserts <= -1 dBTP on both paths.

**Not run:** the re-stitch of Seinfeld. `output/seinfeld_s01e03` is gitignored and lives on Will's Mac; it does not exist on the machine this fix was made on, so there is no measured true peak or loudness delta for the real file. To check it, re-stitch from the existing clips to a new filename and run `ffmpeg -i <new.mp4> -af ebur128=peak=true -f null -`: expect true peak <= -1 dBTP and integrated loudness within about 1 LU of the old file (the limiter only acts on the loudest peaks, but the speech boost may now be clipped slightly harder than before).

## Sceptical review, 15 Nov 2026: a pod can still leak

**Verdict: NO-GO for an unattended full-film paid run.** The premises are partly
right: `08a72e8` and `dfe0a70` are ancestors of this checkout, and cancel/replace
and limiting exist. They do not establish crash-safe cleanup or a universal
small loudness change. This review used mocks and local ffmpeg only: no keys,
API calls, pods, or spend. The preceding addendum is dated 16 Nov; this review
does not treat that date or its proposed real-file check as completed evidence.

Confidence labels below: **observed/mock** means exercised with a fake account;
**observed/local** means local ffmpeg or installed SDK inspection; **source** is
a code-reading conclusion; **unproved/live** needs a real host/account.

The data shape is `RunPodSession`: optional active `pod_id: str`, append-only
`_owned_pod_ids: set[str]`, unique session name, and `wedged: bool`.
States are unprovisioned, provisioning (possibly an unknown server-side id),
active, wedged, abandoned, and terminated. Abandoning a handle must retain
cleanup ownership; an API request is not confirmation that billing has stopped.
`runpod_pod.json` is connection state for one pod, not an ownership journal.

### Creation and terminate-path inventory

There is one production `runpod.create_pod` call, in `RunPodSession.create_pod`.
`ensure_pod` reaches it after a failed reconnect; `with_setup_retry` can repeat
it. WAN, VACE, LTX-2, LTX-2.5, MMAudio and MusicGen all use this session rather
than allocating independently (`strategies_video.py`, `strategies_audio.py`).
Tests below are in `tests/test_runpod_pod.py` unless otherwise specified.

| Creation/use path | Matching terminate path and named evidence | Confidence / limit |
|---|---|---|
| Initial create; each GPU/cloud fallback request | `_cleanup -> terminate`, then `sweep_stray_pods`; `TestNoAbandonedPodStaysLive.test_normal_exit_leaves_nothing_live` | observed/mock |
| Create succeeds server-side but response times out (no id) | Register handlers before requesting; `_create_attempted` enables name sweep; `TestPodLeakReview.test_unknown_id_create_failure_registers_exit_cleanup` and `test_a_create_that_timed_out_client_side_is_still_swept` | observed/mock; eventual visibility remains unproved/live |
| Runtime readiness timeout | `create_pod -> terminate`; `test_provision_timeout_terminates_the_pod` | observed/mock |
| Setup/model/ComfyUI failure, followed by fresh create | `with_setup_retry -> terminate`; `test_setup_retry_after_timeout_ends_with_one_live_pod_then_none`, `test_every_setup_attempt_failing_leaves_nothing_live` | observed/mock; same shared lifecycle for all backends |
| Re-create over a held handle | `create_pod -> terminate` before allocation; `test_create_pod_over_a_held_pod_terminates_the_old_one` | observed/mock |
| Missing ports, dead host or wedged worker replacement | `_reprovision_if_pod_gone -> forget_pod -> _terminate_quietly`; `test_reprovision_terminates_the_pod_it_replaces`, `TestHungComfyUIJob.test_the_wedged_pod_is_terminated_before_its_replacement_is_created` | observed/mock; successful terminate precedes replacement |
| Workflow timeout or polling exception | `submit_workflow -> cancel_job`, delete queue item, interrupt, confirm absence; if still held mark wedged and replace; `TestHungComfyUIJob` covers cancellation, replacement and two-wedge abandonment | observed/mock; cancellation alone keeps a usable pod alive intentionally |
| Exhausted wedged-clip retries | `clip_wedges_pods -> forget_pod`; `test_everything_hanging_on_every_pod_still_ends_with_zero_live_pods` | observed/mock; two wedges bounded per clip call, not per entire run |
| Ordinary unhandled exception | registered `atexit -> _cleanup`; `test_crash_mid_job_leaves_nothing_live` exercises cleanup body | source for interpreter exit dispatch; observed/mock for body |
| KeyboardInterrupt/SIGINT or SIGTERM | registered handler calls `_cleanup` before previous handler/exit; `TestPodLeakReview.test_sigint_terminates_the_pod_then_raises_keyboardinterrupt`, `test_sigterm_terminates_the_pod_then_exits` | observed/mock |
| Clean `--keep-pod` exit | armed remote self-destruct, state file; fail closed to terminate if arming fails; `test_a_kept_pod_with_an_armed_reaper_survives_the_exit`, `test_a_kept_pod_whose_reaper_could_not_be_armed_is_terminated` | observed/mock; actual reaper deletion unproved/live |
| Cap/projection firing; failed child; interrupted supervisor | process-group TERM, bounded grace, KILL, account sweep; `tests/test_capped_run.py::TestInterruptedSupervisor` and `TestFinalSweep` | observed/mock; includes actual local sleeping children, no remote processes |
| SIGKILL, host power loss, prolonged API outage, late server-side creation | no guaranteed terminate; no durable all-pod journal or startup orphan reconciliation | source; follow-ups below cover these missing guarantees |

**Fixed here, observed/mock:** single transient terminate errors now retry three
times before abandoning; unknown-id create failures now register and enable
exit cleanup; keep-pod fails closed when its deadline cannot be armed;
supervisor SIGTERM/SIGHUP/Ctrl-C stops its detached child and sweeps; final
listing errors retry; unsuccessful cleanup is not reported as exit 0; unexpected
exceptions also sweep in `finally`. Regression tests fail against `08a72e8`.
The fleet assertions measure pods still live, not just calls to terminate.

**Replacement and persistence, source:** terminate is attempted before a new pod
is created, never after waiting for the replacement. Exhausting terminate retries
still permits replacement, so overlap can last until exit or longer. Owned ids
remain in memory, but `forget_pod` and `terminate` delete connection state even
on failure; newly created pods are not automatically journaled to disk. A new
session gets a different name and does not automatically adopt old orphans.
The installed RunPod SDK uses `requests.post(timeout=30)`; this is a socket
timeout, not a hard end-to-end deadline. Three attempts plus backoff can exceed
the supervisor's 20-second child grace; the account sweep is the backstop.
The standalone session sweep still makes one listing/termination pass and marks
it swept even on failure. Keep-pod skips that sweep, including abandoned siblings.

**Spend, observed/mock + source:** `capped_run` uses original account balance
minus current balance, not the active pod's elapsed time. Replacement time and
overlap therefore count when charges reach the balance. The new cap test places
both old and replacement pods in the account and checks both disappear at the
stop line. The per-session printed estimate resets with each create and is not
the cap. Balance lag, polling/grace time, account credits/concurrent users, and
indefinitely failing balance reads mean this is not a strict dollar ceiling.
Both the supervisor's account-wide sweep and `--attach-pid` can delete another
run's pods; the session sweep itself preserves concurrent session names.

**Retry/resume, observed/mock + source:** exhaustion returns no clip, and decode
writes the shot index to `failed`, never `completed`; existing partial clip
files can remain. `test_exhausted_shot_is_recorded_failed_and_retried_on_resume`
in `tests/test_decode.py` drives failure, saved progress and successful resume:
the shot is retried and its failure clears. Decode also retries the whole shot
once, so the two-wedge limit can be paid twice per shot. `run_decode` can return
normally with failed shots, so supervisor exit 0 is not proof of a complete film.
A/B manifest jobs use their own failed/completed lists and exit nonzero on failure.

**Limiter, observed/local:** both single-track and multi-track graphs are now
tested with full-scale square input before AAC and deterministic hot noise after
AAC. Material below the limiter ceiling stays within **0.2 LU** of the unlimited
graph. A blanket **1 LU** programme bound is false for hot input; strict expected
failure tests retain this counterexample (stereo square at +6 dB: single-track
8.5 -> -1.5 LUFS, multi-track 10.0 -> -1.5 LUFS on this ffmpeg build).
Full-scale square input is held below
-1 dBTP before AAC but exceeds that ceiling after AAC; strict expected failures
retain both paths' codec counterexamples. Seeded hot noise also measured
-0.6 dBTP after multi-track AAC. The older random-noise test also produced an
intermittent -0.8 dBTP failure; it now uses a fixed seed for reproducibility.
Evidence: workspace `evidence/final-tests-green.txt`
and `evidence/limiter-review.txt`; tests: `TestFullScaleInputAndLoudness`.
No new limiter policy is chosen here. Real episode loudness/true peak and audible
quality remain unproved; synthetic tests do not establish them.

### Follow-ups to file

Dispatcher: file under goal `b33ea4e0`; do not dispatch automatically.

- **Persist pod ownership and reconcile crash leftovers (M). DONE: ADR 0010 + `pod_journal.py`, mock-tested.** Journal every
  allocation/adoption and unresolved terminate, use a stable run identity, and
  reconcile before creating on restart. Include unknown-id requests, late API
  success, SIGKILL/power loss and kept-pod sibling leaks. Needs an ownership and
  reconciliation policy; no Will interaction or spend for implementation.
- **Bound replacement overlap and confirm deletion before declaring clean (M). DONE: ADR 0010 addendum, mock-tested.**
  Decide whether to stop allocating after failed terminate; retry/confirm the
  standalone exit sweep and cover interrupted cleanup/state-write failure.
  Mock persistent API outage and delayed listing visibility. No Will or spend
  for design/implementation; actual deletion timing belongs to the trial below.
- **Make cap enforcement fail closed on stale balance and scope account sweeps (M). DONE (job `0443bd59`, mock-tested; ADR 0010 addendum "capped_run"). Account exclusivity and cap semantics still need Will.**
  Current polling can fail forever; balance lag is unbounded, a watcher exception
  can leave the child unsupervised, and sweeping all new pods assumes exclusive
  account use. Define maximum stale interval, ownership, launch/stop coordination
  and a conservative spend reserve. No spend; Will must confirm account exclusivity
  and acceptable cap semantics before relying on this unattended.
- **Make incomplete decode visible to supervisors (S).** Return nonzero when
  shots remain failed and document the whole-shot versus clip retry budget;
  inspect partial-file adoption and duplicated failure entries on resume.
  No Will or real-pod spend required.
- **Set final audio loudness and AAC true-peak policy (M).** The 1 LU bound and
  universal -1 dBTP AAC bound are disproved. Decide gain/normalization/headroom
  policy, then retain full-scale and representative programme tests for both
  paths. Needs Will's listening/target decision; no API or pod spend.
- **Re-stitch and measure Seinfeld programme audio (S).** Existing source clips
  are absent here. On the machine holding them, compare integrated LUFS and
  post-AAC true peak with the old file, and listen. Needs Will/source-file access;
  estimated pod/API spend **$0** (local ffmpeg only).
- **Observe cancel, replacement, billing and remote self-destruct on a real pod (M).**
  After the safety follow-ups, force one timeout, verify the old id disappears
  before replacement, check account billing and final empty inventory, then
  prove a kept pod's remote reaper removes it after the client exits. Needs Will's
  paid-run authorization and credentials in a separate job. Estimate **$0.50-$1.25**
  including boot/deadline time; use the planned **$1.25** keyframe cap. Do not
  infer these guarantees from mocks or proceed directly to the full film.

## Review, 21 Nov 2026: the pod journal, exit sweep and ssh changes before the 30 Nov run

**Scope.** `b7f7977` (ownership journal, ADR 0010), `d61666c` (refuse to rent, confirmed
retrying exit sweep, state-write tolerance) and `13cbaf1` (one verified ssh key,
sshd probing). The Star Wars IV run (job 08db840b, 2,069 shots, $27 cap) is on 30 Nov.
Mocks and reading only: no keys, pods or spend. Premise check: all three commits are on
main as the task says; the task's "pod SSH reliability lands just before this" is `13cbaf1`,
and it is **mock-tested only** (its commit says so).

**Verdict: GO for the journal and exit sweep as designed, with two items that must land
first (F5, F6).** No scenario produced a pod that two runs would leave billing *when the
RunPod API behaves as the fake account does*. The defects are at the edges where the
real API may not (F2), around the supervisor (F1, fixed), and in operability (F5, F6).

Confidence labels: **observed/mock** = run against `_FleetRunpod`; **source** = read in code;
**unproved/live** = needs a real account. All runs are in `tests/test_pod_review_scenarios.py`.

### What was sound (observed/mock)

| Scenario asked about | Result | Evidence |
|---|---|---|
| Hundreds of replacements, journal growing | 300 `forget_pod` + `create_pod` cycles: exactly one pod live at every step, journal constant at 224 bytes (one claim, no windows), 300 terminate calls; `_owned_pod_ids` grows to 301 ints in memory (harmless). Claims are dropped on confirmed terminate, so the file only grows while terminates fail, and then create refuses. | `test_300_replacements_keep_the_journal_constant_and_one_pod_live` |
| Crash mid-replacement | Process killed at each of the first 12 journal writes / API calls of `forget_pod` + `create_pod` (the sequence has 9; later points are the finished case): the next run rents exactly one pod and ends with none, every time. | `test_a_crash_at_any_point_of_a_replacement_is_settled_by_the_next_run[1-12]` |
| RunPod terminate outage, "can refuse-to-rent stall the run" | Never rents across 7 restarted processes (capped_run `--retries 6`), carries one journal (not one per process), recovers on the first process after the outage. The stall is real, see F5. | `test_a_terminate_outage_never_rents_across_a_restart_storm_and_recovers` |
| Journal on a full disk | Terminate still happens, nothing stays live; `create_pod` refuses (fail closed) until space returns, then rents. | `test_a_full_disk_never_stops_a_terminate_and_only_blocks_renting` |
| `get_pod` raising "not found" for a gone pod (instead of returning None) | A terminate that landed but timed out does not block the replacement and the exit confirms. | `test_get_pod_raising_for_a_gone_pod_does_not_block_or_unconfirm` |
| Two runs at once, different output dirs | Journals are per dir and sweeps match by name or id: neither touches the other (existing `test_journals_of_a_different_output_dir_are_not_this_runs_business`). Different dirs are still unsafe under `capped_run`, see F7. | existing tests, source |
| Concurrent audio | Not a second session: audio runs over ssh on the video pod from a thread (`strategies_video.py` `_generate_audio_ssh`), so there is one journal and no signal-handler-in-thread problem. | source |

### Findings

Severity: **High** can lose money unattended; **Medium** breaks the run or a stated guarantee;
**Low** edge or cosmetic. "Before 30 Nov" = must land (or be worked around) before the paid run.

| # | Finding | Sev. | Confidence | Before 30 Nov | Status |
|---|---|---|---|---|---|
| F1 | **Exit cleanup outlasts the supervisor's grace.** `terminate` (3 attempts + 9 s backoff) runs before the exit sweep, and `CLEANUP_BUDGET_S` (90 s) is only checked between sweep passes. With every request costing the SDK's 30 s timeout, a total outage measured **199 s** of cleanup against `CHILD_GRACE_S = 120`, so `capped_run` SIGKILLed the child mid-cleanup. The existing test only compared two constants. The claim stays journaled, so nothing leaks, but the "confirmed exit" never completes. | Medium | observed/mock | yes | **Fixed:** `CHILD_GRACE_S = 240`; the new test measures cleanup time against the grace. Slow-connect hosts can still exceed any grace (requests may take 60 s per call), so the journal remains the real guarantee. |
| F2 | **"Not found" on a pod we just created is taken as proof it is gone.** If the API cannot see a new pod for longer than `POD_READY_TIMEOUT` (600 s), `terminate()` gets "not found", `_release` drops the claim, and the exit sweep reports "confirmed" while the pod appears later and bills with no claim and no window (the window was closed once the id was journaled). | Medium-low | observed/mock; how often RunPod does this is unproved/live | no | Task `c34bad53`; strict xfail test. |
| F3 | **Reconcile runs once per session.** A same-dir peer that dies after we reconciled is not reaped while we run, and our exit only prints "not this session's". Needs two decode processes in one output dir (operator error; the code does not forbid it). | Low | observed/mock | no (rule: one decode process per output dir) | Task `c34bad53`; strict xfail test. |
| F4 | **Ambiguity is sticky.** One timed-out create keeps the late-create window open after a later success, so a confirmed-clean run leaves a journal for up to 1 h; the next process then needs a pod listing, and a listing outage at that moment makes it refuse to rent although nothing is owed. | Low | observed/mock | no | Task `c34bad53`; strict xfail test. |
| F5 | **"Refuse to rent" stalls an unattended run with no escape hatch.** Each process refuses after ~290 s (4 setup attempts, 30+60+90 s backoff); 7 processes ~37 min, then `capped_run` exits 4. If a pod's terminate fails for a reason the code does not recognise (error text without "not found" while `get_pod` also errors), the stall is permanent, and the only cure is deleting `<output>/runpod_journal/` by hand, which nothing tells you. `generate()` calls `_ensure_pod()` unguarded, so the refusal ends `run_decode` with a traceback (progress is saved only every 5 clips; clips on disk are re-adopted, so no re-spend). Never double-rents, which is the design trade. | Medium | observed/mock + source; permanence unproved/live | **yes** | Runbook added to ADR 0010 now; CLI task `4189b5dc`. |
| F6 | **`13cbaf1` is unproved on a real pod, and a rejected key is fatal after 2 probes.** Research 0029 saw ~50 of 55 pods refuse ssh. RunPod writes `PUBLIC_KEY` into `authorized_keys` at boot, so sshd can plausibly answer "Permission denied (publickey)" briefly on a good pod; two rejections 5-10 s apart raise `PodSshAuthError`, which `with_setup_retry` does not retry, so one unlucky replacement pod kills the process (`capped_run` restarts it, up to 6 times). Worst-case a refused pod is probed ~4 min (15 probes) before `PodSetupError`. | Medium | source; boot race unproved/live | **yes** | Task `fa27c947`. The live proof of the key fix is task `236284e2`. |
| F7 | **`capped_run` sweeps every pod not present at its start**, including another run's. Running a second paid job on the account during the 30 Nov run (e.g. the Better Call Saul control, any research 0029 arm) can have its pod killed by the first run's sweep. Pre-existing; follow-up 3. | Medium | source (`tools/capped_run.py` `sweep`) | **yes, as a rule** | Existing task `29a4bc50`. **Fixed (job `0443bd59`)**: the sweep is scoped to the run's ownership journal. Pods in no journal are never touched; `--sweep-account` is the explicit opt-in to the old rule. |
| F8 | **Durability on power loss is best effort.** `_write_atomic` uses `os.fsync`, which on macOS does not flush the drive cache (`F_FULLFSYNC`), and does not fsync the directory. SIGKILL and crashes are unaffected (the kernel keeps the page cache); a power cut may lose the last claim. | Low | source | no | Task `c34bad53`. |
| F9 | **Edges:** an unreadable journal does not block renting (inconsistent with fail-closed on a listing failure); an `OSError` from `scan_peers` escapes `create_pod` as a non-`PodSetupError`; `scan_peers` can leave an empty `.lock` file when it races a closing peer; `_pod_status` treats only `EXITED` as gone, so a `TERMINATED` status (if RunPod reports one) reads as running and the exit stays unconfirmed until the listing drops it. | Low | source | no | Task `c34bad53`. |

Looked for and not found: a path where `create_pod` renders a second pod while a claim is
unconfirmed; journal growth with replacement count; a crash point in the replacement
sequence that leaves an unjournaled pod; a fatal `get_pod` error shape (`get_pod` returning
None and raising "not found" both converge).

**Not proved live (carry into task `d69e5f75`, follow-up 7):** what `get_pod` and
`podTerminate` return for a pod that was terminated minutes ago; whether a new pod is ever
invisible to the API; how long the listing lags; how fast sshd accepts the registered key
after boot (count rejected probes in the first real pod's log).

### Operator runbook if the run stalls on "not renting a replacement"

1. Open https://www.runpod.io/console/pods and terminate any `lossy-comfyui-*` pod that is
   not part of a run you know is live.
2. Confirm the console is empty of them. Then delete `<output_dir>/runpod_journal/`
   (both `.json` and `.lock` files) and resume the run command. Skipping step 2 makes the
   next run adopt nothing and can leave a billing pod with no record.
