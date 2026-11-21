---
id: adr-0010-pod-ownership-journal
type: decision
purpose: "Record who answers for a RunPod pod, how that survives a dead process, and when a later run may terminate what an earlier one left."
scope: ["design", "runpod", "decode", "cost-safety"]
non_goals: ["cap policy (see the capped_run addendum for how the supervisor uses the journal)", "other output dirs' journals"]
tags: ["adr", "runpod", "leak", "journal"]
related: ["research/0026-seinfeld-s01e03-first-end-to-end-run/research.md", "design/adr/008-runpod-self-hosted-wan-strategy.md"]
---

# Context

Research 0026 ("Sceptical review, 15 Nov 2026: a pod can still leak") found that
`RunPodSession` kept the ids of pods it owned in memory only. `runpod_pod.json`
is connection state for one pod, deleted on `terminate`/`forget_pod` even when
the terminate failed. Each session had a fresh random name and never adopted an
older run's pods. So SIGKILL, power loss, a create request that lands
server-side after the client gave up, and a `--keep-pod` run that skipped its
exit sweep could each leave a pod billing with nothing that would ever end it.

# Decision

**Ownership is a per-session journal on disk, written before the action it
protects, and reconciled against the account before the next allocation.**

## Data shape

One file per session: `<output_dir>/runpod_journal/<run_id>.json` (code:
`pod_journal.py`). `run_id` is the 8-hex suffix of the session's pod name, so the
journal and the pods it speaks for share an identity.

| Element | Meaning | Written |
|---|---|---|
| claim `owned` | a pod this session answers for (allocated or adopted) | right after the id is returned |
| claim `abandoned` | handle dropped / terminate requested, not confirmed | **before** the terminate call |
| claim `kept` + `kept_until` | left running on purpose, on-pod reaper armed | after the reaper is armed |
| window `{name, until}` | a pod may exist under `name` whose id we never got | **before** each create request |

A claim is removed only on evidence the pod is gone: a successful terminate, a
"not found", or `get_pod` saying it is gone / `EXITED`. Absence from the pod
*listing* is not evidence (it lags); the pod is asked for by id. A window is
closed when the create was acknowledged and no earlier request in the session
failed ambiguously; otherwise it stays until `until` (`LATE_CREATE_GRACE_S`,
1 h). RunPod's own `QueryError` and out-of-funds are definite rejections (nothing
created); anything else (timeout, dropped connection) is ambiguous.

Liveness is a kernel `flock` on `<run_id>.lock`, held for the session's life.
The kernel releases it on any process death, so "owner is dead" needs no pid or
clock reasoning. Each session writes only its own journal (no shared mutable
file); a peer is only read, or discarded after its claims are adopted. A
session that never asks for a pod writes nothing; a session that ends with
nothing owed deletes its files.

## Reconciliation policy

`RunPodSession.reconcile_journals()` runs once per session, before the first
`create_pod` request and when `reconnect` finds a state file. For every peer
journal in the same output dir:

1. **Live peer** (lock held): never touched. Its claimed ids are protected from
   dead journals' claims (video and audio sessions share an output dir).
2. **Unreadable**: left alone, warned about.
3. **Dead peer**: terminate every running pod it claims by id, plus every
   running pod whose name equals a name it spoke for (finds unknown-id and late
   creates). Exceptions, never terminated: ids this session owns or has in
   `runpod_pod.json` (the pod it is about to adopt), and a `kept` claim still
   inside `kept_until` + 10 min. A `kept` claim past that is terminated: the
   reaper evidently failed.
4. Anything not settled (terminate failed, `get_pod` errored) and every
   unexpired window is **adopted** into this session's journal (state
   `abandoned`, or `kept`), then the dead file is deleted. The adopting session's
   exit sweep retries it. A crash between adopt and delete is harmless: both
   files claim the same ids and a repeat terminate answers "not found".
5. If the account cannot be listed, nothing is deleted, the journal is kept,
   and `create_pod` **refuses to allocate** (`UnsafeToAllocateError`, see the
   addendum). With no API key `create_pod` never gets this far.

Pods with the lossy prefix that appear in **no** journal are never touched by
reconcile or the default sweep (a concurrent run's render looks identical).
`LOSSY_RUNPOD_SWEEP=1` still opts in.

The exit sweep now also runs after a `--keep-pod` exit, sparing only the kept
pod, so an abandoned sibling does not outlive the run. It terminates through the
same journaled, retried path.

# Consequences

- SIGKILL, power loss, a create killed mid-request, a late server-side create
  and a failed terminate are each found by the next run in the same output dir,
  before it rents anything. Mock evidence: `tests/test_runpod_pod.py::TestPodOwnershipJournal`,
  every case asserting which pods are still live in the fake account.
- **Limit:** a journal is found only by a later run using the same output dir.
  A crash in `output/a` followed by work only in `output/b` leaves `a`'s pod to
  its on-pod reaper (kept pods), `capped_run`'s account sweep, or a human. Same
  for a pod whose create window expired (an hour) before anyone looked.
- Ambiguous create failures (most "no capacity" answers are definite, but a
  timeout is not) leave a small journal file for up to an hour.
- Not proved live: real RunPod listing visibility after a create, and real
  `flock` behaviour on a network filesystem. Mocks and local files only.
- Windows is unsupported by the journal (`fcntl`).

# Addendum: bounded overlap and a confirmed exit (research 0026 follow-up 2)

**Decision: stop allocating after a failed terminate.** A session never rents a
pod while any pod it abandoned is not confirmed gone, and never rents beside an
earlier run's leftovers it could not check. `create_pod` runs a gate
(`_settle_abandoned`) before the first request: for each `abandoned` claim it asks
the account by id (a terminate that errored may have landed), terminates what is
still up, and raises `UnsafeToAllocateError` (a retryable `PodSetupError`) if any
claim stays unconfirmed.

Why: the alternative bounds nothing. With a terminate API that keeps failing, a
replacement-anyway policy stacks one more billing pod per replacement for as long
as the outage lasts, on a run nobody is watching. Refusing costs a stalled run,
which resumes from disk (`with_setup_retry` re-attempts the terminate on each
retry; the supervisor or a resume re-attempts after that). If terminate is down,
create is very likely down too, so the stall rarely costs real work. The overlap
bound is therefore zero extra pods, not "some".

Same rule, same error: the pod listing failing while a dead journal is waiting
(we cannot tell if its pod is still up), and the write-ahead window failing to
write (the promise "claim before create" cannot be kept).

**Confirmed exit.** `_cleanup` no longer marks itself done after one pass.
`_exit_sweep` repeats list/terminate/list up to `EXIT_SWEEP_ATTEMPTS` (5) passes,
`EXIT_SWEEP_DELAY_S` apart, inside a `CLEANUP_BUDGET_S` (90 s) wall-clock budget.
A pass is *settled* only if the listing worked, none of this session's pods
appeared in it, and no non-kept claim remains; the pass after a terminate is the
confirmation, which also covers a listing that still shows a terminated pod.
The result is `session.cleanup_confirmed` and a printed line; "NOT confirmed"
means the claim stays in the journal for the next run to settle (the process exit
code is unchanged: an atexit hook cannot change it, and `capped_run`'s own account
sweep is the exit-code authority). An open late-create window is reported by the
journal, not counted as unconfirmed.

**Interrupted cleanup.** Every claim is journaled `abandoned` before its terminate
call, so a SIGKILL anywhere in cleanup leaves a claim the next run settles before
renting. `capped_run.CHILD_GRACE_S` is now 120 s (was 20 s, shorter than three
terminate attempts plus backoff) so the child normally finishes first; a repeat
signal during cleanup is ignored rather than aborting it.

**State-write failure.** A journal write that raises `OSError` (full disk,
read-only) is reported and never stops a terminate or the sweep: the in-memory
snapshot and `_owned_pod_ids` still answer for the pod. A failed `remove_state`
is a warning. A `--keep-pod` exit that cannot write `runpod_pod.json` terminates
the pod (nothing could reconnect to it). A claim that could not be journaled
leaves the write-ahead window open as the durable trace. Only the write-ahead
before a create fails closed.

**Not proved live:** how long RunPod's listing really lags after a terminate;
the delay/attempt constants are guesses sized by the mock outage. That is the
paid observation task (follow-up 7).

# Addendum: operator runbook for a stalled run (review of 21 Nov 2026)

If a run keeps ending with "not renting a replacement" (a claim the API will not
let us confirm gone), `create_pod` is refusing by design and will keep refusing.
Terminate any unaccounted `lossy-comfyui-*` pod in the RunPod console, confirm the
console is clear of them, then delete `<output_dir>/runpod_journal/` and resume.
Findings, measurements and the reasoning: research 0026, "Review, 21 Nov 2026".
`capped_run.CHILD_GRACE_S` was raised from 120 s to 240 s: cleanup in a total API outage measured
~200 s (terminate retries precede the sweep budget).

# Addendum: capped_run sweeps by ownership and fails closed (research 0026 follow-up 3, 21 Nov 2026)

**Decision: the supervisor terminates only pods the journals claim.** `tools/capped_run.py`
used to end a run by terminating every account pod that was not there at start, which
also killed a concurrent run's pod. It now takes `--output-dir` and treats a pod as
the run's iff a journal under `<output-dir>/runpod_journal/` claims its id, or a name
it spoke for (windows cover creates whose id was lost). `pod_journal.read_ownership`
reads every journal in the dir, live or dead (a SIGKILLed child's leftover is exactly
what the sweep is for), taking no lock and writing nothing: each session still writes
only its own file. A journal that cannot be parsed makes ownership unknown, so the
sweep terminates nothing on a guess and reports itself unconfirmed (exit 5). The old
rule survives as `--sweep-account`; passing neither flag is a usage error, so the
account-exclusivity assumption is always a visible choice.

Residual: a pod in no journal (the child died before journaling, with no window) is
not swept. The write-ahead window before every create makes that case narrow, but it
is the price of not shooting other runs' pods. `--attach-pid` no longer assumes the
attached run owns every existing pod unless `--sweep-account` is given.

**Fail closed.** The supervisor tracks when it last read the balance (`SpendGuard`).
While reads fail, spend is the last reading plus `--burn-per-hr` (default $3/h, two
overlapping SECURE pods) for the time since, and a reading older than `--max-stale`
(default 600 s) stops the run regardless (exit 6). The stop line is the cap minus a
reserve: at least `--margin`, and more if the assumed burn over one poll interval +
the child's grace + the sweep is larger (`reserve_usd`). A watcher exception in the
progress report is contained; an exception that ends the watcher stops the child
(exit 6). Launch and stop share a lock (`Run`), closing the window in which a retry
could start a child after the watcher's stop.

Not bounded: balance lag itself (RunPod crediting spend late), other providers' spend
(fal), and whether the account is shared at all. Those are Will's call; this change
only stops the supervisor from depending on the answer for the sweep.
