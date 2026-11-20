---
id: adr-0010-pod-ownership-journal
type: decision
purpose: "Record who answers for a RunPod pod, how that survives a dead process, and when a later run may terminate what an earlier one left."
scope: ["design", "runpod", "decode", "cost-safety"]
non_goals: ["cap enforcement", "bounding replacement overlap", "other output dirs' journals"]
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
5. If the account cannot be listed, or there is no API key, nothing is deleted
   and creation proceeds with a warning; the journal is retried by the next run.
   (Whether to *refuse* to allocate in that state is a separate decision, the
   "bound replacement overlap" follow-up in research 0026.)

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
