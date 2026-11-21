"""Shared RunPod pod lifecycle management for ComfyUI strategies."""

import atexit
import json
import os
import signal
import subprocess
import sys
import re
import time
import uuid
from typing import NamedTuple

import pod_journal
import pod_ssh
from pod_ssh import PodSshAuthError, SshKeyError, SshOutcome  # noqa: F401  (re-exported for callers)


# Default GPU preferences (cheapest first, with VRAM headroom preference)
GPU_TYPES = [
    ("NVIDIA GeForce RTX 4090", 0.34),   # 24 GB, fastest consumer GPU
    ("NVIDIA L40S", 0.54),               # 48 GB, Ada Lovelace
    ("NVIDIA RTX A6000", 0.33),          # 48 GB, Ampere (slower inference)
    ("NVIDIA RTX 4000 Ada Generation", 0.34),
]

# Community cloud is cheapest but its stock genuinely runs out -- a rehearsal
# slice found no capacity across all five 48GB types for ~15 minutes. Secure
# cloud is RunPod's own datacentres: pricier, far more reliably available. Try
# community first and fall back rather than failing the run.
CLOUD_TYPES = [t for t in os.environ.get("LOSSY_RUNPOD_CLOUD", "COMMUNITY,SECURE").split(",") if t]
CLOUD_TYPE = CLOUD_TYPES[0]  # back-compat for anything reading the old name

# The (gpu, rate) tables above are COMMUNITY prices. SECURE costs more for the
# same GPU (A6000: $0.59 vs $0.33 -- a pilot logged ~$1.22 against a real
# $2.29). Known SECURE prices go here; any other GPU falls back to the
# community rate times SECURE_MARKUP. A costPerHr on the created pod, if the
# API returns one, beats both.
SECURE_RATES = {"NVIDIA RTX A6000": 0.59}
SECURE_MARKUP = 1.8


def rate_for(cloud_type: str, gpu_type: str, community_rate: float, pod: dict | None = None) -> float:
    """Hourly rate actually billed for a pod on `cloud_type`."""
    reported = (pod or {}).get("costPerHr")
    if isinstance(reported, (int, float)) and reported > 0:
        return float(reported)
    if cloud_type == "COMMUNITY":
        return community_rate
    return SECURE_RATES.get(gpu_type, round(community_rate * SECURE_MARKUP, 2))
DOCKER_IMAGE = "runpod/comfyui:latest"

# Every pod this project creates is named POD_NAME_PREFIX + "-" + a unique
# suffix. The prefix makes lossy's pods findable; the suffix makes *this
# session's* pod distinguishable from a concurrent run's, which is what lets
# the stray sweep below terminate its own leftovers without shooting down
# somebody else's render. Pods created before unique naming are all called
# exactly POD_NAME_PREFIX -- matched, but never claimed as ours.
POD_NAME_PREFIX = "lossy-comfyui"
CONTAINER_DISK_GB = 50
COMFYUI_PORT = 8188
POD_READY_TIMEOUT = 600
COMFYUI_READY_TIMEOUT = 600
COMFYUI_DIR = "/workspace/runpod-slim/ComfyUI"

# How long a kept-alive pod survives with nobody using it. A pod bills until
# it is explicitly terminated, and once the local process exits there is
# nothing of ours left running to notice it went idle -- a dress-rehearsal
# render finished in 2.5hrs and then billed for another 10.5hrs ($3.40, more
# than the render) purely because the next stage was never started. So the
# deadline is armed *on the pod itself*, where it survives the process
# exiting, the session ending, and the machine sleeping.
KEEP_POD_DEFAULT_MINUTES = 30
REAPER_PID_FILE = "/workspace/.lossy_reaper.pid"

# Pod provisioning fails often enough that a long run cannot treat it as
# fatal. Across ten attempts in one rehearsal session, three pods died on
# CUDA init before ComfyUI started and one found no capacity on any GPU type
# -- each stalled the run until a human reissued the command.
POD_SETUP_ATTEMPTS = 4
POD_SETUP_BACKOFF_S = 30
# A pod reports its ports before sshd accepts connections. Probe until it does:
# refused/timed-out is "not up yet" (wait, bounded), but sshd answering with
# "Permission denied" is the server rejecting our key -- retrying the same key
# cannot fix that, so it is confirmed once and then fatal.
SSH_READY_ATTEMPTS = 15
SSH_PROBE_TIMEOUT_S = 30
SSH_AUTH_REJECTIONS_FATAL = 2
# After /interrupt, how many 2 s polls of /queue before declaring the pod wedged.
CANCEL_CONFIRM_POLLS = 5
# A terminate call that errors is retried: it is the only thing standing between
# a replaced (or wedged) pod and billing until the run ends. "Not found" is the
# expected answer for a pod that is already gone and is not retried.
TERMINATE_ATTEMPTS = 3
TERMINATE_BACKOFF_S = 3
# Ownership journal (pod_journal.py, ADR 0010). A create request that failed
# ambiguously may still land server-side; how long a later run keeps looking
# for a pod under that name before concluding it never will.
LATE_CREATE_GRACE_S = 3600
# Slack past a kept pod's armed deadline before a later run concludes the
# on-pod reaper failed and terminates it.
KEPT_DEADLINE_SLACK_S = 600
JOURNAL_DIRNAME = pod_journal.JOURNAL_DIRNAME
# Exit sweep (ADR 0010 addendum): list, terminate, and list again until a pass
# finds nothing of ours and no claim is outstanding. Passes are bounded by count
# AND by a wall-clock budget, so a supervisor's grace period can be set above it.
# The budget is checked between passes; one HTTP call can still outlast it (the
# SDK's timeout is per request), which is why every claim is journaled first.
EXIT_SWEEP_ATTEMPTS = 5
EXIT_SWEEP_DELAY_S = 5
CLEANUP_BUDGET_S = 90



# RunPod reports "balance too low" through the same generic exception path as
# "no instances available", so a run that simply ran out of money reported
# "no capacity on any of 5 GPU types across 2 cloud(s)" and then dutifully
# retried four times against something no retry can fix. That happened 16
# hours into a 690-shot render: the pod was killed when the balance hit zero,
# and the message pointed at the wrong problem entirely.
_OUT_OF_FUNDS_MARKERS = ("balance is too low", "add funds", "insufficient funds")


def _is_not_found(exc: Exception) -> bool:
    """Whether a terminate error means the pod is already gone."""
    return "not found" in str(exc).lower()


def _is_definite_rejection(exc: Exception) -> bool:
    """Whether RunPod itself answered 'no' to a create: nothing was made.

    The SDK raises QueryError only for an error the server returned. Anything
    else (timeout, dropped connection, bad gateway) leaves the outcome unknown.
    """
    return type(exc).__name__ == "QueryError" or _is_out_of_funds(exc)


def _is_out_of_funds(exc: Exception) -> bool:
    """Whether a provisioning failure is about money rather than capacity."""
    text = str(exc).lower()
    return any(marker in text for marker in _OUT_OF_FUNDS_MARKERS)


class OutOfFundsError(RuntimeError):
    """The account cannot afford a pod. NOT retryable -- a human must top up."""


class PodSetupError(RuntimeError):
    """A pod could not be provisioned or brought up. Retryable on a new pod."""


class UnsafeToAllocateError(PodSetupError):
    """Renting another pod now could leave two billing at once, so none is rented.

    Raised when a pod this session abandoned is not confirmed gone, or when an
    earlier run's leftovers cannot be checked, or when the journal that protects
    a create cannot be written. Retryable (a PodSetupError): each retry
    re-attempts the terminate first, so it converges once the API recovers.
    """


class GatedModelError(RuntimeError):
    """A gated Hugging Face file cannot be fetched: no token, or the gate is not accepted.

    Deliberately not a PodSetupError: a new pod would fail the same way, so
    `with_setup_retry` must not re-provision (and bill) for it.
    """


class PinnedModel(NamedTuple):
    """A weights file fetched from a fixed URL and checked against a sha256.

    For files that come from a mirror rather than the vendor's own repo, the
    hash is the only thing tying the bytes to what was evaluated.

    `gate_url` is set for a file behind a Hugging Face access click: the page
    where it is accepted. Such a file is fetched with the HF_TOKEN bearer token.
    """

    dest: str      # path under ComfyUI's models/ dir, e.g. "vae/foo.safetensors"
    url: str
    sha256: str
    gate_url: str | None = None


def gated_help(model: PinnedModel) -> str:
    """The one message every gate failure ends with: where to click, and what to set."""
    return (f"Accept the gate at {model.gate_url} (signed in to Hugging Face, click "
            f"'Agree and access repository'), then set HF_TOKEN=hf_... in .env "
            f"(a Read token from https://huggingface.co/settings/tokens made by the same account).")


def check_gated_access(models: list[PinnedModel], token: str | None, head=None) -> None:
    """Fail before any pod is paid for if a gated model cannot be fetched.

    One HEAD request per gated file, from this machine. Hugging Face answers
    401 with no (or a bad) token and 403 with a valid token whose account has
    not accepted the gate; a 2xx/3xx means the pod's download will be allowed.
    Raises GatedModelError with the fix; a network error raises it too, since
    "could not check" must not read as "fine" before a billed boot.
    `head` is httpx.head by default; tests pass a stand-in.
    """
    import httpx
    head = head or httpx.head
    for m in (m for m in models if m.gate_url):
        name = os.path.basename(m.dest)
        if not token:
            raise GatedModelError(f"{name} is gated and HF_TOKEN is not set. {gated_help(m)}")
        try:
            status = head(m.url, headers={"Authorization": f"Bearer {token}"},
                          follow_redirects=False, timeout=30).status_code
        except httpx.HTTPError as e:
            raise GatedModelError(
                f"{name}: could not reach Hugging Face to check access ({type(e).__name__}); "
                f"no pod was started. Try again.") from e
        if status == 401:
            raise GatedModelError(f"{name}: Hugging Face rejected HF_TOKEN (401). {gated_help(m)}")
        if status == 403:
            raise GatedModelError(f"{name}: the gate is not accepted for the account behind "
                                  f"HF_TOKEN (403). {gated_help(m)}")
        if status >= 400:
            raise GatedModelError(f"{name}: Hugging Face answered {status} for {m.url}")


def parse_comfyui_version(text: str) -> tuple[int, ...] | None:
    """Pull the version out of ComfyUI's comfyui_version.py, e.g. (0, 30, 0)."""
    m = re.search(r'__version__\s*=\s*["\']v?(\d+(?:\.\d+)*)', text)
    return tuple(int(p) for p in m.group(1).split(".")) if m else None


class RunPodSession:
    """Manages a RunPod ComfyUI pod with state file persistence.

    Can create a new pod or reconnect to an existing one via a state file.
    Handles cleanup on exit (terminate or keep-alive depending on flags).
    """

    def __init__(self, output_dir: str, keep_pod: bool | int = False,
                 gpu_types: list[tuple[str, float]] | None = None,
                 disk_gb: int | None = None):
        self.output_dir = output_dir
        # Container disk for pods this session creates. None = the module
        # default, read at create time so a caller overriding
        # CONTAINER_DISK_GB still takes effect.
        self.disk_gb = disk_gb
        # keep_pod is a duration, not a switch: True means the default TTL,
        # an int means that many minutes. There is deliberately no way to say
        # "keep forever" -- that contract is what produced a 10.5hr idle bill.
        self.keep_pod = bool(keep_pod)
        self.keep_pod_minutes = (
            KEEP_POD_DEFAULT_MINUTES if keep_pod is True
            else int(keep_pod) if keep_pod else 0
        )
        # Cheapest-first GPU fallback list to try when creating a pod. Defaults
        # to the module-level GPU_TYPES; strategies with stricter VRAM needs
        # (e.g. a model that OOMs on a 24GB card) can pass a narrower list.
        self.gpu_types = gpu_types or GPU_TYPES
        # Unique per session, decided up front so it is stable across a
        # create/terminate cycle and recorded in the state file for reconnects.
        self.run_id = uuid.uuid4().hex[:8]
        self.pod_name = f"{POD_NAME_PREFIX}-{self.run_id}"
        self.pod_id: str | None = None
        self.base_url: str | None = None
        self.ssh_host: str | None = None
        self.ssh_port: int | None = None
        self._ssh_identity: pod_ssh.SshIdentity | None = None
        self.gpu_hourly_rate: float = self.gpu_types[0][1]
        self.pod_start_time: float | None = None
        self._clean_exit = False
        self._cleanup_registered = False
        self._swept = False
        # Set just before the first runpod.create_pod call. A create can time
        # out client-side after the pod exists server-side, leaving no recorded
        # id; the exit sweep (which identifies pods by name) must still run.
        self._create_attempted = False
        self._http = None
        # Set when a job timed out and /interrupt could not free the worker:
        # the pod is alive and billing but will never run another job, so a
        # caller should replace it rather than queue more work behind the dead
        # one. Cleared whenever the pod handle is dropped.
        self.wedged = False
        # Pod ids this session provisioned or adopted. Belt-and-braces
        # alongside the name check: either one identifying a pod as ours is
        # enough to terminate it. Append-only: the exit sweep is gated on it
        # being non-empty, so an id is never removed when a pod is terminated
        # or forgotten -- that would let an abandoned pod whose terminate
        # failed escape the one check that exists to catch it.
        self._owned_pod_ids: set[str] = set()
        # Durable twin of the above (pod_journal.py): created lazily, so a
        # session that never asks for a pod leaves nothing on disk.
        self._journal = pod_journal.PodJournal(
            os.path.join(output_dir, JOURNAL_DIRNAME), self.run_id, self.pod_name)
        self._reconciled = False
        # A create request has failed in a way that does not prove no pod was
        # made (timeout, dropped connection). Keeps its late-create window open.
        self._create_ambiguous = False
        # True while _cleanup runs, so a signal landing mid-cleanup neither
        # restarts nor aborts it. None until the exit sweep has run; then
        # whether the account confirmed nothing of ours is left (see _cleanup).
        self._cleaning = False
        self.cleanup_confirmed: bool | None = None

    def _register_cleanup(self):
        """Register atexit and signal handlers to terminate pod on exit."""
        if self._cleanup_registered:
            return
        self._cleanup_registered = True

        atexit.register(self._cleanup)

        original_sigint = signal.getsignal(signal.SIGINT)
        original_sigterm = signal.getsignal(signal.SIGTERM)

        def _handler(signum, frame):
            if self._cleaning:
                # A repeat signal mid-cleanup must not abort or restart it: the
                # pod is the thing being protected. A supervisor's SIGKILL still
                # ends us, and the journal's abandoned claim survives that.
                print("  Signal received during pod cleanup; still cleaning up.")
                return
            self._cleanup()
            if signum == signal.SIGINT and callable(original_sigint):
                original_sigint(signum, frame)
            elif signum == signal.SIGTERM and callable(original_sigterm):
                original_sigterm(signum, frame)
            else:
                sys.exit(1)

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)

    def _cleanup(self):
        """Terminate or preserve pod depending on keep_pod flag and exit status.

        Every step is independent of the others failing: a state file that
        cannot be written or removed, or a journal on a full disk, must never
        be the reason the sweep did not run. Sets `cleanup_confirmed`.
        """
        if self._cleaning:
            return
        self._cleaning = True
        try:
            self._cleanup_steps()
        finally:
            self._cleaning = False

    def _cleanup_steps(self):
        keeping = bool(self.keep_pod and self._clean_exit and self.pod_id is not None)

        if self.pod_id is not None:
            if keeping:
                # Normal exit with --keep-pod: write state file, don't terminate --
                # but arm a self-destruct on the pod first so "kept alive" cannot
                # silently mean "billing all night".
                try:
                    self.write_state()
                    armed = self.arm_self_destruct(self.keep_pod_minutes)
                except OSError as e:
                    # No state file means nothing can reconnect to this pod.
                    print(f"  Pod {self.pod_id}: could not write {self.state_path} ({e}).")
                    armed = False
                if armed:
                    # Durable: if a later run finds the reaper never fired,
                    # this is what lets it terminate the pod.
                    self._journal_write(self._journal.set_pod, self.pod_id, self.pod_name,
                                        pod_journal.KEPT, time.time() + self.keep_pod_minutes * 60)
                    print(f"  Pod {self.pod_id} kept alive (self-destructs in "
                          f"{self.keep_pod_minutes}min); state at {self.state_path}")
                else:
                    # A kept pod with no deadline bills until a human notices
                    # (the 10.5 h / $3.40 failure). Fail closed: terminate it.
                    print(f"  Pod {self.pod_id}: self-destruct could NOT be armed; "
                          f"terminating instead of keeping it.")
                    keeping = False
                    self._terminate_guarded()
            else:
                self._terminate_guarded()

        self.close_http()

        # A deliberately kept-alive pod is not a stray, but its abandoned
        # siblings are: sweep anyway, sparing only the pod we kept.
        if self._swept:
            return
        self._swept = True
        try:
            self.cleanup_confirmed = self._exit_sweep(
                spare={self.pod_id} if keeping and self.pod_id else frozenset())
            self._report_cleanup()
        finally:
            try:
                self._journal.close()
            except OSError as e:
                print(f"  Pod journal could not be finalised ({e}); its last written "
                      f"state stands and the next run in this output dir will read it.")

    def _terminate_guarded(self) -> None:
        try:
            self.terminate()
        except Exception as e:   # the sweep below is the backstop; it must still run
            print(f"  Warning: terminate raised {type(e).__name__}: {e}")

    def _report_cleanup(self) -> None:
        if self.cleanup_confirmed is None or not (self._owned_pod_ids or self._create_attempted):
            return
        if self.cleanup_confirmed:
            print("  Exit sweep: every pod of this session confirmed gone.")
        else:
            print("  WARNING: exit cleanup NOT confirmed -- a pod of this session may still "
                  "be billing. The ownership journal keeps the claim; the next run in this "
                  "output dir terminates it before renting. Check "
                  "https://www.runpod.io/console/pods.")

    def _journal_write(self, fn, *args) -> bool:
        """Run a journal mutation; a failed WRITE is reported, never fatal.

        The in-memory snapshot is updated before the write is attempted, so
        ownership still holds for this process; only crash durability is lost.
        """
        try:
            fn(*args)
            return True
        except OSError as e:
            print(f"  Warning: pod journal not written ({e}); this process still "
                  f"answers for its pods, but a crash would not be recoverable.")
            return False

    @property
    def state_path(self) -> str:
        return os.path.join(self.output_dir, "runpod_pod.json")

    def mark_clean_exit(self):
        """Call before normal return to signal that pod should be kept."""
        self._clean_exit = True

    # -- Ownership journal (ADR 0010) --

    def _own(self, pod_id: str) -> bool:
        """Record, durably, that this session answers for `pod_id`. False if only in memory."""
        self._owned_pod_ids.add(pod_id)
        return self._journal_write(self._journal.set_pod, pod_id, self.pod_name, pod_journal.OWNED)

    def _claimed_names(self) -> set[str]:
        """Pod names this session answers for: its own plus any it adopted."""
        return {self.pod_name} | self._journal.snapshot.names

    def _spared_ids(self) -> set[str]:
        """Pods deliberately kept alive whose own reaper is still entitled to them."""
        now = time.time()
        return {pid for pid, c in self._journal.snapshot.claims.items()
                if c.state == pod_journal.KEPT and c.kept_until is not None
                and now < c.kept_until + KEPT_DEADLINE_SLACK_S}

    def _settle_create_window(self) -> None:
        """Close the late-create window if no failed request is still unexplained."""
        if not self._create_ambiguous:
            self._journal_write(self._journal.close_window, self.pod_name)

    def _release(self, pod_id: str) -> Exception | None:
        """Terminate `pod_id`, keeping the journal truthful at every instant.

        'abandoned' is written first, so a crash during the call still leaves a
        claim; the claim is dropped only once the pod is confirmed gone
        (terminated, or the API says it never/no longer existed).
        """
        self._journal_write(
            self._journal.set_pod,
            pod_id,
            (self._journal.snapshot.claims.get(pod_id) or pod_journal.Claim(pod_id, self.pod_name)).name,
            pod_journal.ABANDONED)
        err = self._terminate_pod_id(pod_id)
        if err is None or _is_not_found(err):
            self._journal_write(self._journal.drop_pod, pod_id)
        return err

    def _settle_abandoned(self, runpod) -> list[str]:
        """Confirm every pod this session abandoned is gone; return those not.

        The allocation gate. An 'abandoned' claim means a terminate was asked
        for and never confirmed. Ask the account about each by id (a failed
        terminate call may still have landed); terminate the ones still up.
        """
        unsettled = []
        for pid, claim in list(self._journal.snapshot.claims.items()):
            if claim.state != pod_journal.ABANDONED:
                continue
            if self._pod_status(runpod, pid) == "gone":
                self._journal_write(self._journal.drop_pod, pid)
                continue
            err = self._release(pid)
            if err is not None and not _is_not_found(err):
                unsettled.append(pid)
        return unsettled

    def _pod_status(self, runpod, pod_id: str) -> str:
        """Ask the account about one pod by id: 'gone', 'running' or 'unknown'.

        The listing can lag behind a pod that exists, so absence from it is
        not proof; an error is not proof either.
        """
        try:
            status = runpod.get_pod(pod_id)
        except Exception:
            return "unknown"
        if not status or status.get("desiredStatus") == "EXITED":
            return "gone"
        return "running"

    def reconcile_journals(self, spare: set[str] | frozenset[str] = frozenset()) -> list[str]:
        """Settle what dead sessions left behind, before this one allocates.

        For every journal in this output dir whose owner no longer holds its
        lock: terminate each pod it still claims (and any pod carrying a name it
        spoke for, which catches creates whose id was never received), unless a
        live owner has it. A claim that cannot be settled this time -- terminate
        failed, listing unavailable -- is adopted into this session's journal so
        the exit sweep retries it and nothing is forgotten. Idempotent: running
        it twice, or after a crash halfway, converges to the same account.
        Returns the ids it terminated.
        """
        if self._reconciled:
            return []
        peers = pod_journal.scan_peers(self._journal.directory, self.run_id)
        dead = [p for p in peers if not p.live]
        for p in peers:
            if p.live or p.snapshot is None:
                p.release()
        try:
            return self._reconcile_dead(peers, dead, spare)
        finally:
            for p in dead:
                p.release()

    def _reconcile_dead(self, peers, dead, spare) -> list[str]:
        unreadable = [p for p in dead if p.snapshot is None]
        for p in unreadable:
            print(f"  Journal {p.json_path} is unreadable; leaving it. Check "
                  f"https://www.runpod.io/console/pods for stray pods.")
        dead = [p for p in dead if p.snapshot is not None]
        if not dead:
            self._reconciled = True
            return []
        if not os.environ.get("RUNPOD_API_KEY"):
            print(f"  {len(dead)} earlier run(s) left a pod journal but RUNPOD_API_KEY is "
                  f"not set; cannot check for leftover pods.")
            return []

        import runpod
        runpod.api_key = os.environ["RUNPOD_API_KEY"]
        try:
            listing = runpod.get_pods() or []
        except Exception as e:
            print(f"  Could not list pods to reconcile an earlier run's journal ({e}); "
                  f"it is kept and retried next time. Verify at https://www.runpod.io/console/pods")
            raise UnsafeToAllocateError(
                "cannot tell whether an earlier run's pod is still billing "
                "(pod listing unavailable)") from e
        running = {p["id"]: p for p in listing
                   if p.get("id") and p.get("desiredStatus", "RUNNING") != "EXITED"}

        protected = set(spare) | self._owned_pod_ids
        if self.pod_id:
            protected.add(self.pod_id)
        for p in peers:
            if p.live and p.snapshot is not None:
                protected.update(p.snapshot.claims)

        now = time.time()
        terminated: list[str] = []
        unsettled: dict[str, pod_journal.Claim] = {}
        windows: list[pod_journal.Window] = []
        for peer in dead:
            snap = peer.snapshot
            # id -> claim, for the pods we must end: claimed by id, or running
            # under a name this journal spoke for (a create whose id was lost).
            targets: dict[str, pod_journal.Claim] = {}
            for pid, claim in snap.claims.items():
                if pid in protected:
                    continue            # a live owner (or we) hold it; the stale claim is dropped
                if claim.state == pod_journal.KEPT and claim.kept_until is not None \
                        and now < claim.kept_until + KEPT_DEADLINE_SLACK_S:
                    unsettled[pid] = claim   # its own reaper is still entitled to it
                    continue
                targets[pid] = claim
            for pid, pod in running.items():
                if pid not in protected and pid not in targets and pod.get("name") in snap.names \
                        and pid not in unsettled:
                    targets[pid] = pod_journal.Claim(pid, pod["name"], pod_journal.ABANDONED)
            for pid, claim in targets.items():
                status = "running" if pid in running else self._pod_status(runpod, pid)
                if status == "running":
                    print(f"  Reconcile: terminating pod {pid} left by an earlier run "
                          f"({claim.name}, {claim.state})")
                    err = self._terminate_pod_id(pid)
                    if err is None or _is_not_found(err):
                        terminated.append(pid)
                    else:
                        print(f"  Reconcile: FAILED to terminate {pid} ({err}); "
                              f"retrying at exit. Check https://www.runpod.io/console/pods")
                        unsettled[pid] = claim
                elif status == "unknown":
                    unsettled[pid] = claim   # cannot tell; keep the claim rather than forget a billing pod
            windows += [w for w in snap.windows if w.until > now]

        if unsettled or windows:
            for pid, claim in unsettled.items():
                if claim.state != pod_journal.KEPT:
                    self._owned_pod_ids.add(pid)
            self._journal.adopt(list(unsettled.values()), windows)
        for peer in dead:
            peer.discard()
        self._reconciled = True
        return terminated

    # -- State file I/O --

    def write_state(self):
        """Write pod connection info to state file (atomic)."""
        state = {
            "pod_id": self.pod_id,
            "pod_name": self.pod_name,
            "base_url": self.base_url,
            "ssh_host": self.ssh_host,
            "ssh_port": self.ssh_port,
            "gpu_hourly_rate": self.gpu_hourly_rate,
            "start_time": self.pod_start_time,
        }
        os.makedirs(self.output_dir, exist_ok=True)
        tmp_path = self.state_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp_path, self.state_path)

    def read_state(self) -> bool:
        """Read pod state file. Returns True if state was loaded."""
        if not os.path.exists(self.state_path):
            return False
        with open(self.state_path) as f:
            state = json.load(f)
        self.pod_id = state["pod_id"]
        # Pods created before unique naming have no name in their state file;
        # fall back to the bare prefix they were actually created with.
        self.pod_name = state.get("pod_name") or POD_NAME_PREFIX
        self.base_url = state["base_url"]
        self.ssh_host = state.get("ssh_host")
        self.ssh_port = state.get("ssh_port")
        self.gpu_hourly_rate = state.get("gpu_hourly_rate", GPU_TYPES[0][1])
        self.pod_start_time = state.get("start_time")
        return True

    def remove_state(self):
        """Remove the pod state file. A file that cannot be removed is reported, not fatal.

        Callers run this right after a pod was dropped; an exception here used
        to propagate past the cleanup that follows it.
        """
        try:
            if os.path.exists(self.state_path):
                os.remove(self.state_path)
        except OSError as e:
            print(f"  Warning: could not remove {self.state_path} ({e}); "
                  f"delete it before resuming or the run will try to reconnect to a dead pod.")

    # -- Pod lifecycle --

    def create_pod(self):
        """Create a new RunPod pod and wait for it to be ready."""
        import runpod

        runpod.api_key = os.environ.get("RUNPOD_API_KEY")
        if not runpod.api_key:
            print("Error: RUNPOD_API_KEY not set in .env or environment.")
            sys.exit(1)

        # A pod we cannot ssh into is worthless and still bills: prove the key
        # works for both halves (registered with RunPod, offered by ssh) first.
        ssh_identity = self.ssh_identity

        # Even a failed request may have created a server-side pod whose id
        # never reached us. Register before the request so exit sweeps by name.
        self._register_cleanup()

        # Earlier runs' leftovers (SIGKILL, power loss, a create that landed
        # late) bill alongside whatever we are about to rent: settle them first.
        self.reconcile_journals()

        if self.pod_id is not None:
            # Overwriting the handle would orphan a pod that is still billing.
            print(f"  Replacing pod {self.pod_id}: terminating it first.")
            self.terminate()

        # Never overlap: renting while a pod we abandoned is not confirmed gone
        # leaves two billing until exit or longer. Fail closed -- the caller's
        # retry (with_setup_retry, the supervisor, a resume) re-attempts the
        # terminate first. The cost is a stalled run, resumable from disk; the
        # alternative is an unbounded second bill nobody is watching.
        unsettled = self._settle_abandoned(runpod)
        if unsettled:
            raise UnsafeToAllocateError(
                f"not renting a replacement: pod(s) {', '.join(unsettled)} could not be "
                f"confirmed terminated. Check https://www.runpod.io/console/pods")

        # Write-ahead: if the process dies mid-request, or the request lands
        # server-side after we gave up, a later run finds this name in the
        # journal and looks for the pod by it. If that promise cannot be kept,
        # do not rent.
        try:
            self._journal.open_window(self.pod_name, time.time() + LATE_CREATE_GRACE_S)
        except OSError as e:
            raise UnsafeToAllocateError(
                f"not renting: the pod journal cannot be written ({e})") from e

        pod = None
        gpu_rate = self.gpu_types[0][1]
        # Cheapest cloud first, then each GPU within it; only escalate to the
        # pricier cloud once the cheap one is genuinely out of everything.
        attempts = [(c, g, r) for c in CLOUD_TYPES for g, r in self.gpu_types]
        for cloud_type, gpu_type, rate in attempts:
            label = gpu_type if cloud_type == CLOUD_TYPES[0] else f"{gpu_type} [{cloud_type}]"
            print(f"  Trying {label}...")
            try:
                self._create_attempted = True
                pod = runpod.create_pod(
                    name=self.pod_name,
                    image_name=DOCKER_IMAGE,
                    gpu_type_id=gpu_type,
                    cloud_type=cloud_type,
                    gpu_count=1,
                    container_disk_in_gb=self.disk_gb or CONTAINER_DISK_GB,
                    ports=f"{COMFYUI_PORT}/http,22/tcp",
                    support_public_ip=True,
                    start_ssh=True,
                    env={"PUBLIC_KEY": ssh_identity.public_key},
                )
                gpu_rate = rate_for(cloud_type, gpu_type, rate, pod)
                print(f"  Got {label} @ ${gpu_rate}/hr")
                break
            except Exception as e:
                if not _is_definite_rejection(e):
                    self._create_ambiguous = True   # the pod may exist; keep the window open
                if _is_out_of_funds(e):
                    # Not a capacity problem, and no amount of retrying or
                    # cycling GPU types fixes it. Say so immediately.
                    self._settle_create_window()
                    raise OutOfFundsError(
                        f"RunPod account balance is too low to rent a pod "
                        f"({label}). Add funds at "
                        f"https://www.runpod.io/console/billing and resume -- "
                        f"completed shots are already on disk."
                    ) from e
                print(f"  {label} unavailable: {e}")
                continue

        if pod is None:
            self._settle_create_window()
            raise PodSetupError(
                f"no capacity on any of {len(self.gpu_types)} GPU types "
                f"across {len(CLOUD_TYPES)} cloud(s)"
            )

        pod_id = str(pod["id"])
        self.pod_id = pod_id
        # If the claim could not be journaled, leave the write-ahead window open:
        # it is then the only durable trace of this pod's name.
        if self._own(pod_id):
            self._settle_create_window()
        self.gpu_hourly_rate = gpu_rate
        self.pod_start_time = time.time()
        print(f"  Pod created: {self.pod_id}")

        # Wait for runtime
        print("  Waiting for pod to start...")
        start = time.time()
        while time.time() - start < POD_READY_TIMEOUT:
            status = runpod.get_pod(self.pod_id)
            if status is None:
                time.sleep(5)
                continue
            runtime = status.get("runtime")
            if runtime is not None and runtime.get("ports"):
                break
            time.sleep(5)
        else:
            print(f"  Error: Pod did not start within {POD_READY_TIMEOUT}s")
            self.terminate()
            # Retryable: a host that never comes up is the same class of
            # problem as one whose CUDA fails -- a fresh pod usually works.
            raise PodSetupError(f"pod did not start within {POD_READY_TIMEOUT}s")

        self.base_url = f"https://{self.pod_id}-{COMFYUI_PORT}.proxy.runpod.net"
        print(f"  Pod ready: {self.base_url}")

        # Get SSH info
        pod_info = runpod.get_pod(self.pod_id)
        for port_info in pod_info.get("runtime", {}).get("ports", []):
            if port_info["privatePort"] == 22 and port_info["isIpPublic"]:
                self.ssh_host = port_info["ip"]
                self.ssh_port = port_info["publicPort"]
                break
        self.wait_for_ssh()

    def reconnect(self) -> bool:
        """Reconnect to an existing pod via state file. Returns True if successful."""
        if not self.read_state():
            return False

        self._register_cleanup()
        print(f"  Reconnecting to pod {self.pod_id}...")

        # Verify pod is still running
        import runpod
        runpod.api_key = os.environ.get("RUNPOD_API_KEY")
        if not runpod.api_key:
            print("Error: RUNPOD_API_KEY not set in .env or environment.")
            sys.exit(1)

        # The state file's pod is the one we are here to adopt: never reap it.
        try:
            self.reconcile_journals(spare={self.pod_id} if self.pod_id else frozenset())
        except UnsafeToAllocateError as e:
            # Adopting a pod rents nothing, so this need not stop us; create_pod
            # retries the reconcile (it stays pending) before it ever allocates.
            print(f"  Reconcile deferred: {e}")

        try:
            status = runpod.get_pod(self.pod_id)
            runtime = status.get("runtime")
            if runtime is None or not runtime.get("ports"):
                print(f"  Pod {self.pod_id} is no longer running.")
                self.remove_state()
                return False
        except Exception as e:
            print(f"  Failed to check pod {self.pod_id}: {e}")
            self.remove_state()
            return False

        print(f"  Reconnected: {self.base_url}")
        if self.pod_id:
            self._own(self.pod_id)
        # This pod was left alive with a deadline armed; clear it so it does
        # not terminate underneath the run that just picked it up.
        self.disarm_self_destruct()
        return True

    def with_setup_retry(self, setup_fn, attempts: int = POD_SETUP_ATTEMPTS) -> None:
        """Run a pod-setup routine, re-provisioning onto a fresh pod on failure.

        Provisioning is unreliable enough that a long run cannot treat it as
        fatal: in one rehearsal session three of ten pods died on CUDA init
        before ComfyUI started and one attempt found no capacity at all, and
        every one of them stalled the run until a human reissued the command.

        Each retry starts from scratch -- the failed pod is terminated here
        (not every PodSetupError site does it: a failed ComfyUI upgrade or a
        sha256 mismatch leaves a healthy-looking, billing pod behind), and the
        stale state file is cleared so the next attempt provisions rather than
        reconnects.
        """
        last: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                setup_fn()
                return
            except PodSshAuthError:
                # Definite: the next pod is given the same key and refuses it
                # the same way. Release this one, then fail instead of renting more.
                self._drop_failed_pod()
                raise
            except PodSetupError as e:
                last = e
                self._drop_failed_pod()
                if attempt < attempts:
                    wait = POD_SETUP_BACKOFF_S * attempt
                    print(f"  Pod setup failed ({e}). "
                          f"Retrying on a fresh pod in {wait}s "
                          f"[attempt {attempt + 1}/{attempts}]...")
                    time.sleep(wait)
        print(f"Error: pod setup failed {attempts} times; giving up.")
        raise last if last else PodSetupError("pod setup failed")

    def _drop_failed_pod(self) -> None:
        self.terminate()  # no-op if the raising site already did
        self.pod_id = None
        self.base_url = None
        self.ssh_host = None
        self.remove_state()

    def arm_self_destruct(self, minutes: int) -> bool:
        """Schedule the pod to terminate itself after `minutes`. Returns success.

        Runs entirely on the pod: a detached sleep-then-terminate, using the
        RUNPOD_POD_ID that RunPod injects, so the pod removes only itself and
        no account API key is ever copied onto a community-cloud host we do
        not control.

        This is the only cleanup that survives the local process exiting, so
        it is the backstop for every other mechanism -- atexit handlers,
        signal handlers and session watchdogs all die with their process.
        """
        if not self.ssh_host or minutes <= 0:
            return False
        self.disarm_self_destruct()  # never stack two reapers
        script = (
            f"nohup setsid sh -c 'sleep {minutes * 60}; "
            f"runpodctl remove pod $RUNPOD_POD_ID' "
            f">/tmp/lossy_reaper.log 2>&1 & echo $! > {REAPER_PID_FILE}"
        )
        try:
            r = self.ssh_cmd(script, timeout=30)
            return r.returncode == 0
        except Exception as e:
            print(f"  Failed to arm pod self-destruct: {e}")
            return False

    def disarm_self_destruct(self) -> None:
        """Cancel a pending self-destruct, e.g. because a new run reconnected.

        Without this, resuming onto a kept-alive pod would inherit the old
        deadline and the pod would vanish mid-render.
        """
        if not self.ssh_host:
            return
        try:
            self.ssh_cmd(
                f"[ -f {REAPER_PID_FILE} ] && kill $(cat {REAPER_PID_FILE}) 2>/dev/null; "
                f"rm -f {REAPER_PID_FILE}; true",
                timeout=30,
            )
        except Exception:
            pass  # a stale reaper is worth a warning, not a crash

    def ensure_pod(self):
        """Reconnect to existing pod or create a new one."""
        if self.pod_id is not None:
            return
        if not self.reconnect():
            self.create_pod()

    def _is_ours(self, pod: dict) -> bool:
        """Whether a pod belongs to this session.

        The name is the load-bearing test, not the id: the failure this whole
        sweep exists for is precisely the recorded id going stale, so an
        id-only check would miss the case it is meant to catch. Names include
        any adopted from an earlier run's journal.
        """
        return pod.get("name") in self._claimed_names() or pod.get("id") in self._owned_pod_ids

    def sweep_stray_pods(self, spare: set[str] | frozenset[str] = frozenset()) -> list[str]:
        """Terminate this session's leftover pods; report anyone else's.

        terminate() removes the pod id it recorded and warns if that call
        fails -- but a run has been observed reporting "pod not found to
        terminate" while a pod under a *different* id was still up and
        billing. Nothing catches that today: a pod bills until something
        explicitly removes it, and once this process exits there is nobody
        left to notice. So before exiting, ask what is actually still running.

        Pods whose name matches this session are terminated outright. Anything
        else carrying the lossy prefix is another run's (or an older orphan's)
        and is only reported -- killing it would take down a render that is
        very likely someone's active work. Set LOSSY_RUNPOD_SWEEP=1 to
        terminate those too. `spare` (and any kept pod still inside its armed
        deadline) is never touched.

        A claim in the ownership journal is dropped only on evidence the pod
        is gone: a confirmed terminate, or the account no longer showing it.

        Returns the ids of the stray pods found, for tests and callers. One
        pass only: `_exit_sweep` repeats it until the account confirms clean.
        """
        return self._sweep_pass(spare)[0]

    def _exit_sweep(self, spare: set[str] | frozenset[str] = frozenset()) -> bool:
        """Sweep until a pass finds nothing of ours and no claim is outstanding.

        A single pass cannot say "clean": a listing can fail, lag behind a
        pod that exists, or still show one just terminated; a terminate can
        fail. So repeat, bounded by EXIT_SWEEP_ATTEMPTS and CLEANUP_BUDGET_S,
        and return whether the LAST pass was clean. The caller reports; nothing
        here raises. A clean pass is one that had nothing to do, so the pass
        after a terminate is the confirmation.
        """
        deadline = time.time() + CLEANUP_BUDGET_S
        for attempt in range(1, EXIT_SWEEP_ATTEMPTS + 1):
            _, settled = self._sweep_pass(spare)
            if settled:
                return True
            if attempt == EXIT_SWEEP_ATTEMPTS or time.time() + EXIT_SWEEP_DELAY_S >= deadline:
                break
            time.sleep(EXIT_SWEEP_DELAY_S)
        return False

    def _sweep_pass(self, spare: set[str] | frozenset[str]) -> tuple[list[str], bool]:
        """One listing + termination pass. Returns (stray ids found, settled).

        `settled` is True only if the listing worked, none of this session's
        pods appeared in it, and no journal claim is left unconfirmed.
        """
        if not self._owned_pod_ids and not self._create_attempted:
            return [], True  # never asked for a pod, so nothing can have escaped
        if not os.environ.get("RUNPOD_API_KEY"):
            return [], False

        try:
            import runpod
            runpod.api_key = os.environ["RUNPOD_API_KEY"]
            pods = runpod.get_pods() or []
        except Exception as e:
            print(f"  Stray-pod check failed ({e}) -- "
                  f"verify at https://www.runpod.io/console/pods")
            return [], False

        spared = set(spare) | self._spared_ids()
        strays = [
            p for p in pods
            if str(p.get("name", "")).startswith(POD_NAME_PREFIX)
            and p.get("desiredStatus", "RUNNING") != "EXITED"
            and p.get("id") and p["id"] not in spared
        ]

        force = os.environ.get("LOSSY_RUNPOD_SWEEP", "").lower() in ("1", "true", "yes")
        settled = True
        for pod in strays:
            pod_id = pod["id"]
            ours = self._is_ours(pod)
            if ours:
                settled = False   # it was up this pass; only the next pass can confirm it went
            if ours or force:
                why = "ours" if ours else "LOSSY_RUNPOD_SWEEP=1"
                print(f"  Sweep: terminating stray pod {pod_id} ({why})")
                if ours:
                    err = self._release(pod_id)
                else:
                    try:
                        runpod.terminate_pod(pod_id)
                        err = None
                    except Exception as e:
                        err = e
                if err is not None and not _is_not_found(err):
                    print(f"  Sweep: FAILED to terminate {pod_id} ({err}) -- "
                          f"terminate it at https://www.runpod.io/console/pods")
            else:
                print(f"  NOTE: pod {pod_id} ({pod.get('name')}) is still running and "
                      f"billing, but is not this session's.")
                print("        Probably a concurrent run. If it is orphaned, terminate it "
                      "at https://www.runpod.io/console/pods (or re-run with "
                      "LOSSY_RUNPOD_SWEEP=1).")

        # Claims for pods the listing does not show: drop those the account
        # confirms are gone. A listing can lag, so ask for the pod by id first.
        running = {p.get("id") for p in pods if p.get("desiredStatus", "RUNNING") != "EXITED"}
        for pid in list(self._journal.snapshot.claims):
            if pid in running or pid in spared:
                continue
            status = self._pod_status(runpod, pid)
            if status == "gone":
                self._journal_write(self._journal.drop_pod, pid)
            elif status == "running":
                print(f"  Sweep: pod {pid} is running though the listing omits it; terminating")
                self._release(pid)
                settled = False
            else:
                settled = False   # the account would not say; do not claim it is gone
        # A claim that survives the pass (terminate failed, or still listed)
        # is owed to the account whatever the loop above concluded.
        if any(pid not in spared for pid in self._journal.snapshot.claims):
            settled = False
        return [p["id"] for p in strays], settled

    def pod_alive(self) -> bool:
        """Whether our pod still exists and is running.

        Conservative on purpose: any doubt (no id, API error) reports alive,
        so a flaky status call never triggers a needless re-provision.
        """
        if self.pod_id is None:
            return False
        if not os.environ.get("RUNPOD_API_KEY"):
            return True
        try:
            import runpod
            runpod.api_key = os.environ["RUNPOD_API_KEY"]
            status = runpod.get_pod(self.pod_id)
        except Exception:
            return True
        if not status:
            return False
        if status.get("desiredStatus") == "EXITED":
            return False
        runtime = status.get("runtime")
        return bool(runtime and runtime.get("ports"))

    def forget_pod(self) -> None:
        """Abandon the current pod and clear all state so the next ensure_pod
        provisions rather than reconnecting to it.

        Tries to terminate the pod first. pod_alive() reports False for a pod
        that is merely not exposing ports at that instant (restarting, status
        API lagging), so "forgotten" does not mean "gone": a run once dropped
        its handle on a pod that was still polling a job and billed for it
        alongside the replacement. Terminating an already-gone pod just fails,
        which is fine -- that failure is quiet because it is the expected case.

        The id stays in _owned_pod_ids either way, so if the terminate call
        failed on a pod that is in fact alive, the exit sweep still finds it.
        """
        pod_id = self.pod_id
        if pod_id:
            self._terminate_quietly(pod_id)
        self.pod_id = None
        self.base_url = None
        self.ssh_host = None
        self.ssh_port = None
        self.wedged = False
        self.close_http()
        self.remove_state()

    def _terminate_pod_id(self, pod_id: str) -> Exception | None:
        """Ask RunPod to terminate `pod_id`, retrying transient errors.

        Returns None on success, else the last error. A single API hiccup used
        to be final: the pod kept billing next to its replacement until the
        run ended. "Not found" means the pod is already gone and is not retried.
        """
        import runpod

        runpod.api_key = os.environ.get("RUNPOD_API_KEY")
        err: Exception | None = None
        for attempt in range(1, TERMINATE_ATTEMPTS + 1):
            try:
                runpod.terminate_pod(pod_id)
                return None
            except Exception as e:
                err = e
                if _is_not_found(e) or attempt == TERMINATE_ATTEMPTS:
                    break
                time.sleep(TERMINATE_BACKOFF_S * attempt)
        return err

    def _terminate_quietly(self, pod_id: str) -> None:
        err = self._release(pod_id)
        if err is None:
            print(f"  Terminated abandoned pod {pod_id}.")
        else:
            print(f"  Abandoned pod {pod_id} not confirmed terminated ({err}); "
                  f"it may still be billing. The exit sweep will retry.")

    def terminate(self):
        """Terminate the pod and remove state file."""
        if self.pod_id is None:
            return

        pod_id = self.pod_id
        self.pod_id = None  # Prevent double-terminate
        self.wedged = False

        elapsed_h = (time.time() - self.pod_start_time) / 3600 if self.pod_start_time else 0
        estimated_cost = elapsed_h * self.gpu_hourly_rate

        err = self._release(pod_id)
        if err is None:
            print(f"  Pod {pod_id} terminated. Session cost: ~${estimated_cost:.2f} ({elapsed_h:.1f}hrs)")
        else:
            print(f"  Warning: Failed to terminate pod {pod_id}: {err}")
            print("  Manually terminate at https://www.runpod.io/console/pods")

        self.remove_state()

    # -- SSH helpers --

    @property
    def ssh_identity(self) -> pod_ssh.SshIdentity:
        """The one key used to register with RunPod and to ssh in. Resolved
        (and verified) on first use; raises SshKeyError if it is unusable."""
        if self._ssh_identity is None:
            self._ssh_identity = pod_ssh.resolve_ssh_identity()
            print(f"  Pod ssh key: {self._ssh_identity.private_path} "
                  f"({self._ssh_identity.fingerprint})")
        return self._ssh_identity

    def _ssh_opts(self) -> list[str]:
        """Common SSH options for all remote commands: offer our key only."""
        return pod_ssh.ssh_options(self.ssh_identity)

    def wait_for_ssh(self) -> None:
        """Block until the pod's sshd accepts our key.

        Not-ready (refused, timed out, dropped) backs off and retries up to
        SSH_READY_ATTEMPTS, then raises PodSetupError (a fresh pod may be
        healthy). A server that answers "Permission denied" rejected our key:
        confirmed once, then PodSshAuthError, which with_setup_retry does not
        retry. Either way the pod is terminated first.
        """
        if not self.ssh_host:
            return
        identity = self.ssh_identity
        rejections = 0
        detail = ""
        for attempt in range(1, SSH_READY_ATTEMPTS + 1):
            try:
                r = self.ssh_cmd("true", timeout=SSH_PROBE_TIMEOUT_S)
                outcome = pod_ssh.classify_ssh_failure(r.returncode, r.stderr)
                err_lines = (r.stderr or "").strip().splitlines()
                detail = err_lines[-1] if err_lines else ""
            except subprocess.TimeoutExpired:
                outcome, detail = SshOutcome.NOT_READY, "timed out"
            if outcome is SshOutcome.OK:
                print(f"  SSH ready (offered {identity.fingerprint}, attempt {attempt}).")
                return
            if outcome is SshOutcome.AUTH_REJECTED:
                rejections += 1
                if rejections >= SSH_AUTH_REJECTIONS_FATAL:
                    self.terminate()
                    raise PodSshAuthError(
                        f"pod {self.pod_id} rejected ssh key {identity.private_path} "
                        f"({identity.fingerprint}): {detail}. Not retrying: another pod would "
                        f"get the same key. Check the key RunPod was given (PUBLIC_KEY) and "
                        f"{pod_ssh.KEY_ENV}.")
            else:
                rejections = 0
            if attempt < SSH_READY_ATTEMPTS:
                wait = min(5 * attempt, 20)
                print(f"  SSH not ready ({detail or outcome.value}); retrying in {wait}s "
                      f"[{attempt}/{SSH_READY_ATTEMPTS}]")
                time.sleep(wait)
        self.terminate()
        raise PodSetupError(f"sshd on pod did not accept connections after {SSH_READY_ATTEMPTS} "
                            f"attempts ({detail or 'no answer'})")

    def ssh_cmd(self, cmd: str, timeout: int = 60, input: str | None = None) -> subprocess.CompletedProcess:
        """Run a command on the pod via SSH. `input` goes to the command's stdin
        (how a secret reaches the pod without appearing in any argv)."""
        return subprocess.run(
            ["ssh", *self._ssh_opts(), "-o", "ServerAliveInterval=30",
             "-p", str(self.ssh_port), f"root@{self.ssh_host}", cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
            input=input,
        )

    def ssh_bytes(self, cmd: str, timeout: int = 60) -> subprocess.CompletedProcess:
        """Like ssh_cmd but stdout is raw bytes (binary file transfer)."""
        return subprocess.run(
            ["ssh", *self._ssh_opts(),
             "-p", str(self.ssh_port), f"root@{self.ssh_host}", cmd],
            capture_output=True,
            timeout=timeout,
        )

    def ssh_bg(self, cmd: str):
        """Run a command on the pod via SSH in the background (detached)."""
        p = subprocess.Popen(
            ["ssh", "-f", *self._ssh_opts(),
             "-p", str(self.ssh_port), f"root@{self.ssh_host}", cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            p.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()

    def scp_to(self, local_paths: list[str], remote_dir: str, timeout: int = 60) -> subprocess.CompletedProcess:
        """Copy local files to a directory on the pod via scp."""
        return subprocess.run(
            ["scp", *self._ssh_opts(),
             "-P", str(self.ssh_port),
             *local_paths,
             f"root@{self.ssh_host}:{remote_dir}/"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    # -- ComfyUI helpers --

    @property
    def http(self):
        """Shared, connection-pooled HTTP client for every ComfyUI call.

        Each call used to be a bare `httpx.get`/`httpx.post`, which opens a
        fresh TCP+TLS connection and drops it. Generation polls /history every
        ~2s, so the first 50 shots of the full run opened 1,249 connections
        (~25 a shot, ~48,000 extrapolated across 2069). That churn is what
        exhausted the local ephemeral port range and lost shot 56 outright to
        `[Errno 49] Can't assign requested address` -- a failure on this
        machine, not on the pod. Pooling collapses it to a handful of
        connections held open per host.
        """
        if self._http is None:
            import httpx
            self._http = httpx.Client(
                limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
                timeout=30,
            )
        return self._http

    def close_http(self):
        """Release pooled connections. Safe to call more than once."""
        if self._http is not None:
            try:
                self._http.close()
            except Exception:
                pass
            self._http = None

    def wait_for_comfyui(self):
        """Poll ComfyUI until it responds to /system_stats."""
        import httpx

        print("  Waiting for ComfyUI...")
        start = time.time()
        while time.time() - start < COMFYUI_READY_TIMEOUT:
            try:
                resp = self.http.get(f"{self.base_url}/system_stats", timeout=10)
                if resp.status_code == 200:
                    print("  ComfyUI is ready.")
                    return
            except (httpx.ConnectError, httpx.TimeoutException, httpx.ReadError):
                pass
            elapsed = int(time.time() - start)
            print(f"  Waiting for ComfyUI... ({elapsed}s)", end="\r")
            time.sleep(5)

        print(f"\n  Error: ComfyUI did not become ready within {COMFYUI_READY_TIMEOUT}s")
        if self.ssh_host:
            try:
                result = self.ssh_cmd("tail -50 /tmp/comfyui.log", timeout=10)
                if result.stdout:
                    print(f"  ComfyUI log:\n{result.stdout}")
            except Exception:
                pass
        self.terminate()
        # Retryable: this is overwhelmingly a bad host (CUDA failing to
        # initialise before ComfyUI starts), and a fresh pod usually works.
        raise PodSetupError("ComfyUI did not come up on this pod")

    def comfyui_python_bin(self) -> str:
        """The python that runs ComfyUI on the pod (its venv if it has one)."""
        find_python = self.ssh_cmd(
            f"if [ -x {COMFYUI_DIR}/.venv/bin/python ]; then echo .venv/bin/python; "
            f"elif command -v python3 >/dev/null; then echo python3; "
            f"else echo python; fi",
            timeout=10,
        )
        return find_python.stdout.strip() if find_python.returncode == 0 else "python3"

    def ensure_comfyui_version(self, minimum: tuple[int, ...], tag: str) -> None:
        """Upgrade the pod's ComfyUI to `tag` if it is older than `minimum`.

        The stock runpod/comfyui:latest image shipped v0.30.0, which predates
        the LTX-2.5 nodes. Checking out a release tag (not master) keeps the
        pod reproducible. Raises PodSetupError if the upgrade does not take,
        so with_setup_retry can try a fresh pod. The caller restarts ComfyUI.
        """
        def current() -> tuple[int, ...] | None:
            r = self.ssh_cmd(f"cat {COMFYUI_DIR}/comfyui_version.py", timeout=15)
            return parse_comfyui_version(r.stdout or "") if r.returncode == 0 else None

        have = current()
        if have is not None and have >= minimum:
            print(f"  ComfyUI v{'.'.join(map(str, have))}: new enough")
            return
        want = ".".join(map(str, minimum))
        print(f"  ComfyUI is {have or 'unknown'}, need >= {want}; upgrading to {tag}...")
        python_bin = self.comfyui_python_bin()
        r = self.ssh_cmd(
            f"cd {COMFYUI_DIR} && git fetch --tags origin 2>&1 | tail -2 "
            f"&& git checkout -f {tag} 2>&1 | tail -2 "
            f"&& {python_bin} -m pip install -q -r requirements.txt 2>&1 | tail -5 && echo UPGRADE_OK",
            timeout=900,
        )
        if "UPGRADE_OK" not in (r.stdout or ""):
            raise PodSetupError(f"ComfyUI upgrade to {tag} failed: {(r.stdout or r.stderr or '')[-300:]}")
        have = current()
        if have is None or have < minimum:
            raise PodSetupError(f"ComfyUI still {have} after upgrade to {tag}")
        print(f"  ComfyUI now v{'.'.join(map(str, have))}")

    def restart_comfyui(self):
        """Stop ComfyUI, then start it again. Used after installing custom nodes."""
        print("  Restarting ComfyUI...")
        self.ssh_cmd('pkill -f "main.py" || true', timeout=10)
        time.sleep(2)

        python_bin = self.comfyui_python_bin()
        self.ssh_bg(
            f"cd {COMFYUI_DIR} && {python_bin} main.py --listen 0.0.0.0 --port 8188 "
            f"</dev/null >/tmp/comfyui.log 2>&1"
        )
        self.wait_for_comfyui()

    def download_models(self, models: list[tuple[str, str]], timeout: int = 600):
        """Download model files to ComfyUI models dir via SSH.

        Args:
            models: List of (relative_dest_path, url) tuples.
                    e.g. ("vae/wan_2.1_vae.safetensors", "https://...")
            timeout: Per-file SSH command timeout in seconds. Default 600s covers
                     the ~10GB Wan model files; larger checkpoints (e.g. LTX-2's
                     22B FP8, ~22GB) need a longer allowance.
        """
        models_dir = f"{COMFYUI_DIR}/models"
        print(f"  Downloading models via SSH ({self.ssh_host}:{self.ssh_port})...")

        for dest_path, url in models:
            full_path = f"{models_dir}/{dest_path}"
            filename = os.path.basename(dest_path)

            check = self.ssh_cmd(f'[ -s {full_path} ] && echo exists || echo missing')
            if check.returncode == 0 and "exists" in check.stdout:
                print(f"    {filename}: already exists")
                continue

            print(f"    {filename}: downloading...")
            dir_path = os.path.dirname(full_path)
            result = self.ssh_cmd(
                f"mkdir -p {dir_path} && wget -q -O {full_path} '{url}' && echo OK",
                timeout=timeout,
            )
            if result.returncode == 0 and "OK" in result.stdout:
                print(f"    {filename}: done")
            else:
                print(f"    {filename}: FAILED (exit {result.returncode})")
                if result.stderr:
                    print(f"      {result.stderr[:200]}")

    def download_pinned_models(self, models: list[PinnedModel], timeout: int = 1800) -> None:
        """Download weights and verify each against its sha256.

        Files land at `<dest>.part` and are only renamed into place once the
        hash matches, so an interrupted or corrupt download can never be
        mistaken for a finished one by the "already exists" check (plain
        download_models leaves a truncated file behind on failure). A mismatch
        raises PodSetupError: the pod is retryable, the bytes are not trusted.
        """
        models_dir = f"{COMFYUI_DIR}/models"
        print(f"  Downloading {len(models)} pinned models via SSH ({self.ssh_host}:{self.ssh_port})...")
        for m in models:
            full = f"{models_dir}/{m.dest}"
            name = os.path.basename(m.dest)
            check = self.ssh_cmd(f"[ -s {full} ] && echo exists || echo missing")
            if check.returncode == 0 and "exists" in check.stdout:
                print(f"    {name}: already exists")
                continue
            print(f"    {name}: downloading...")
            token = None
            if m.gate_url:
                token = os.environ.get("HF_TOKEN")
                if not token:
                    self.terminate()  # not a PodSetupError: with_setup_retry would leave this pod billing
                    raise GatedModelError(f"{name} is gated and HF_TOKEN is not set. {gated_help(m)}")
                # The token rides ssh's stdin and is read into a shell variable on the
                # pod, so it is in no argv, no log line and no error message here.
                fetch = ('IFS= read -r HF_TOKEN && wget -q --header="Authorization: Bearer $HF_TOKEN"'
                         f" -O {full}.part '{m.url}'")
            else:
                fetch = f"wget -q -O {full}.part '{m.url}'"
            r = self.ssh_cmd(
                f"mkdir -p {os.path.dirname(full)} && {fetch} "
                f"&& echo '{m.sha256}  {full}.part' | sha256sum -c --status "
                f"&& mv {full}.part {full} && echo VERIFIED",
                timeout=timeout,
                **({"input": f"{token}\n"} if token else {}),
            )
            if r.returncode != 0 or "VERIFIED" not in (r.stdout or ""):
                self.ssh_cmd(f"rm -f {full}.part", timeout=30)
                # wget exits 6 on a 401/403: the token was refused at download time.
                if m.gate_url and r.returncode == 6:
                    self.terminate()
                    raise GatedModelError(f"{name}: Hugging Face refused the download on the pod "
                                          f"(401/403). {gated_help(m)}")
                raise PodSetupError(
                    f"{name}: download or sha256 check failed (exit {r.returncode}) "
                    f"{(r.stderr or '')[:200]}")
            print(f"    {name}: verified")

    def submit_workflow(self, workflow: dict, timeout: int = 600) -> dict | None:
        """Submit a ComfyUI workflow and wait for completion.

        Returns the history entry for the prompt, or None on failure.
        """
        resp = self.http.post(
            f"{self.base_url}/prompt",
            json={"prompt": workflow},
            timeout=30,
        )
        if resp.status_code != 200:
            error_detail = resp.text[:300] if resp.text else "no body"
            print(f"  ComfyUI rejected workflow ({resp.status_code}): {error_detail}")
            return None

        prompt_id = resp.json()["prompt_id"]

        start = time.time()
        try:
            while time.time() - start < timeout:
                resp = self.http.get(
                    f"{self.base_url}/history/{prompt_id}",
                    timeout=10,
                )
                history = resp.json()
                if prompt_id in history:
                    status = history[prompt_id].get("status", {})
                    if status.get("status_str") != "success":
                        print(f"  ComfyUI error: {status}")
                        return None
                    return history[prompt_id]
                time.sleep(2)
        except Exception:
            # A poll that errors out abandons the job just as a timeout does.
            self.cancel_job(prompt_id)
            raise

        print(f"  Timeout waiting for workflow ({timeout}s)")
        self.cancel_job(prompt_id)
        return None

    def cancel_job(self, prompt_id: str) -> bool:
        """Stop a ComfyUI job we have given up on. Returns True once it is gone.

        Abandoning a poll does not stop the job: ComfyUI runs one job at a
        time, so a hung one sits at the head of the queue and every retry
        queued behind it waits forever (research 0026). Drop it from the queue,
        interrupt it if running, then confirm. If the worker is still held
        afterwards -- or the pod cannot be asked -- mark the session wedged so
        the caller replaces the pod instead of retrying on it.
        """
        try:
            self.http.post(f"{self.base_url}/queue", json={"delete": [prompt_id]}, timeout=10)
            self.http.post(f"{self.base_url}/interrupt", json={"prompt_id": prompt_id}, timeout=10)
            for _ in range(CANCEL_CONFIRM_POLLS):
                q = self.http.get(f"{self.base_url}/queue", timeout=10).json()
                held = [item[1] for item in
                        q.get("queue_running", []) + q.get("queue_pending", [])]
                if prompt_id not in held:
                    return True
                time.sleep(2)
        except Exception as e:
            print(f"  Could not cancel job {prompt_id}: {e}")
        print(f"  Job {prompt_id} would not stop; marking pod {self.pod_id} wedged.")
        self.wedged = True
        return False

    def download_output(self, output_file: dict) -> bytes | None:
        """Download an output file from ComfyUI's /view endpoint."""
        try:
            resp = self.http.get(
                f"{self.base_url}/view",
                params={
                    "filename": output_file["filename"],
                    "subfolder": output_file.get("subfolder", ""),
                    "type": output_file.get("type", "output"),
                },
                timeout=60,
                follow_redirects=True,
            )
            resp.raise_for_status()
            return resp.content
        except Exception as e:
            print(f"  Error downloading output: {e}")
            return None

    def free_vram(self):
        """Free cached VRAM from last generation (keep models loaded)."""
        try:
            self.http.post(f"{self.base_url}/free", json={"free_memory": True}, timeout=10)
        except Exception:
            pass
