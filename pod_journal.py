"""Durable record of which RunPod pods a session is answerable for.

`runpod_pod.json` is connection state for ONE pod and is deleted whenever a pod
is dropped, even when the terminate failed. This journal is the ownership
record: it survives SIGKILL and power loss, so the next run can find and end
whatever the dead one left billing. Policy: docs/design/adr/0010-pod-ownership-journal.md.

Shape (one file per session, `<dir>/<run_id>.json`):

    {"version": 1, "run_id": "ab12cd34", "name": "lossy-comfyui-ab12cd34",
     "pods": {"<pod id>": {"name": ..., "state": "owned"|"kept"|"abandoned",
                           "kept_until": epoch|null}},
     "windows": [{"name": ..., "until": epoch}]}

* `owned`     a live pod this session answers for.
* `kept`      deliberately left running with a self-destruct armed; spared until
              `kept_until`, terminated after it if the reaper evidently failed.
* `abandoned` the handle was dropped / a terminate was requested but not yet
              confirmed. Written BEFORE the terminate call.
* window      a pod may exist under `name` whose id we never received (a create
              request that timed out, or was in flight when the process died).
              Written BEFORE the create request; expires, since a late create
              cannot be late forever.

Liveness is a kernel `flock` on `<run_id>.lock`, held for the session's life.
The kernel releases it when the process dies by any means, so "is the owner
dead" has no pid-reuse or clock question. Each session writes only its own
file; peers are only ever read, or discarded after their claims were adopted.
"""

import fcntl
import json
import os
import time
from dataclasses import dataclass, field

VERSION = 1

OWNED = "owned"
KEPT = "kept"
ABANDONED = "abandoned"
STATES = (OWNED, KEPT, ABANDONED)


@dataclass
class Claim:
    pod_id: str
    name: str
    state: str = OWNED
    kept_until: float | None = None

    def to_json(self) -> dict:
        return {"name": self.name, "state": self.state, "kept_until": self.kept_until}


@dataclass
class Window:
    name: str
    until: float


@dataclass
class Snapshot:
    """What one journal file says. Pure data, however it was obtained."""

    run_id: str
    name: str
    claims: dict[str, Claim] = field(default_factory=dict)
    windows: list[Window] = field(default_factory=list)

    @property
    def names(self) -> set[str]:
        """Every pod name this journal can speak for."""
        return ({self.name} | {c.name for c in self.claims.values()}
                | {w.name for w in self.windows})

    def is_empty(self, now: float) -> bool:
        return not self.claims and not any(w.until > now for w in self.windows)

    def to_json(self) -> dict:
        return {
            "version": VERSION, "run_id": self.run_id, "name": self.name,
            "pods": {pid: c.to_json() for pid, c in self.claims.items()},
            "windows": [{"name": w.name, "until": w.until} for w in self.windows],
        }

    @classmethod
    def from_json(cls, data: dict) -> "Snapshot":
        snap = cls(run_id=str(data["run_id"]), name=str(data["name"]))
        for pid, c in (data.get("pods") or {}).items():
            state = c.get("state", OWNED)
            snap.claims[str(pid)] = Claim(
                str(pid), c.get("name") or snap.name,
                state if state in STATES else OWNED, c.get("kept_until"))
        for w in data.get("windows") or []:
            snap.windows.append(Window(str(w["name"]), float(w["until"])))
        return snap


def _lock(path: str):
    """Take the exclusive lock at `path`; None if another process holds it."""
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    return fd


def _unlock(fd: int | None) -> None:
    if fd is not None:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _write_atomic(path: str, payload: dict) -> None:
    """Tmp + fsync + rename: a power cut leaves the old or the new file, never half."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class PodJournal:
    """This session's own journal. Every mutation is on disk before it returns."""

    def __init__(self, directory: str, run_id: str, name: str):
        self.directory = directory
        self.snapshot = Snapshot(run_id=run_id, name=name)
        self._fd: int | None = None

    @property
    def json_path(self) -> str:
        return os.path.join(self.directory, f"{self.snapshot.run_id}.json")

    @property
    def lock_path(self) -> str:
        return os.path.join(self.directory, f"{self.snapshot.run_id}.lock")

    @property
    def is_open(self) -> bool:
        return self._fd is not None

    def open(self) -> None:
        if self._fd is not None:
            return
        os.makedirs(self.directory, exist_ok=True)
        fd = _lock(self.lock_path)
        if fd is None:
            raise RuntimeError(f"pod journal {self.lock_path} is held by another process")
        self._fd = fd

    def _save(self) -> None:
        self.open()
        _write_atomic(self.json_path, self.snapshot.to_json())

    # -- mutations (each one persists) --

    def set_pod(self, pod_id: str, name: str, state: str = OWNED,
                kept_until: float | None = None) -> None:
        self.snapshot.claims[pod_id] = Claim(pod_id, name, state, kept_until)
        self._save()

    def drop_pod(self, pod_id: str) -> None:
        if self.snapshot.claims.pop(pod_id, None) is not None:
            self._save()

    def open_window(self, name: str, until: float) -> None:
        self.snapshot.windows = [w for w in self.snapshot.windows if w.name != name]
        self.snapshot.windows.append(Window(name, until))
        self._save()

    def close_window(self, name: str) -> None:
        before = len(self.snapshot.windows)
        self.snapshot.windows = [w for w in self.snapshot.windows if w.name != name]
        if len(self.snapshot.windows) != before:
            self._save()

    def adopt(self, claims: list[Claim], windows: list[Window]) -> None:
        """Take over a dead peer's unresolved claims, in one write."""
        for c in claims:
            state = KEPT if c.state == KEPT else ABANDONED
            self.snapshot.claims[c.pod_id] = Claim(c.pod_id, c.name, state, c.kept_until)
        for w in windows:
            self.snapshot.windows = [x for x in self.snapshot.windows if x.name != w.name]
            self.snapshot.windows.append(w)
        self._save()

    def close(self) -> None:
        """Release the lock; delete the files when nothing is owed to anyone."""
        if self._fd is None:
            return
        now = time.time()
        self.snapshot.windows = [w for w in self.snapshot.windows if w.until > now]
        if self.snapshot.claims or self.snapshot.windows:
            _write_atomic(self.json_path, self.snapshot.to_json())
        else:
            _remove(self.json_path)
            _remove(self.lock_path)
        _unlock(self._fd)
        self._fd = None


def _remove(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


@dataclass
class Peer:
    """Another session's journal found on disk."""

    snapshot: Snapshot | None       # None: unreadable
    json_path: str
    lock_path: str
    live: bool                      # its owner still holds the lock
    _fd: int | None = None

    def discard(self) -> None:
        """Delete a dead peer's files. Only call once its claims are adopted or resolved."""
        _remove(self.json_path)
        _remove(self.lock_path)
        _unlock(self._fd)
        self._fd = None

    def release(self) -> None:
        _unlock(self._fd)
        self._fd = None


def scan_peers(directory: str, exclude_run_id: str) -> list[Peer]:
    """Every other journal in `directory`. Dead peers come back with their lock held.

    A journal is dead if its lock can be taken. Holding it while we process it
    stops a second reconciling process from handling the same leftovers.
    """
    if not os.path.isdir(directory):
        return []
    peers: list[Peer] = []
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".json") or fname == f"{exclude_run_id}.json":
            continue
        json_path = os.path.join(directory, fname)
        lock_path = json_path[:-len(".json")] + ".lock"
        fd = _lock(lock_path)
        try:
            with open(json_path) as f:
                snap = Snapshot.from_json(json.load(f))
        except FileNotFoundError:
            _unlock(fd)           # discarded under us by a concurrent reconciler
            continue
        except (ValueError, KeyError, TypeError, AttributeError):
            snap = None
        peers.append(Peer(snap, json_path, lock_path, live=fd is None, _fd=fd))
    return peers
