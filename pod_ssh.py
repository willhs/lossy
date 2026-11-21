"""How lossy authenticates to a RunPod pod over ssh.

Research 0029: about 50 of ~55 pods refused ssh with "Permission denied / Too
many authentication failures". The session handed RunPod the public half of
~/.ssh/id_ed25519 but then ran a bare `ssh`, which offers whatever the agent
holds first (1Password's agent serves many keys) and reads ~/.ssh/config. The
server hangs up after a handful of offers, before our key is ever tried.

The fix is one identity, decided once and used for both halves:

- `resolve_ssh_identity()` picks a private key, derives its public key from the
  private file itself, and refuses (SshKeyError) if the key is unreadable,
  passphrase-protected or does not match its `.pub`. That runs before any pod is
  rented, so a bad key costs nothing.
- `ssh_options()` makes every ssh/scp offer exactly that key and nothing else.
- `classify_ssh_failure()` separates "sshd is not up yet" (retry) from "the
  server rejected our key" (definite; retrying cannot fix it).

Pure and side-effect free apart from running `ssh-keygen -y` on the key. Private
key material is never read into this process, printed or copied.
"""

import base64
import hashlib
import os
import subprocess
from enum import Enum
from pathlib import Path
from typing import Mapping, NamedTuple

# Search order when LOSSY_RUNPOD_SSH_KEY is unset.
DEFAULT_KEY_NAMES = ("id_ed25519", "id_rsa")
KEY_ENV = "LOSSY_RUNPOD_SSH_KEY"


class SshKeyError(RuntimeError):
    """The key lossy would use for pods is missing or unusable. NOT retryable:
    raised before any pod is rented, so nothing is billing."""


class PodSshAuthError(RuntimeError):
    """A booted pod's sshd definitively rejected our key. NOT retryable: a fresh
    pod is given the same key and refuses it the same way."""


class SshIdentity(NamedTuple):
    private_path: str
    public_key: str      # "<type> <base64>", no comment; what PUBLIC_KEY carries
    fingerprint: str     # "SHA256:...", what `ssh-keygen -l` prints


class SshOutcome(Enum):
    OK = "ok"
    AUTH_REJECTED = "auth_rejected"   # sshd answered and said no: definite
    NOT_READY = "not_ready"           # refused / timed out / dropped: sshd not up yet


def _fingerprint(public_key: str) -> str:
    blob = base64.b64decode(public_key.split()[1])
    digest = base64.b64encode(hashlib.sha256(blob).digest()).decode().rstrip("=")
    return f"SHA256:{digest}"


def _key_fields(line: str) -> str:
    """'<type> <base64>' from an authorized_keys-style line (drops the comment)."""
    parts = line.split()
    if len(parts) < 2:
        raise ValueError("not a public key line")
    return f"{parts[0]} {parts[1]}"


def _derive_public(private_path: str) -> str:
    try:
        r = subprocess.run(
            ["ssh-keygen", "-y", "-P", "", "-f", private_path],
            capture_output=True, text=True, timeout=15, stdin=subprocess.DEVNULL,
        )
    except FileNotFoundError as e:
        raise SshKeyError("ssh-keygen is not installed; cannot check the pod ssh key") from e
    except subprocess.TimeoutExpired as e:
        raise SshKeyError(f"ssh-keygen timed out reading {private_path}") from e
    if r.returncode != 0:
        why = (r.stderr or "").strip().splitlines()
        hint = ""
        if why and "passphrase" in why[-1].lower():
            hint = (" Pod ssh runs without an agent, so the key must have no passphrase; "
                    f"point {KEY_ENV} at a dedicated passphrase-less key.")
        raise SshKeyError(f"cannot use {private_path} for pod ssh: "
                          f"{why[-1] if why else 'ssh-keygen failed'}.{hint}")
    try:
        return _key_fields(r.stdout)
    except ValueError as e:
        raise SshKeyError(f"ssh-keygen gave no public key for {private_path}") from e


def resolve_ssh_identity(env: Mapping[str, str] | None = None,
                         home: str | None = None) -> SshIdentity:
    """Pick the private key for pod ssh and prove its public half is the one we register.

    `LOSSY_RUNPOD_SSH_KEY` names the private key explicitly; otherwise the first
    of ~/.ssh/id_ed25519, ~/.ssh/id_rsa that exists is used. A key that exists
    but fails any check raises rather than falling through to the next: a
    quiet switch is how a pod ends up registered with one key and ssh'd with
    another.
    """
    env = os.environ if env is None else env
    explicit = env.get(KEY_ENV)
    if explicit:
        candidates = [Path(explicit).expanduser()]
    else:
        ssh_dir = Path(home).expanduser() / ".ssh" if home else Path("~/.ssh").expanduser()
        candidates = [ssh_dir / n for n in DEFAULT_KEY_NAMES]
    private = next((p for p in candidates if p.is_file()), None)
    if private is None:
        searched = ", ".join(str(p) for p in candidates)
        raise SshKeyError(
            f"no ssh private key for RunPod pods (looked for {searched}). "
            f"Create one or set {KEY_ENV} to a passphrase-less private key readable by this process.")
    if not os.access(private, os.R_OK):
        raise SshKeyError(f"{private} exists but this process cannot read it "
                          f"(a sandbox or permissions issue); set {KEY_ENV} to a readable key.")

    derived = _derive_public(str(private))
    pub_file = Path(str(private) + ".pub")
    if pub_file.is_file():
        try:
            registered = _key_fields(pub_file.read_text())
        except (OSError, ValueError) as e:
            raise SshKeyError(f"{pub_file} is not a readable public key ({e})") from e
        if registered != derived:
            raise SshKeyError(
                f"{pub_file} does not match {private}: the pod would be given one key and "
                f"ssh'd with another. Fix or delete the .pub (fingerprint of the private key is "
                f"{_fingerprint(derived)}).")
    return SshIdentity(str(private), derived, _fingerprint(derived))


def ssh_options(identity: SshIdentity) -> list[str]:
    """ssh/scp options that offer `identity` and nothing else.

    -F /dev/null ignores ~/.ssh/config (its IdentityAgent, ProxyJump, Include...);
    IdentityAgent=none + IdentitiesOnly=yes stop agent keys and default key files
    being offered ahead of ours; BatchMode makes a rejection fail instead of
    prompting.
    """
    return [
        "-F", "/dev/null",
        "-i", identity.private_path,
        "-o", "IdentitiesOnly=yes",
        "-o", "IdentityAgent=none",
        "-o", "PreferredAuthentications=publickey",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=15",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
    ]


_AUTH_MARKERS = ("permission denied", "too many authentication failures")


def classify_ssh_failure(returncode: int, stderr: str) -> SshOutcome:
    if returncode == 0:
        return SshOutcome.OK
    low = (stderr or "").lower()
    if any(m in low for m in _AUTH_MARKERS):
        return SshOutcome.AUTH_REJECTED
    return SshOutcome.NOT_READY
