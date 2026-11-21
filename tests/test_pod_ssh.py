"""Pod ssh authentication: one explicit key, verified up front, probed with backoff.

Research 0029: ~50 of ~55 pods refused ssh with "Permission denied / Too many
authentication failures". These tests use real keys (ssh-keygen, local) and a
fake `ssh` executable on PATH that behaves like the failing servers did, so what
is asserted is what the session actually runs -- no real pod, no network.
"""

import json
import os
import stat
import subprocess
import sys
import textwrap

import pytest

import pod_ssh
import runpod_pod
from pod_ssh import PodSshAuthError, SshKeyError
from runpod_pod import PodSetupError, RunPodSession
from test_runpod_pod import _FleetRunpod


def _keygen(path, passphrase=""):
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", passphrase, "-C", "test", "-f", str(path)],
                   check=True)
    return str(path)


def _fingerprint(pub_path):
    out = subprocess.run(["ssh-keygen", "-lf", str(pub_path)], capture_output=True, text=True, check=True)
    return out.stdout.split()[1]


@pytest.fixture
def key(tmp_path):
    return _keygen(tmp_path / "pod_key")


# ---------------------------------------------------------------- key resolution

class TestResolveIdentity:
    def test_explicit_key_gives_matching_public_key_and_fingerprint(self, key):
        ident = pod_ssh.resolve_ssh_identity({pod_ssh.KEY_ENV: key})
        assert ident.private_path == key
        assert ident.fingerprint == _fingerprint(key + ".pub")
        assert ident.public_key == " ".join(open(key + ".pub").read().split()[:2])

    def test_default_search_prefers_ed25519_then_rsa(self, tmp_path):
        ssh_dir = tmp_path / ".ssh"
        ssh_dir.mkdir()
        _keygen(ssh_dir / "id_ed25519")
        ident = pod_ssh.resolve_ssh_identity({}, home=str(tmp_path))
        assert ident.private_path == str(ssh_dir / "id_ed25519")

    def test_public_key_is_derived_when_no_pub_file(self, key):
        os.remove(key + ".pub")
        assert pod_ssh.resolve_ssh_identity({pod_ssh.KEY_ENV: key}).public_key.startswith("ssh-ed25519 ")

    def test_stale_pub_file_fails_loudly(self, tmp_path, key):
        other = _keygen(tmp_path / "other")
        os.replace(other + ".pub", key + ".pub")   # .pub now belongs to a different key
        with pytest.raises(SshKeyError, match="does not match"):
            pod_ssh.resolve_ssh_identity({pod_ssh.KEY_ENV: key})

    def test_passphrase_key_is_refused_with_a_remedy(self, tmp_path):
        k = _keygen(tmp_path / "locked", passphrase="hunter2")
        with pytest.raises(SshKeyError, match="passphrase"):
            pod_ssh.resolve_ssh_identity({pod_ssh.KEY_ENV: k})

    def test_missing_key_names_where_it_looked(self, tmp_path):
        with pytest.raises(SshKeyError, match="id_ed25519"):
            pod_ssh.resolve_ssh_identity({}, home=str(tmp_path))

    def test_missing_explicit_key(self, tmp_path):
        with pytest.raises(SshKeyError, match="nope"):
            pod_ssh.resolve_ssh_identity({pod_ssh.KEY_ENV: str(tmp_path / "nope")})

    @pytest.mark.skipif(os.geteuid() == 0, reason="root reads anything")
    def test_unreadable_key_says_so(self, key):
        os.chmod(key, 0)
        with pytest.raises(SshKeyError, match="cannot read"):
            pod_ssh.resolve_ssh_identity({pod_ssh.KEY_ENV: key})


class TestSshOptions:
    def test_offers_only_our_key_and_ignores_agent_and_config(self, key):
        opts = pod_ssh.ssh_options(pod_ssh.resolve_ssh_identity({pod_ssh.KEY_ENV: key}))
        flat = " ".join(opts)
        assert f"-i {key}" in flat
        assert "IdentitiesOnly=yes" in flat and "IdentityAgent=none" in flat
        assert "-F /dev/null" in flat and "BatchMode=yes" in flat

    @pytest.mark.parametrize("rc,err,want", [
        (0, "", pod_ssh.SshOutcome.OK),
        (255, "root@1.2.3.4: Permission denied (publickey).", pod_ssh.SshOutcome.AUTH_REJECTED),
        (255, "Received disconnect from 1.2.3.4 port 22:2: Too many authentication failures",
         pod_ssh.SshOutcome.AUTH_REJECTED),
        (255, "ssh: connect to host 1.2.3.4 port 22: Connection refused", pod_ssh.SshOutcome.NOT_READY),
        (255, "kex_exchange_identification: read: Connection reset by peer", pod_ssh.SshOutcome.NOT_READY),
        (255, "", pod_ssh.SshOutcome.NOT_READY),
    ])
    def test_classify(self, rc, err, want):
        assert pod_ssh.classify_ssh_failure(rc, err) is want


# ------------------------------------------------------- session against a fake sshd

FAKE_SSH = textwrap.dedent('''\
    #!{python}
    """Stands in for ssh. Scenario in $FAKE_SSH_SCENARIO (json), call log beside it.

    agent_keys: keys an agent would offer first when ssh is not told IdentitiesOnly
                + IdentityAgent=none (the server allows 6 attempts, then hangs up).
    refuse_first: connection refused for the first N calls (sshd not up yet).
    authorized: private key whose public half the pod accepts.
    """
    import json, os, subprocess, sys
    scn_path = os.environ["FAKE_SSH_SCENARIO"]
    scn = json.load(open(scn_path))
    argv = sys.argv[1:]
    scn["calls"] = scn.get("calls", []) + [argv]
    n = len(scn["calls"])
    def done(rc, err=""):
        json.dump(scn, open(scn_path, "w"))
        sys.stderr.write(err)
        sys.exit(rc)
    if n <= scn.get("refuse_first", 0):
        done(255, "ssh: connect to host 1.2.3.4 port 22: Connection refused\\n")
    opts = [argv[i + 1] for i, a in enumerate(argv) if a == "-o"]
    hardened = "IdentitiesOnly=yes" in opts and "IdentityAgent=none" in opts
    if not hardened and scn.get("agent_keys", 0) >= 6:
        done(255, "Received disconnect from 1.2.3.4 port 22:2: Too many authentication failures\\n")
    if "-i" not in argv:
        done(255, "root@1.2.3.4: Permission denied (publickey).\\n")
    offered = argv[argv.index("-i") + 1]
    pub = lambda p: subprocess.run(["ssh-keygen", "-y", "-P", "", "-f", p], capture_output=True,
                                   text=True).stdout.split()[:2]
    if pub(offered) != pub(scn["authorized"]):
        done(255, "root@1.2.3.4: Permission denied (publickey).\\n")
    done(0)
''')


@pytest.fixture
def fake_ssh(tmp_path, monkeypatch, key):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    exe = bin_dir / "ssh"
    exe.write_text(FAKE_SSH.format(python=sys.executable))
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    scenario = tmp_path / "scenario.json"
    monkeypatch.setenv("FAKE_SSH_SCENARIO", str(scenario))
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv(pod_ssh.KEY_ENV, key)
    monkeypatch.setattr(runpod_pod.time, "sleep", lambda s: sleeps.append(s))
    sleeps = []

    class Handle:
        def set(self, **kw):
            scenario.write_text(json.dumps({"authorized": key, **kw}))

        @property
        def calls(self):
            return json.loads(scenario.read_text()).get("calls", [])
        slept = sleeps
    return Handle()


def _session(tmp_path):
    s = RunPodSession(str(tmp_path / "out"))
    s.pod_id, s.ssh_host, s.ssh_port = "pod-1", "1.2.3.4", 22
    return s


class TestWaitForSsh:
    def test_agent_with_many_keys_does_not_break_us(self, tmp_path, fake_ssh, capsys, key):
        fake_ssh.set(agent_keys=12)
        # A bare ssh -- what the session used to run -- dies on this server.
        bare = subprocess.run(["ssh", "-p", "22", "root@1.2.3.4", "true"], capture_output=True, text=True)
        assert "Too many authentication failures" in bare.stderr

        s = _session(tmp_path)
        s.wait_for_ssh()

        argv = fake_ssh.calls[-1]
        assert argv[argv.index("-i") + 1] == key
        assert "IdentityAgent=none" in argv and "IdentitiesOnly=yes" in argv
        # ...and the log says which key was offered.
        assert _fingerprint(key + ".pub") in capsys.readouterr().out

    def test_sshd_not_ready_then_ready(self, tmp_path, fake_ssh):
        fake_ssh.set(refuse_first=3)
        s = _session(tmp_path)
        s.wait_for_ssh()
        assert len(fake_ssh.calls) == 4
        assert fake_ssh.slept == [5, 10, 15]   # backed off, no hammering

    def test_sshd_never_ready_gives_up_and_releases_the_pod(self, tmp_path, fake_ssh, monkeypatch):
        fake_ssh.set(refuse_first=10_000)
        s = _session(tmp_path)
        terminated = []
        monkeypatch.setattr(s, "terminate", lambda: terminated.append(s.pod_id))
        with pytest.raises(PodSetupError, match="did not accept connections"):
            s.wait_for_ssh()
        assert len(fake_ssh.calls) == runpod_pod.SSH_READY_ATTEMPTS
        assert terminated == ["pod-1"]

    def test_probe_timeout_counts_as_not_ready(self, tmp_path, monkeypatch):
        s = _session(tmp_path)
        outcomes = iter([subprocess.TimeoutExpired("ssh", 30), subprocess.CompletedProcess([], 0, "", "")])

        def ssh_cmd(cmd, timeout=60, input=None):
            r = next(outcomes)
            if isinstance(r, Exception):
                raise r
            return r
        monkeypatch.setattr(s, "ssh_cmd", ssh_cmd)
        monkeypatch.setattr(runpod_pod.time, "sleep", lambda _s: None)
        s.wait_for_ssh()


class TestBootRaceRejection:
    def _race(self, fake_ssh, tmp_path, key, rejects):
        # pod rejects the key for the first `rejects` probes, then accepts
        fake_ssh.set(authorized=key)
        scn = fake_ssh.calls  # noqa: F841
        s = _session(tmp_path)
        n = {"i": 0}
        real = s.ssh_cmd

        def ssh_cmd(cmd, timeout=60, input=None):
            n["i"] += 1
            if n["i"] <= rejects:
                return subprocess.CompletedProcess([], 255, "", "root@1.2.3.4: Permission denied (publickey).")
            return real(cmd, timeout=timeout, input=input)
        s.ssh_cmd = ssh_cmd
        return s

    def test_brief_boot_rejection_is_survived(self, tmp_path, fake_ssh, key):
        s = self._race(fake_ssh, tmp_path, key, rejects=3)
        s.wait_for_ssh()   # would have been fatal at 2 rejections

    def test_persistent_rejection_after_a_working_pod_is_retryable(self, tmp_path, fake_ssh, key, monkeypatch):
        s = self._race(fake_ssh, tmp_path, key, rejects=10_000)
        s._ssh_ever_worked = True
        monkeypatch.setattr(s, "terminate", lambda: None)
        with pytest.raises(PodSetupError, match="earlier pod"):
            s.wait_for_ssh()

    def test_persistent_rejection_on_first_pod_is_fatal(self, tmp_path, fake_ssh, key, monkeypatch):
        s = self._race(fake_ssh, tmp_path, key, rejects=10_000)
        monkeypatch.setattr(s, "terminate", lambda: None)
        with pytest.raises(PodSshAuthError):
            s.wait_for_ssh()


class TestWrongKeyFailsFast:
    @pytest.fixture
    def fleet(self, monkeypatch):
        fleet = _FleetRunpod()
        fleet.env_seen = []
        orig = fleet.create_pod

        def create_pod(name, **kw):
            fleet.env_seen.append(kw.get("env"))
            return orig(name, **kw)
        fleet.create_pod = create_pod
        monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
        monkeypatch.delenv("LOSSY_RUNPOD_SWEEP", raising=False)
        monkeypatch.setitem(sys.modules, "runpod", fleet)
        monkeypatch.setattr(RunPodSession, "_register_cleanup", lambda self: None)
        return fleet

    def test_pod_is_registered_with_the_key_we_will_offer(self, tmp_path, fake_ssh, fleet, key):
        fake_ssh.set()
        s = RunPodSession(str(tmp_path / "out"))
        s.create_pod()
        assert fleet.env_seen == [{"PUBLIC_KEY": pod_ssh.resolve_ssh_identity({pod_ssh.KEY_ENV: key}).public_key}]
        assert s.pod_id in fleet.live

    def test_rejected_key_is_not_retried_and_the_pod_is_terminated(self, tmp_path, fake_ssh, fleet, tmp_path_factory):
        other = _keygen(tmp_path_factory.mktemp("other") / "k")
        fake_ssh.set(authorized=other)             # the pod only accepts some other key
        s = RunPodSession(str(tmp_path / "out"))
        with pytest.raises(PodSshAuthError, match="rejected ssh key"):
            s.with_setup_retry(s.create_pod)
        assert fleet.created == 1                  # no re-provisioning onto a fresh pod
        assert fleet.live == {}                    # and the one pod is gone, via the normal path
        assert [e for e, _ in fleet.events] == ["create", "terminate"]
        assert len(fake_ssh.calls) >= runpod_pod.SSH_AUTH_REJECTIONS_FATAL
        assert sum(fake_ssh.slept) >= runpod_pod.SSH_AUTH_REJECTION_MIN_S

    def test_unusable_key_fails_before_any_pod_is_rented(self, tmp_path, fleet, monkeypatch):
        monkeypatch.setenv(pod_ssh.KEY_ENV, str(tmp_path / "missing"))
        s = RunPodSession(str(tmp_path / "out"))
        with pytest.raises(SshKeyError):
            s.create_pod()
        assert fleet.created == 0
