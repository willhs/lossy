"""Shared test fixtures."""

import pytest

import pod_ssh
import runpod_pod

FAKE_IDENTITY = pod_ssh.SshIdentity("/nonexistent/test_key", "ssh-ed25519 AAAATESTKEY", "SHA256:test")


@pytest.fixture(autouse=True)
def _no_real_pod_ssh(request, monkeypatch):
    """Keep tests off the developer's real ~/.ssh and off the network.

    Pod provisioning now resolves an ssh key and probes sshd after create. Tests
    of the lifecycle do not care about either, so both are stubbed. test_pod_ssh.py
    is the one module that exercises the real resolution and probing.
    """
    if request.module.__name__ == "test_pod_ssh":
        return
    monkeypatch.setattr(pod_ssh, "resolve_ssh_identity", lambda *a, **k: FAKE_IDENTITY)
    monkeypatch.setattr(runpod_pod.RunPodSession, "wait_for_ssh", lambda self: None)
