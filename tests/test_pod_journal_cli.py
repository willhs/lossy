"""`python -m pod_journal list|forget` and the exit-4 stall line (research 0026 review F5)."""
import os
import sys

import pytest

import pod_journal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))


def _journal(out, run_id="aaaa1111", pod="pod-1", keep_open=False):
    j = pod_journal.PodJournal(os.path.join(out, pod_journal.JOURNAL_DIRNAME), run_id, f"lossy-comfyui-{run_id}")
    j.set_pod(pod, f"lossy-comfyui-{run_id}", pod_journal.ABANDONED)
    if not keep_open:
        pod_journal._unlock(j._fd)   # dead session: lock released, files stay
        j._fd = None
    return j


def test_list_prints_claim_with_api_status(tmp_path, capsys, monkeypatch):
    _journal(str(tmp_path))
    monkeypatch.setattr(pod_journal, "_api_status", lambda pid: "not found")
    assert pod_journal.cmd_list(os.path.join(tmp_path, "runpod_journal"), status=lambda p: "not found") == 0
    out = capsys.readouterr().out
    assert "pod-1" in out and "abandoned" in out and "api=not found" in out and "dead session" in out


def test_forget_one_pod_removes_journal(tmp_path, capsys):
    _journal(str(tmp_path))
    assert pod_journal.main(["forget", str(tmp_path), "pod-1"]) == 0
    assert pod_journal.read_ownership(os.path.join(tmp_path, "runpod_journal")).ids == set()
    assert not os.listdir(os.path.join(tmp_path, "runpod_journal"))


def test_forget_keeps_other_pods(tmp_path):
    j = _journal(str(tmp_path))
    j2 = pod_journal.Snapshot.from_json({"run_id": "aaaa1111", "name": "n",
        "pods": {"pod-1": {"state": "abandoned"}, "pod-2": {"state": "owned"}}})
    pod_journal._write_atomic(j.json_path, j2.to_json())
    assert pod_journal.main(["forget", str(tmp_path), "pod-1"]) == 0
    assert pod_journal.read_ownership(os.path.join(tmp_path, "runpod_journal")).ids == {"pod-2"}


def test_forget_refuses_a_live_session(tmp_path, capsys):
    j = _journal(str(tmp_path), keep_open=True)
    try:
        assert pod_journal.main(["forget", str(tmp_path), "pod-1"]) == 1
        assert "still running" in capsys.readouterr().out
        assert pod_journal.read_ownership(os.path.join(tmp_path, "runpod_journal")).ids == {"pod-1"}
    finally:
        j.close()


def test_forget_unknown_pod_is_an_error(tmp_path):
    _journal(str(tmp_path))
    assert pod_journal.main(["forget", str(tmp_path), "nope"]) == 1


def test_stall_hint_names_the_cli(tmp_path):
    hint = pod_journal.stall_hint(os.path.join(tmp_path, "runpod_journal"))
    assert "python -m pod_journal forget" in hint and str(tmp_path) in hint


def test_exit_4_with_outstanding_claim_prints_loud_line(tmp_path, monkeypatch, capsys):
    import capped_run
    _journal(str(tmp_path))
    monkeypatch.setattr(capped_run, "balance_and_pods", lambda: (5.0, []))
    monkeypatch.setattr(capped_run, "terminate", lambda ids: None)
    monkeypatch.setattr(capped_run, "sweep", lambda scope, **k: [])
    monkeypatch.setattr(capped_run.time, "sleep", lambda s: None)
    monkeypatch.setattr(sys, "argv", [
        "capped_run", "--cap", "5", "--margin", "0.1", "--no-project", "--output-dir", str(tmp_path),
        "--poll", "3600", "--retries", "0", "--", sys.executable, "-c", "import sys; sys.exit(1)", str(tmp_path)])
    import signal
    saved = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        assert capped_run.main() == 4
    finally:
        for s, h in saved.items():
            signal.signal(s, h)
    out = capsys.readouterr().out
    assert "RUN STALLED" in out and "pod_journal forget" in out
