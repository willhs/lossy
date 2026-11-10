"""Style prefix (research 0028 fix 1) and the A/B manifest runner.

No pod, no network, no spend: the strategy is a stand-in and the one test that
drives tools/capped_run.py patches its RunPod balance calls.
"""

import json
import os
import subprocess
import sys
from unittest.mock import MagicMock

import pytest

import decode
import manifest
from clip_types import ClipResult
from strategies_video import (CharacterIdentityMixin, GenerationStrategy,
                              RunPodLtx25Strategy)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AB_MANIFEST = os.path.join(REPO, "docs", "research", "0028-style-drift", "ab_manifest.json")

PREFIX = "Live-action 1990s sitcom footage of real actors, photographed on 35mm film."
CHARS = {"characters": [
    {"name": "kramer", "display_name": "Cosmo Kramer", "shots": [1],
     "description": "A real man in his late 30s, tall and slim."},
]}


def _mixin(style_prefix=None):
    class _S(CharacterIdentityMixin, GenerationStrategy):
        def format_prompt(self, entry):
            return super().format_prompt(entry)

    s = _S()
    s._init_character_identity({1: ["kramer"]}, CHARS, style_prefix)
    return s


ENTRY = {"index": 1, "description": {"action": "He opens the door."}}
EMPTY = {"index": 2, "description": {"action": "An empty hallway."}}


class TestStylePrefix:
    def test_prefix_precedes_identity_block(self):
        out = _mixin(PREFIX).format_prompt(ENTRY)
        assert out.startswith(PREFIX + " A real man in his late 30s")
        assert out.index(PREFIX) < out.index("A real man") < out.index("He opens the door.")

    def test_prefix_is_on_shots_without_characters(self):
        assert _mixin(PREFIX).format_prompt(EMPTY).startswith(PREFIX + " ")

    def test_no_prefix_leaves_prompt_unchanged(self):
        assert _mixin(None).format_prompt(ENTRY) == _mixin("").format_prompt(ENTRY)
        assert not _mixin(None).format_prompt(ENTRY).startswith("Live-action")

    def test_same_string_on_every_shot(self):
        s = _mixin(PREFIX)
        assert s.format_prompt(ENTRY)[:len(PREFIX)] == s.format_prompt(EMPTY)[:len(PREFIX)]

    def test_ltx_strategy_accepts_the_prefix(self):
        s = object.__new__(RunPodLtx25Strategy)
        s._init_character_identity({1: ["kramer"]}, CHARS, PREFIX)
        assert s.format_prompt(ENTRY).startswith(PREFIX)


class TestStylePrefixInShotsJson:
    def _write(self, d, **extra):
        manifest.save_shots(str(d), [], [])
        raw = json.load(open(manifest.shots_path(str(d))))
        raw.update(extra)
        json.dump(raw, open(manifest.shots_path(str(d)), "w"))

    def test_loads_prefix(self, tmp_path):
        self._write(tmp_path, style_prefix=f"  {PREFIX} ")
        assert manifest.load_style_prefix(str(tmp_path)) == PREFIX

    def test_missing_file_or_field_is_empty(self, tmp_path):
        assert manifest.load_style_prefix(str(tmp_path)) == ""
        self._write(tmp_path)
        assert manifest.load_style_prefix(str(tmp_path)) == ""

    def test_reencode_keeps_a_hand_set_prefix(self, tmp_path):
        self._write(tmp_path, style_prefix=PREFIX)
        manifest.save_shots(str(tmp_path), [{"index": 0}], [])
        assert manifest.load_style_prefix(str(tmp_path)) == PREFIX

    def test_decode_factory_prefers_flag_over_shots_json(self, tmp_path, monkeypatch):
        self._write(tmp_path, style_prefix="from file")
        seen = {}

        class Fake:
            def __init__(self, **kw):
                seen.update(kw)

        for flag, want in ((None, "from file"), ("from flag", "from flag")):
            args = MagicMock(output_dir=str(tmp_path), keep_pod=False, style_prefix=flag)
            decode._create_ltx2_strategy(args, Fake)
            assert seen["style_prefix"] == want


class TestIdentitySpec:
    def test_asks_for_photographic_real_actor(self):
        import encode
        spec = encode.IDENTITY_DESCRIPTION_SPEC
        assert "real actor" in spec and "photographer" in spec
        assert "natural skin texture" in spec

    def test_names_the_caricature_cues_to_avoid(self):
        import encode
        spec = encode.IDENTITY_DESCRIPTION_SPEC
        for cue in ("very", "angular", "lanky", "mop", "hunched"):
            assert f"'{cue}'" in spec

    def test_still_shared_by_stage3_prompts_and_keeps_droid_rule(self):
        import encode
        assert encode.IDENTITY_DESCRIPTION_SPEC in encode.STAGE3_SYSTEM_PROMPT
        assert "mechanical robot" in encode.IDENTITY_DESCRIPTION_SPEC


# ---------------------------------------------------------------------------
# A/B manifest runner
# ---------------------------------------------------------------------------

def _jobs():
    return decode.load_ab_jobs(AB_MANIFEST)


class TestAbManifestLoad:
    def test_real_manifest_loads_21_jobs(self):
        jobs = _jobs()
        assert len(jobs) == 21
        assert {j["arm"] for j in jobs} == {
            "control_rerender", "A_prefix", "B_photo_block", "C_prefix_and_block"}

    def test_rejects_missing_keys_duplicates_and_empty(self, tmp_path):
        good = {"shot": 1, "arm": "A", "seed": 1, "frames": 25, "prompt": "p"}
        for jobs in ([], [{"shot": 1}], [good, dict(good)]):
            p = tmp_path / "m.json"
            p.write_text(json.dumps({"jobs": jobs}))
            with pytest.raises(ValueError):
                decode.load_ab_jobs(str(p))

    def test_dry_run_builds_every_workflow_and_catches_bad_frames(self, capsys):
        decode.dry_run_ab(RunPodLtx25Strategy, _jobs())
        assert "Dry run OK: 21 jobs" in capsys.readouterr().out
        bad = [{"shot": 1, "arm": "A", "seed": 1, "frames": 26, "prompt": "p"}]
        with pytest.raises(ValueError):
            decode.dry_run_ab(RunPodLtx25Strategy, bad)


class _FakeStrategy:
    name = "runpod-ltx25"

    def __init__(self, fail=()):
        self.calls, self.fail, self.pods, self.clean = [], set(fail), 0, False

    def _ensure_pod(self):
        self.pods += 1

    def mark_clean_exit(self):
        self.clean = True

    def _generate_one_clip(self, prompt, clips_dir, clip_name, frames, seed, start_image=None):
        self.calls.append((prompt, clips_dir, clip_name, frames, seed))
        if (os.path.basename(clips_dir), clip_name) in self.fail:
            return None
        path = os.path.join(clips_dir, clip_name)
        open(path, "wb").write(b"x")
        return ClipResult(path=path, actual_duration_s=1.0, cost=0.01)


class TestRunAbManifest:
    def _args(self, d):
        return MagicMock(output_dir=str(d))

    def test_renders_every_job_verbatim_into_per_arm_dirs(self, tmp_path):
        s, jobs = _FakeStrategy(), _jobs()
        decode.run_ab_manifest(self._args(tmp_path), s, jobs)
        assert len(s.calls) == 21 and s.clean
        for job, (prompt, clips_dir, name, frames, seed) in zip(jobs, s.calls):
            assert prompt == job["prompt"]                   # nothing rewrote it
            assert (frames, seed) == (job["frames"], job["seed"])
            assert clips_dir == manifest.ab_clips_dir(str(tmp_path), "runpod-ltx25", job["arm"])
            assert name == f"{job['shot']:04d}.mp4"
        # Arms share shot indices, so the same filename exists in several dirs.
        assert os.path.exists(tmp_path / "clips/runpod-ltx25-ab/A_prefix/0079.mp4")
        assert os.path.exists(tmp_path / "clips/runpod-ltx25-ab/B_photo_block/0079.mp4")
        progress = json.load(open(manifest.ab_progress_path(str(tmp_path), "runpod-ltx25")))
        assert len(progress["completed"]) == 21 and progress["failed"] == []

    def test_resume_skips_completed_jobs(self, tmp_path):
        jobs = _jobs()
        first = _FakeStrategy(fail={("C_prefix_and_block", "0079.mp4")})
        with pytest.raises(SystemExit) as e:
            decode.run_ab_manifest(self._args(tmp_path), first, jobs)
        assert e.value.code == 1
        progress = json.load(open(manifest.ab_progress_path(str(tmp_path), "runpod-ltx25")))
        assert progress["failed"] == ["C_prefix_and_block/79"]
        assert len(progress["completed"]) == 20

        second = _FakeStrategy()
        decode.run_ab_manifest(self._args(tmp_path), second, jobs)
        assert [c[2] for c in second.calls] == ["0079.mp4"]   # only the failed job reruns
        assert second.calls[0][1].endswith("C_prefix_and_block")


class TestUnderCappedRun:
    """The A/B run as it will be launched: decode as the child of capped_run.

    capped_run's RunPod balance/pod calls are patched out, and the child is
    decode --dry-run, which builds every workflow but never creates a pod.
    """

    def test_capped_run_supervises_a_dry_run_and_sees_progress_format(self, tmp_path, monkeypatch, capsys):
        sys.path.insert(0, os.path.join(REPO, "tools"))
        import capped_run

        calls = []
        monkeypatch.setattr(capped_run, "balance_and_pods", lambda: (5.0, []))
        monkeypatch.setattr(capped_run, "terminate", lambda ids: calls.append(ids))
        real_sleep = capped_run.time.sleep
        monkeypatch.setattr(capped_run.time, "sleep", lambda s: None if s <= 30 else real_sleep(s))
        monkeypatch.setattr(sys, "argv", [
            "capped_run", "--cap", "0.50", "--margin", "0.02", "--no-project", "--poll", "3600",
            "--progress", manifest.ab_progress_path(str(tmp_path), "runpod-ltx25"),
            "--total-shots", "21", "--",
            sys.executable, os.path.join(REPO, "decode.py"), str(tmp_path),
            "--strategy", "runpod-ltx25", "--ab-manifest", AB_MANIFEST, "--dry-run"])
        assert capped_run.main() == 0
        assert "attempt 1" in capsys.readouterr().out
        # decode's progress file is what capped_run counts: completed is a list.
        p = manifest.ab_progress_path(str(tmp_path), "runpod-ltx25")
        json.dump({"completed": ["A/1", "A/2"]}, open(p, "w"))
        assert capped_run.done_shots(p) == 2

    def test_dry_run_never_builds_a_strategy(self, tmp_path, monkeypatch):
        def boom(*a, **k):
            raise AssertionError("dry run must not create a RunPod session")

        monkeypatch.setattr(RunPodLtx25Strategy, "__init__", boom)
        monkeypatch.setattr(sys, "argv", [
            "decode.py", str(tmp_path), "--strategy", "runpod-ltx25",
            "--ab-manifest", AB_MANIFEST, "--dry-run"])
        decode.main()
