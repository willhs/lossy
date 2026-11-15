"""Research 0029 arm I: the LTX-2.5 Ingredients IC-LoRA (sheet, prompt, graph, HF gate).

No pod, no network, no spend: the HEAD request, the ssh session and the strategy's
RunPod session are stand-ins, and the graph is only built, never submitted.
"""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

import config
import decode
import ingredients
import runpod_pod
from ingredients import SheetPanel, compose_sheet, format_sheet_prompt, panel_labels
from runpod_pod import GatedModelError, PodSetupError, RunPodSession, check_gated_access
from strategies_video import RunPodLtx25IngredientsStrategy, RunPodLtx25Strategy

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRIAL = os.path.join(REPO, "docs", "research", "0029-character-reference-continuity", "trial")
GATE = "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Ingredients"
TOKEN = "hf_secret_token_value"


def _img(h=100, w=100, v=200):
    return np.full((h, w, 3), v, dtype=np.uint8)


def _panel(desc="A real man in his mid 30s.", **kw):
    return SheetPanel("Character", desc, kw.get("image", _img()))


class TestSheetPrompt:
    def test_one_panel_matches_the_cards_single_character_form(self):
        out = format_sheet_prompt([_panel("A man.")], "He walks.")
        assert out == ("### Reference Sheet Description **Top Row (Character):** A man. "
                       "### Target Description He walks.")

    def test_panels_are_named_by_position_left_to_right(self):
        assert panel_labels(2) == ("Top Row Left", "Top Row Right")
        out = format_sheet_prompt([_panel("A."), _panel("B."), _panel("C.")], "t")
        assert out.index("Top Row Left (Character):** A.") < out.index("Top Row Middle (Character):** B.") \
            < out.index("Top Row Right (Character):** C.") < out.index("### Target Description t")

    @pytest.mark.parametrize("n", [0, 4])
    def test_sheet_size_is_bounded(self, n):
        with pytest.raises(ValueError, match="1 to 3"):
            panel_labels(n)

    def test_empty_target_and_blank_description_are_rejected(self):
        with pytest.raises(ValueError, match="target"):
            format_sheet_prompt([_panel()], "  ")
        with pytest.raises(ValueError, match="description"):
            _panel("   ")

    def test_non_colour_image_is_rejected(self):
        with pytest.raises(ValueError, match="HxWx3"):
            SheetPanel("Character", "x", np.zeros((10, 10), dtype=np.uint8))


class TestComposeSheet:
    def test_canvas_has_the_clip_size_and_black_background(self):
        out = compose_sheet([_panel(image=_img(100, 100))], 1280, 704)
        assert out.shape == (704, 1280, 3)
        assert out[0, 0].tolist() == [0, 0, 0]

    def test_panels_sit_left_to_right_without_distortion(self):
        left, right = _img(1024, 1024, 50), _img(1024, 1024, 250)
        out = compose_sheet([_panel(image=left), _panel(image=right)], 1280, 704)
        assert out[352, 320, 0] == 50     # centre of the left cell
        assert out[352, 960, 0] == 250    # centre of the right cell
        lit = np.where(out[352, :640, 0] == 50)[0]
        assert lit.max() - lit.min() + 1 == 640   # a square panel stays square: 640 wide in a 640 cell
        assert out[0, 320, 0] == 0                # ... and 640 tall leaves black above and below

    def test_wide_panel_is_fitted_to_the_cell_width(self):
        out = compose_sheet([_panel(image=_img(100, 400, 90))], 1280, 704)
        rows = np.where(out[:, 640, 0] == 90)[0]
        assert rows.max() - rows.min() + 1 == 320   # 400x100 scaled to 1280x320


def _strategy():
    return RunPodLtx25IngredientsStrategy.__new__(RunPodLtx25IngredientsStrategy)


def _link(wf, nid, key):
    return wf[nid]["inputs"][key]


class TestGraph:
    def wf(self, length=49):
        return _strategy()._build_workflow("### Reference Sheet Description ...", seed=3, length=length,
                                           start_image="0010.png")

    def test_every_reference_points_at_a_node_in_the_graph(self):
        wf = self.wf()
        for nid, node in wf.items():
            for key, val in node["inputs"].items():
                if isinstance(val, list) and len(val) == 2 and isinstance(val[0], str):
                    assert val[0] in wf, f"node {nid}.{key} references missing node {val[0]}"

    def test_sheet_is_a_static_video_guide_of_the_clip_length(self):
        wf = self.wf(length=49)
        assert wf["52"]["inputs"]["image"] == "0010.png"
        assert wf["53"]["class_type"] == "RepeatImageBatch" and wf["53"]["inputs"]["amount"] == 49
        guide = wf["54"]
        assert guide["class_type"] == "LTXVAddGuide"
        assert guide["inputs"]["image"] == ["53", 0] and guide["inputs"]["frame_idx"] == 0
        assert guide["inputs"]["iclora_parameters"] == ["51", 0]
        assert wf["51"]["inputs"]["iclora_model"] == ["50", 0]

    def test_stage1_samples_the_guided_latent_with_the_lora_model(self):
        wf = self.wf()
        assert wf["50"]["class_type"] == "LoraLoaderModelOnly"
        assert wf["50"]["inputs"]["lora_name"] == RunPodLtx25IngredientsStrategy.IC_LORA
        assert wf["50"]["inputs"]["strength_model"] == 1.0
        assert wf["10"]["inputs"]["video_latent"] == ["54", 2]         # guide rides into the AV latent
        g1 = wf["11"]["inputs"]
        assert (g1["model"], g1["positive"], g1["negative"]) == (["50", 0], ["54", 0], ["54", 1])
        assert wf["15"]["inputs"]["guider"] == ["11", 0]

    def test_guide_is_cut_out_before_upsample_and_stage2_uses_the_plain_model(self):
        wf = self.wf()
        crop = wf["55"]
        assert crop["class_type"] == "LTXVCropGuides" and crop["inputs"]["latent"] == ["16", 0]
        assert wf["21"]["inputs"]["samples"] == ["55", 2]
        g2 = wf["56"]["inputs"]
        assert (g2["model"], g2["positive"], g2["negative"]) == (["1", 0], ["55", 0], ["55", 1])
        assert wf["25"]["inputs"]["guider"] == ["56", 0]

    def test_text_and_seed_reach_the_graph_and_audio_path_is_kept(self):
        wf = self.wf()
        assert wf["5"]["inputs"]["text"].startswith("### Reference Sheet Description")
        assert wf["13"]["inputs"]["noise_seed"] == 3
        assert wf["31"]["class_type"] == "LTXVAudioVAEDecode" and wf["33"]["class_type"] == "SaveVideo"

    def test_sheet_is_required_and_frame_rule_still_applies(self):
        with pytest.raises(ValueError, match="reference sheet"):
            _strategy()._build_workflow("p", seed=1, length=49)
        with pytest.raises(ValueError, match="% 8 == 1"):
            _strategy()._build_workflow("p", seed=1, length=50, start_image="a.png")

    def test_base_strategy_graph_is_untouched(self):
        wf = RunPodLtx25Strategy.__new__(RunPodLtx25Strategy)._build_workflow("p", seed=1, length=49)
        assert not any(n["class_type"] in ("LTXVAddGuide", "LoraLoaderModelOnly") for n in wf.values())

    def test_normal_decode_path_is_refused(self):
        with pytest.raises(SystemExit, match="--ab-manifest"):
            _strategy().generate(1, {}, "/tmp")


class TestPinnedAdapter:
    def test_adapter_is_pinned_by_revision_and_hash_and_gated(self):
        m = next(m for m in RunPodLtx25IngredientsStrategy.PINNED_MODELS if m.dest.startswith("loras/"))
        assert m.dest == "loras/ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors"
        assert RunPodLtx25IngredientsStrategy.IC_LORA_REV in m.url and "/resolve/main/" not in m.url
        assert len(m.sha256) == 64 and m.gate_url == GATE

    def test_base_models_are_kept_and_ungated(self):
        base = RunPodLtx25Strategy.PINNED_MODELS
        assert RunPodLtx25IngredientsStrategy.PINNED_MODELS[:len(base)] == base
        assert not any(m.gate_url for m in base)

    def test_registered_as_ab_only_and_production_defaults_unchanged(self):
        spec = config.VIDEO_STRATEGIES["runpod-ltx25-ingredients"]
        assert spec.cls is RunPodLtx25IngredientsStrategy and spec.ab_manifest_only
        assert not any(s.ab_manifest_only for n, s in config.VIDEO_STRATEGIES.items()
                       if n != "runpod-ltx25-ingredients")


MODEL = RunPodLtx25IngredientsStrategy.PINNED_MODELS[-1]


def _head(status=None, exc=None):
    calls = []

    def head(url, headers=None, **kw):
        calls.append((url, headers))
        if exc:
            raise exc
        return SimpleNamespace(status_code=status)
    head.calls = calls
    return head


class TestGateCheck:
    def test_accepted_gate_passes_and_sends_the_token_as_bearer(self):
        head = _head(302)
        check_gated_access([MODEL], TOKEN, head=head)
        assert head.calls == [(MODEL.url, {"Authorization": f"Bearer {TOKEN}"})]

    @pytest.mark.parametrize("status,phrase", [(401, "rejected HF_TOKEN"), (403, "gate is not accepted")])
    def test_refused_gate_says_where_to_click_and_to_set_hf_token(self, status, phrase):
        with pytest.raises(GatedModelError) as e:
            check_gated_access([MODEL], TOKEN, head=_head(status))
        msg = str(e.value)
        assert phrase in msg and f"Accept the gate at {GATE}" in msg and "HF_TOKEN" in msg
        assert TOKEN not in msg

    def test_missing_token_fails_without_any_request(self):
        head = _head(200)
        with pytest.raises(GatedModelError, match=f"HF_TOKEN is not set.*Accept the gate at {GATE}"):
            check_gated_access([MODEL], None, head=head)
        assert head.calls == []

    def test_unreachable_hf_is_not_read_as_ok(self):
        import httpx
        with pytest.raises(GatedModelError, match="could not reach.*no pod was started"):
            check_gated_access([MODEL], TOKEN, head=_head(exc=httpx.ConnectError("down")))

    def test_ungated_models_are_not_checked(self):
        head = _head(200)
        check_gated_access(RunPodLtx25Strategy.PINNED_MODELS, None, head=head)
        assert head.calls == []

    def test_strategy_init_fails_before_a_pod_session_exists(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        made = []
        monkeypatch.setattr(runpod_pod, "RunPodSession", lambda *a, **k: made.append(1))
        with pytest.raises(GatedModelError, match="HF_TOKEN"):
            RunPodLtx25IngredientsStrategy(output_dir="/tmp/x")
        assert made == []


class TestGatedDownload:
    def _session(self, rc=0, terminated=None):
        s = RunPodSession("/tmp/x")
        s.ssh_host, s.ssh_port = "h", 1
        s.terminate = lambda: terminated.append(1)
        calls = []

        def ssh(cmd, timeout=60, input=None):
            calls.append((cmd, input))
            if "echo exists" in cmd:
                return subprocess.CompletedProcess(cmd, 0, "missing\n", "")
            if "sha256sum" in cmd:
                return subprocess.CompletedProcess(cmd, rc, "VERIFIED\n" if rc == 0 else "", "")
            return subprocess.CompletedProcess(cmd, 0, "", "")

        s.ssh_cmd = ssh
        return s, calls

    def test_token_goes_over_stdin_never_into_the_command_line(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", TOKEN)
        s, calls = self._session()
        s.download_pinned_models([MODEL])
        cmd, stdin = next((c, i) for c, i in calls if "wget" in c)
        assert TOKEN not in cmd and stdin == f"{TOKEN}\n"
        assert "read -r HF_TOKEN" in cmd and 'Authorization: Bearer $HF_TOKEN' in cmd
        assert all(TOKEN not in c for c, _ in calls)

    def test_wget_auth_failure_is_a_gate_error_and_the_pod_is_terminated(self, monkeypatch, capsys):
        monkeypatch.setenv("HF_TOKEN", TOKEN)
        killed = []
        s, calls = self._session(rc=6, terminated=killed)
        with pytest.raises(GatedModelError, match=f"Accept the gate at {GATE}") as e:
            s.download_pinned_models([MODEL])
        assert not isinstance(e.value, PodSetupError)   # with_setup_retry must not re-provision for it
        assert killed == [1]
        out = capsys.readouterr()
        assert TOKEN not in out.out + out.err + str(e.value)

    def test_no_token_on_the_pod_path_fails_clean_without_downloading(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        killed = []
        s, calls = self._session(terminated=killed)
        with pytest.raises(GatedModelError, match="HF_TOKEN is not set"):
            s.download_pinned_models([MODEL])
        assert not any("wget" in c for c, _ in calls) and killed == [1]

    def test_other_download_failures_stay_retryable(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", TOKEN)
        s, _ = self._session(rc=1, terminated=[])
        with pytest.raises(PodSetupError):
            s.download_pinned_models([MODEL])

    def test_gate_error_is_not_swallowed_by_setup_retry(self, monkeypatch):
        s = RunPodSession("/tmp/x")
        s.terminate = lambda: pytest.fail("retry wrapper must not run for a gate error")

        def setup():
            raise GatedModelError("gated")
        with pytest.raises(GatedModelError):
            s.with_setup_retry(setup, attempts=3)


def _trial_script():
    spec = importlib.util.spec_from_file_location("make_ingredients_arm", os.path.join(TRIAL, "make_ingredients_arm.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestTrialArm:
    """Arm I built from the committed trial plan and manifest, with stand-in sheets."""

    @pytest.fixture
    def run_dir(self, tmp_path):
        for f in ("trial_plan.json", "trial_manifest.json"):
            shutil.copy(os.path.join(TRIAL, f), tmp_path / f)
        (tmp_path / "sheets").mkdir()
        for i, name in enumerate(json.load(open(tmp_path / "trial_plan.json"))["identity"]):
            cv2.imwrite(str(tmp_path / "sheets" / f"{name}.png"), _img(1024, 1024, 40 + 50 * i))
        return str(tmp_path)

    def test_arm_i_matches_arm_t_shot_for_shot(self, run_dir, monkeypatch):
        monkeypatch.chdir(REPO)
        m = _trial_script().build(run_dir)
        t = [j for j in json.load(open(os.path.join(run_dir, "trial_manifest.json")))["jobs"] if j["arm"] == "T"]
        assert [(j["shot"], j["seed"], j["frames"]) for j in m["jobs"]] == [(j["shot"], j["seed"], j["frames"]) for j in t]
        assert all(j["arm"] == "I" for j in m["jobs"])
        for ij, tj in zip(m["jobs"], t):
            assert ij["prompt"].startswith("### Reference Sheet Description **Top Row")
            assert ij["prompt"].endswith("### Target Description " + tj["prompt"])

    def test_pair_shots_get_two_panels_and_sheets_have_the_frame_size(self, run_dir, monkeypatch):
        monkeypatch.chdir(REPO)
        m = _trial_script().build(run_dir)
        plan = json.load(open(os.path.join(run_dir, "trial_plan.json")))
        two = {s["shot"] for s in plan["shots"] if len(s["characters"]) == 2}
        assert two
        for j in m["jobs"]:
            assert ("Top Row Left (Character)" in j["prompt"]) == (j["shot"] in two)
            assert cv2.imread(os.path.join(run_dir, j["start_image"])).shape == (704, 1280, 3)

    def test_the_runner_accepts_the_manifest_and_dry_run_builds_every_graph(self, run_dir, monkeypatch, capsys):
        monkeypatch.chdir(REPO)
        m = _trial_script().build(run_dir)
        path = os.path.join(run_dir, "trial_manifest_I.json")
        json.dump(m, open(path, "w"))
        monkeypatch.setattr(decode, "check_gated_access", lambda *a, **k: None)
        decode.dry_run_ab(RunPodLtx25IngredientsStrategy, decode.load_ab_jobs(path))
        out = capsys.readouterr().out
        assert f"Dry run OK: {len(m['jobs'])} jobs" in out and "Gate check OK" in out

    def test_missing_sheets_name_the_command_that_makes_them(self, tmp_path, monkeypatch):
        monkeypatch.chdir(REPO)
        for f in ("trial_plan.json", "trial_manifest.json"):
            shutil.copy(os.path.join(TRIAL, f), tmp_path / f)
        with pytest.raises(SystemExit, match="make_refs.py RUN_DIR --sheets-only"):
            _trial_script().build(str(tmp_path))


class TestRunnerSelection:
    def _argv(self, monkeypatch, tmp_path, *extra):
        monkeypatch.setattr(sys, "argv", ["decode.py", str(tmp_path), *extra])
        monkeypatch.setattr(decode, "load_env", lambda: None)

    def test_dry_run_reports_a_failed_gate_without_failing_the_graph_check(self, tmp_path, monkeypatch, capsys):
        (tmp_path / "s.png").write_bytes(b"x")
        mf = tmp_path / "m.json"
        mf.write_text(json.dumps({"jobs": [{"shot": 1, "arm": "I", "seed": 1, "frames": 25, "prompt": "p",
                                            "start_image": "s.png"}]}))

        def refuse(*a, **k):
            raise GatedModelError(f"nope. Accept the gate at {GATE}")
        monkeypatch.setattr(decode, "check_gated_access", refuse)
        self._argv(monkeypatch, tmp_path, "--strategy", "runpod-ltx25-ingredients", "--ab-manifest", str(mf), "--dry-run")
        decode.main()
        out = capsys.readouterr().out
        assert "Dry run OK" in out and "Gate check FAILED" in out and GATE in out

    def test_real_run_with_a_closed_gate_exits_with_the_message_before_any_pod(self, tmp_path, monkeypatch):
        (tmp_path / "s.png").write_bytes(b"x")
        mf = tmp_path / "m.json"
        mf.write_text(json.dumps({"jobs": [{"shot": 1, "arm": "I", "seed": 1, "frames": 25, "prompt": "p",
                                            "start_image": "s.png"}]}))
        monkeypatch.delenv("HF_TOKEN", raising=False)
        made = []
        monkeypatch.setattr(runpod_pod, "RunPodSession", lambda *a, **k: made.append(1))
        self._argv(monkeypatch, tmp_path, "--strategy", "runpod-ltx25-ingredients", "--ab-manifest", str(mf))
        with pytest.raises(SystemExit) as e:
            decode.main()
        assert f"Accept the gate at {GATE}" in str(e.value) and "HF_TOKEN" in str(e.value)
        assert made == []

    def test_strategy_without_a_manifest_is_refused(self, tmp_path, monkeypatch):
        self._argv(monkeypatch, tmp_path, "--strategy", "runpod-ltx25-ingredients")
        with pytest.raises(SystemExit, match="only runs with --ab-manifest"):
            decode.main()

    def test_ab_manifest_still_rejects_non_ltx_strategies(self, tmp_path, monkeypatch):
        (tmp_path / "m.json").write_text(json.dumps({"jobs": [{"shot": 1, "arm": "T", "seed": 1, "frames": 25, "prompt": "p"}]}))
        self._argv(monkeypatch, tmp_path, "--strategy", "fal-seedance", "--ab-manifest", str(tmp_path / "m.json"))
        with pytest.raises(SystemExit, match="LTX strategies only"):
            decode.main()


def test_module_has_no_pod_or_network_imports():
    src = open(ingredients.__file__).read()
    assert "runpod" not in src and "httpx" not in src
