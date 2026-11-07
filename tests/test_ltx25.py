"""Tests for the runpod-ltx25 strategy (research 0025) and the pod helpers it uses."""

import shutil
import subprocess
from unittest.mock import MagicMock

import pytest

import config
from runpod_pod import PinnedModel, PodSetupError, RunPodSession, parse_comfyui_version
from strategies_video import RunPodLtx25Strategy, ltx25_frames


def _strategy():
    # Graph / frame maths only -- no pod, no session.
    return RunPodLtx25Strategy.__new__(RunPodLtx25Strategy)


def _inputs_ref_exist(wf):
    """Every ["node_id", slot] reference must point at a node in the graph."""
    for nid, node in wf.items():
        for key, val in node["inputs"].items():
            if isinstance(val, list) and len(val) == 2 and isinstance(val[0], str):
                assert val[0] in wf, f"node {nid}.{key} references missing node {val[0]}"


class TestFrameRounding:
    @pytest.mark.parametrize("target_s", [0.1, 1.0, 1.3, 2.0, 2.5, 3.33, 4.0, 4.8, 5.0, 12.0])
    def test_always_one_more_than_a_multiple_of_eight_and_in_range(self, target_s):
        n = ltx25_frames(target_s)
        assert n % 8 == 1
        assert 25 <= n <= 121

    @pytest.mark.parametrize("target_s,expected", [
        (0.5, 25),    # clamped up to the 1s minimum
        (2.0, 49),    # 50 raw -> 49
        (3.0, 73),    # 75 raw -> 73
        (4.0, 97),    # 100 raw -> 97
        (4.8, 121),   # 120 raw -> 121
        (9.0, 121),   # clamped to the ~4.8s maximum
    ])
    def test_known_values(self, target_s, expected):
        assert ltx25_frames(target_s) == expected

    def test_strategy_target_frames_uses_it(self):
        s = _strategy()
        for t in (0.7, 1.9, 2.6, 3.9, 4.7):
            assert s._target_frames(t) % 8 == 1

    def test_split_parts_are_all_valid_frame_counts(self):
        # A 17.7s shot splits into several parts; the remainder part too.
        s = _strategy()
        parts = s._target_durations(17.7)
        assert len(parts) > 1
        assert all(p % 8 == 1 for p in parts)

    def test_build_workflow_rejects_invalid_length(self):
        with pytest.raises(ValueError, match="% 8 == 1"):
            _strategy()._build_workflow("p", seed=1, length=100)


class TestWorkflow:
    def _wf(self, start_image=None, length=97):
        return _strategy()._build_workflow("a prompt", seed=7, length=length, start_image=start_image)

    def test_references_resolve(self):
        _inputs_ref_exist(self._wf())
        _inputs_ref_exist(self._wf(start_image="chain_0001_00.png"))

    def test_stage1_runs_at_half_resolution_and_decode_at_full_via_upsample(self):
        wf = self._wf()
        s = _strategy()
        assert wf["8"]["inputs"]["width"] == s.WIDTH // 2
        assert wf["8"]["inputs"]["height"] == s.HEIGHT // 2
        assert wf["21"]["class_type"] == "LTXVLatentUpsampler"
        # the refine pass consumes the upsampled latent, and decode consumes the refine pass
        assert wf["22"]["inputs"]["video_latent"] == ["21", 0]
        assert wf["25"]["inputs"]["latent_image"] == ["22", 0]
        assert wf["30"]["inputs"]["samples"] == ["26", 0]

    def test_both_dimensions_divisible_by_32_at_both_stages(self):
        s = _strategy()
        for dim in (s.WIDTH, s.HEIGHT, s.WIDTH // 2, s.HEIGHT // 2):
            assert dim % 32 == 0

    def test_uses_distilled_sigma_schedules(self):
        wf = self._wf()
        assert wf["14"]["inputs"]["sigmas"] == RunPodLtx25Strategy.STAGE1_SIGMAS
        assert wf["24"]["inputs"]["sigmas"] == RunPodLtx25Strategy.STAGE2_SIGMAS
        # 8 steps then 3 steps (n sigmas = steps + 1)
        assert len(RunPodLtx25Strategy.STAGE1_SIGMAS.split(",")) == 9
        assert len(RunPodLtx25Strategy.STAGE2_SIGMAS.split(",")) == 4

    def test_uses_conv_vae_and_int8_weights(self):
        wf = self._wf()
        assert "conv" in wf["3"]["inputs"]["vae_name"]
        assert "int8" in wf["1"]["inputs"]["unet_name"]

    def test_every_loaded_file_is_a_pinned_download(self):
        wf = self._wf()
        loaded = {
            wf["1"]["inputs"]["unet_name"], wf["2"]["inputs"]["clip_name"],
            wf["3"]["inputs"]["vae_name"], wf["4"]["inputs"]["vae_name"],
            wf["20"]["inputs"]["model_name"],
        }
        pinned = {m.dest.split("/", 1)[1] for m in RunPodLtx25Strategy.PINNED_MODELS}
        assert loaded == pinned

    def test_seed_and_length_flow_through(self):
        wf = self._wf(length=49)
        assert wf["13"]["inputs"]["noise_seed"] == 7
        assert wf["8"]["inputs"]["length"] == 49
        assert wf["9"]["inputs"]["frames_number"] == 49

    def test_keeps_native_audio_in_saved_clip(self):
        wf = self._wf()
        assert wf["31"]["class_type"] == "LTXVAudioVAEDecode"
        assert wf["32"]["inputs"]["audio"] == ["31", 0]
        assert wf["33"]["class_type"] == "SaveVideo"

    def test_t2v_has_no_image_nodes(self):
        wf = self._wf()
        assert wf["10"]["inputs"]["video_latent"] == ["8", 0]
        assert wf["7"]["inputs"]["positive"] == ["5", 0]
        assert not any(n["class_type"].startswith("LTXVImgToVideo") for n in wf.values())
        assert "LoadImage" not in {n["class_type"] for n in wf.values()}

    def test_i2v_routes_conditioning_and_latent_through_img_to_video(self):
        wf = self._wf(start_image="chain_0846_00.png")
        assert wf["40"]["inputs"]["image"] == "chain_0846_00.png"
        assert wf["41"]["class_type"] == "LTXVImgToVideo"
        # at stage-1 (half) resolution, matching the empty latent it replaces
        assert (wf["41"]["inputs"]["width"], wf["41"]["inputs"]["height"]) == (
            wf["8"]["inputs"]["width"], wf["8"]["inputs"]["height"])
        assert wf["7"]["inputs"]["positive"] == ["41", 0]
        assert wf["7"]["inputs"]["negative"] == ["41", 1]
        assert wf["10"]["inputs"]["video_latent"] == ["41", 2]

    def test_i2v_repins_start_frame_before_refine_pass(self):
        wf = self._wf(start_image="chain_0846_00.png")
        assert wf["42"]["class_type"] == "LTXVImgToVideoInplace"
        assert wf["42"]["inputs"]["latent"] == ["21", 0]
        # ComfyUI v0.39 rejects the node without an explicit bypass flag
        assert wf["42"]["inputs"]["bypass"] is False
        assert wf["22"]["inputs"]["video_latent"] == ["42", 0]

    def test_stage2_repin_can_be_disabled(self, monkeypatch):
        monkeypatch.setattr(RunPodLtx25Strategy, "REAPPLY_START_IMAGE_STAGE2", False)
        wf = self._wf(start_image="chain_0846_00.png")
        assert "42" not in wf
        assert wf["22"]["inputs"]["video_latent"] == ["21", 0]


class TestRegistration:
    def test_registered_in_config(self):
        spec = config.VIDEO_STRATEGIES["runpod-ltx25"]
        assert spec.cls is RunPodLtx25Strategy
        assert "runpod-ltx25" in config.VIDEO_STRATEGY_NAMES
        assert RunPodLtx25Strategy.name == "runpod-ltx25"

    def test_chains_split_parts_and_has_no_custom_node_install(self):
        assert RunPodLtx25Strategy._supports_i2v is True
        # LTX-2.5 uses only core nodes; the LTX-2.3 custom-node clone must not run
        assert RunPodLtx25Strategy._setup_pod_once is not RunPodLtx25Strategy.__mro__[1]._setup_pod_once

    def test_pinned_models_are_hash_pinned_and_revision_pinned(self):
        assert len(RunPodLtx25Strategy.MIRROR_REV) == 40
        for m in RunPodLtx25Strategy.PINNED_MODELS:
            assert len(m.sha256) == 64 and int(m.sha256, 16) >= 0
            assert RunPodLtx25Strategy.MIRROR_REV in m.url
            assert m.url.endswith(m.dest.split("/", 1)[1])

    def test_creates_pod_with_enough_disk(self, tmp_path):
        s = RunPodLtx25Strategy(output_dir=str(tmp_path))
        assert s._session.disk_gb == RunPodLtx25Strategy.DISK_GB >= 100


class TestPodSetup:
    def test_setup_upgrades_comfyui_downloads_pinned_then_restarts(self):
        s = _strategy()
        s._setup_done = False
        order = []
        sess = MagicMock()
        sess.ssh_host = "1.2.3.4"
        sess.ensure_comfyui_version.side_effect = lambda *a: order.append("version")
        sess.download_pinned_models.side_effect = lambda *a, **k: order.append("download")
        sess.restart_comfyui.side_effect = lambda: order.append("restart")
        s._session = sess
        import strategies_video
        orig_sleep = strategies_video.time.sleep
        strategies_video.time.sleep = lambda *_: None
        try:
            s._setup_pod_once()
        finally:
            strategies_video.time.sleep = orig_sleep
        assert order == ["version", "download", "restart"]
        sess.ensure_comfyui_version.assert_called_once_with((0, 32, 0), RunPodLtx25Strategy.COMFYUI_TAG)
        sess.download_pinned_models.assert_called_once()


class TestComfyVersion:
    def test_parse(self):
        assert parse_comfyui_version('__version__ = "0.30.0"\n') == (0, 30, 0)
        assert parse_comfyui_version("__version__ = 'v0.39.2'") == (0, 39, 2)
        assert parse_comfyui_version("nothing here") is None

    def _session(self, versions):
        """A session whose pod reports successive versions on `cat comfyui_version.py`."""
        s = RunPodSession("/tmp/x")
        s.ssh_host, s.ssh_port = "h", 1
        calls = []
        versions = list(versions)

        def ssh(cmd, timeout=60):
            calls.append(cmd)
            if "comfyui_version.py" in cmd:
                v = versions.pop(0) if len(versions) > 1 else versions[0]
                return subprocess.CompletedProcess(cmd, 0, f'__version__ = "{v}"\n', "")
            if "UPGRADE_OK" in cmd:
                return subprocess.CompletedProcess(cmd, 0, "UPGRADE_OK\n", "")
            return subprocess.CompletedProcess(cmd, 0, "python3\n", "")

        s.ssh_cmd = ssh
        return s, calls

    def test_new_enough_is_left_alone(self):
        s, calls = self._session(["0.39.2"])
        s.ensure_comfyui_version((0, 32, 0), "v0.39.2")
        assert not any("git checkout" in c for c in calls)

    def test_old_comfyui_is_checked_out_to_the_tag(self):
        s, calls = self._session(["0.30.0", "0.39.2"])
        s.ensure_comfyui_version((0, 32, 0), "v0.39.2")
        upgrade = [c for c in calls if "git checkout" in c]
        assert len(upgrade) == 1 and "v0.39.2" in upgrade[0] and "pip install" in upgrade[0]

    def test_upgrade_that_does_not_take_raises_retryable_error(self):
        s, _ = self._session(["0.30.0"])  # still 0.30.0 after "upgrade"
        with pytest.raises(PodSetupError):
            s.ensure_comfyui_version((0, 32, 0), "v0.39.2")


class TestPinnedDownload:
    MODEL = PinnedModel("vae/x.safetensors", "https://example.test/x", "a" * 64)

    def _session(self, exists, verified=True):
        s = RunPodSession("/tmp/x")
        s.ssh_host, s.ssh_port = "h", 1
        calls = []

        def ssh(cmd, timeout=60):
            calls.append(cmd)
            if "echo exists" in cmd:
                return subprocess.CompletedProcess(cmd, 0, "exists\n" if exists else "missing\n", "")
            if "sha256sum" in cmd:
                return subprocess.CompletedProcess(
                    cmd, 0 if verified else 1, "VERIFIED\n" if verified else "", "")
            return subprocess.CompletedProcess(cmd, 0, "", "")

        s.ssh_cmd = ssh
        return s, calls

    def test_downloads_to_part_file_and_verifies_before_moving_into_place(self):
        s, calls = self._session(exists=False)
        s.download_pinned_models([self.MODEL])
        dl = next(c for c in calls if "wget" in c)
        assert ".part" in dl and "a" * 64 in dl
        assert dl.index("sha256sum") < dl.index("mv ")

    def test_existing_file_is_not_redownloaded(self):
        s, calls = self._session(exists=True)
        s.download_pinned_models([self.MODEL])
        assert not any("wget" in c for c in calls)

    def test_hash_mismatch_raises_and_cleans_up_the_part_file(self):
        s, calls = self._session(exists=False, verified=False)
        with pytest.raises(PodSetupError, match="x.safetensors"):
            s.download_pinned_models([self.MODEL])
        assert any(c.startswith("rm -f") and ".part" in c for c in calls)

    def test_session_disk_defaults_to_module_value(self):
        assert RunPodSession("/tmp/x").disk_gb is None
        assert RunPodSession("/tmp/x", disk_gb=120).disk_gb == 120


class TestMp4OutputFromPod:
    """SaveVideo returns .mp4 already; saving it must not collide with the clip path.

    The first live smoke test lost every clip to this: raw_path == clip_path, so
    ffmpeg converted a file onto itself, failed, and the fallback deleted it.
    """

    @pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
    def test_mp4_output_is_saved_as_the_clip(self, tmp_path):
        src = tmp_path / "src.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x64:rate=25:duration=1",
             "-pix_fmt", "yuv420p", str(src)], check=True)
        s = _strategy()
        sess = MagicMock()
        sess.submit_workflow.return_value = {"outputs": {"33": {"images": [{"filename": "lossy_00001_.mp4"}]}}}
        sess.download_output.return_value = src.read_bytes()
        sess.pod_start_time = None
        s._session = sess
        clips = tmp_path / "clips"
        clips.mkdir()

        result = s._attempt_one_clip("p", str(clips), "0001.mp4", 25, 1)

        assert result is not None
        assert (clips / "0001.mp4").stat().st_size > 0
        assert sorted(p.name for p in clips.iterdir()) == ["0001.mp4"]  # no raw leftovers
