"""Research 0035: re-anchoring a keyframe shot on its keyframe later in the clip.

Graph-only: the strategy's RunPod session is a stand-in and nothing is submitted.
"""
from unittest.mock import MagicMock

import pytest

import decode
from strategies_video import RunPodLtx25KeyframeStrategy

CHARS = {"characters": [{"name": "han", "description": "A real man.", "shots": [1]}]}


def _make(tmp_path, one_stage=True, **kw):
    s = RunPodLtx25KeyframeStrategy(
        output_dir=str(tmp_path), character_shot_map=decode.build_character_shot_map(CHARS),
        characters_data=CHARS, **kw)
    s._session = MagicMock()
    s.one_stage = one_stage
    s._upload_image = lambda local: "kf_0001.png"
    return s


def _clip(s, shot, start="kf_0001.png", frames=49):
    s._keyframes = {1: "/local/kf_0001.png"}
    s._first_part_start_image(shot)
    return s._build_clip_workflow("p", seed=shot, frames=frames, start_image=start)


def _links_resolve(wf):
    for nid, node in wf.items():
        for key, val in node["inputs"].items():
            if isinstance(val, list) and len(val) == 2 and isinstance(val[0], str):
                assert val[0] in wf, f"node {nid}.{key} references missing node {val[0]}"


def _guides(wf):
    return [n["inputs"] for n in wf.values() if n["class_type"] == "LTXVAddGuide"]


class TestAnchor:
    def test_off_by_default_graph_is_the_plain_keyframe_graph(self, tmp_path):
        wf = _clip(_make(tmp_path), 1)
        assert not _guides(wf) and "69" not in wf

    @pytest.mark.parametrize("one_stage", [True, False])
    def test_end_anchor_pins_the_keyframe_as_the_last_frame_and_is_cropped(self, tmp_path, one_stage):
        wf = _clip(_make(tmp_path, one_stage=one_stage, anchor_end=0.7), 1)
        _links_resolve(wf)
        [g] = _guides(wf)
        assert (g["frame_idx"], g["strength"], g["image"]) == (-1, 0.7, ["60", 0])
        assert wf["60"]["inputs"]["image"] == "kf_0001.png"
        assert g["latent"] == ["41", 2]                      # after the start-frame conditioning
        assert wf["10"]["inputs"]["video_latent"] == ["61", 2]
        assert wf["11"]["inputs"]["positive"] == ["61", 0]
        assert wf["69"]["inputs"]["latent"] == ["16", 0]
        if one_stage:
            assert wf["30"]["inputs"]["samples"] == ["69", 2]
        else:
            assert wf["21"]["inputs"]["samples"] == ["69", 2]
            assert wf["25"]["inputs"]["guider"] == ["68", 0]

    def test_mid_and_end_chain_in_clip_order(self, tmp_path):
        wf = _clip(_make(tmp_path, anchor_end=0.5, anchor_mid=0.5), 1, frames=49)
        _links_resolve(wf)
        mid, end = _guides(wf)
        assert (mid["frame_idx"], end["frame_idx"]) == (24, -1)
        assert end["positive"] == ["61", 0] and end["latent"] == ["61", 2]
        assert wf["11"]["inputs"]["positive"] == ["62", 0]

    def test_text_shot_after_a_keyframe_shot_gets_no_anchor(self, tmp_path):
        s = _make(tmp_path, anchor_end=1.0)
        _clip(s, 1)
        wf = _clip(s, 2, start=None)                         # shot 2 has no keyframe
        assert not _guides(wf)

    def test_later_parts_keep_the_shots_anchor(self, tmp_path):
        s = _make(tmp_path, anchor_end=1.0)
        _clip(s, 1)
        wf = s._build_clip_workflow("p", seed=2, frames=49, start_image="chain_0001_p0.png")
        [g] = _guides(wf)
        assert wf["40"]["inputs"]["image"] == "chain_0001_p0.png"
        assert wf["60"]["inputs"]["image"] == "kf_0001.png"

    @pytest.mark.parametrize("bad", [-0.1, 1.5])
    def test_strength_out_of_range_is_rejected(self, tmp_path, bad):
        with pytest.raises(ValueError, match="anchor_end"):
            _make(tmp_path, anchor_end=bad)
