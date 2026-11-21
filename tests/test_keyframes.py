"""Research 0029 arm K as a strategy: keyframe-first LTX-2.5 with the two sheet fixes.

No pod, no network, no spend: the image editor, the face embedder and the pod are
stand-ins. fal_client/httpx are replaced by in-memory fakes where the fal generator
itself is under test.
"""

import json
import os
import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

import config
import decode
import keyframes
import manifest
from clip_types import ClipResult
from keyframes import (DiskCache, FalKleinGenerator, PodKleinGenerator, klein_graph, Fallback, KeyframeBudgetExceeded, KeyframeError,
                       KeyframeJob, KeyframeSetupError, OpenCVFaceEmbedder, SHEET_ATTEMPTS, ShotRoute,
                       keyframe_prompt, make_keyframes, make_sheets, plan_route, sheet_prompt, sheet_seed)
from strategies_video import (RunPodLtx25KeyframeStrategy, RunPodLtx25Strategy, RunPodWanStrategy,
                              _PodComfyRunner)

JERRY = "A real man in his mid 30s with short dark wavy hair and a lean build."
ELAINE = "A real woman in her early 30s with dark, voluminous curly hair."
KRAMER = "A real man in his late 30s, tall and slim, with thick dark curly hair."
STYLE = "Live-action 1990s sitcom footage of real actors, photographed on 35mm film."


def _chars():
    def c(name, desc, shots, **kw):
        return {"name": name, "display_name": name.title(), "description": desc, "shots": shots, **kw}
    return {"characters": [
        c("jerry", JERRY, [1, 2, 3, 6]),
        c("elaine", ELAINE, [2, 7]),
        c("kramer", KRAMER, [4]),
        c("c3po", "A gold humanoid droid.", [5, 6], keep_name=True),
        c("nodesc", "", [8]),
    ]}


def _shot(i, dur=2.0, **desc):
    d = {"subjects": f"Jerry in a blazer, shot {i}", "shot_type": "medium", "camera": "static"}
    d.update(desc)
    return {"index": i, "start_s": float(i), "end_s": i + dur, "duration_s": dur, "description": d}


SHOTS = [_shot(1), _shot(2), _shot(3), _shot(4), _shot(5), _shot(6), _shot(7), _shot(8), _shot(9)]
# 1 jerry | 2 jerry+elaine | 3 jerry | 4 kramer | 5 c3po only | 6 jerry+c3po | 7 elaine
# 8 character with no description | 9 nobody


def _make(tmp_path, generator=None, embedder=None, **kw):
    s = RunPodLtx25KeyframeStrategy(
        output_dir=str(tmp_path), character_shot_map=decode.build_character_shot_map(_chars()),
        characters_data=_chars(), style_prefix=STYLE, generator=generator, embedder=embedder, **kw)
    s._session = MagicMock()
    return s


class FakeEmbedder:
    """png bytes -> a face vector chosen by the test: b"face:<key>" looks up VECS[key]."""

    def __init__(self, vecs):
        self.vecs = {k: np.asarray(v, float) / np.linalg.norm(v) for k, v in vecs.items()}
        self.ok = True

    def preflight(self):
        if not self.ok:
            raise KeyframeSetupError("no models")

    def faces(self, png):
        key = png.decode().split(":", 1)[1]
        return [self.vecs[key]] if key in self.vecs else []


class FakeGenerator:
    id = "fake"

    def __init__(self, sheet_draws=None, fail_keyframes=(), budget_after=None):
        self.sheet_draws = dict(sheet_draws or {})   # description-keyword -> [png, png, ...]
        self.fail_keyframes = set(fail_keyframes)
        self.budget_after = budget_after
        self.sheet_calls, self.keyframe_calls = [], []
        self.ok = True

    def preflight(self):
        if not self.ok:
            raise KeyframeSetupError("FAL_KEY is not set")

    def make_sheet(self, prompt, seed, size):
        self.sheet_calls.append((prompt, seed))
        for kw, draws in self.sheet_draws.items():
            if kw in prompt:
                png = draws[min(sum(kw in p for p, _ in self.sheet_calls) - 1, len(draws) - 1)]
                if isinstance(png, Exception):
                    raise png
                return png
        who = "elaine" if "woman" in prompt else "kramer" if "tall and slim" in prompt else "jerry"
        return b"face:" + who.encode()

    def make_keyframe(self, prompt, sheets, seed, size):
        self.keyframe_calls.append((prompt, list(sheets), seed, size))
        if self.budget_after is not None and len(self.keyframe_calls) > self.budget_after:
            raise KeyframeBudgetExceeded("cap")
        if seed in self.fail_keyframes:
            raise KeyframeError("editor said no")
        return b"\x89PNG-keyframe-%d" % seed


DISTINCT = {"jerry": [1, 0, 0], "elaine": [0, 1, 0], "kramer": [0, 0, 1]}


def _embedder(extra=None):
    return FakeEmbedder({**DISTINCT, **(extra or {})})


# ----------------------------------------------------------------- routing

class TestRouting:
    def route(self, names, chars=None):
        chars = {c["name"]: c for c in (chars or _chars())["characters"]}
        return plan_route(1, names, chars, lambda n: n[:2])

    def test_a_person_gets_a_keyframe_from_their_own_sheet(self):
        r = self.route(["jerry"])
        assert r.keyframe and r.characters == ("jerry",) and r.fallback is None

    def test_two_people_both_get_sheets(self):
        assert self.route(["jerry", "elaine"]).characters == ("jerry", "elaine")

    def test_only_the_capped_people_are_shown_to_the_editor(self):
        chars = _chars()
        chars["characters"].append({"name": "george", "description": "A man.", "shots": [1]})
        r = self.route(["jerry", "elaine", "george"], chars)
        assert r.characters == ("jerry", "elaine")

    def test_keep_name_design_character_falls_back_to_text(self):
        r = self.route(["c3po"])
        assert not r.keyframe and r.fallback is Fallback.DESIGN_CHARACTER and r.characters == ()

    def test_a_design_character_anywhere_in_the_shot_sends_it_to_text(self):
        assert self.route(["jerry", "c3po"]).fallback is Fallback.DESIGN_CHARACTER

    def test_no_character_falls_back(self):
        assert self.route([]).fallback is Fallback.NO_CHARACTERS

    def test_person_without_a_description_falls_back(self):
        assert self.route(["nodesc"]).fallback is Fallback.NO_DESCRIPTION

    def test_a_route_must_be_one_or_the_other(self):
        with pytest.raises(ValueError):
            ShotRoute(1, (), None)
        with pytest.raises(ValueError):
            ShotRoute(1, ("jerry",), Fallback.NO_CHARACTERS)

    def test_strategy_plan_matches_its_own_shot_map(self, tmp_path):
        routes = {r.shot: r for r in _make(tmp_path).plan(SHOTS)}
        assert [s for s, r in routes.items() if r.keyframe] == [1, 2, 3, 4, 7]
        assert routes[5].fallback is Fallback.DESIGN_CHARACTER
        assert routes[6].fallback is Fallback.DESIGN_CHARACTER
        assert routes[8].fallback is Fallback.NO_DESCRIPTION
        assert routes[9].fallback is Fallback.NO_CHARACTERS


# ------------------------------------------- fix 1: the sheet's clothes leak

class TestWardrobeFix:
    def test_sheets_are_head_and_shoulders_with_no_clothing_in_frame(self):
        p = sheet_prompt(JERRY)
        assert "no clothing" in p and "head and neck" in p
        for leak in ("t-shirt", "trousers", "full-body", "standing pose"):
            assert leak not in p

    def test_sheet_prompt_is_built_from_the_description_alone(self):
        p = sheet_prompt(JERRY)
        assert JERRY in p
        assert "Jerry" not in p and "Seinfeld" not in p

    def test_keyframe_prompt_puts_the_scene_before_the_sheet_instruction(self):
        p = keyframe_prompt("Jerry wears a green suede jacket and walks.", 1)
        assert p.index("suede jacket") < p.index("take ONLY the face, hair, skin, age and build")
        assert "come only from this scene" in p
        assert "Do not copy any clothing" in p

    def test_keyframe_prompt_covers_two_sheets_and_rejects_other_counts(self):
        assert "reference sheet 2" in keyframe_prompt("x", 2)
        for n in (0, 3):
            with pytest.raises(ValueError):
                keyframe_prompt("x", n)

    def test_strategy_sends_the_editor_the_shots_own_wardrobe_text(self, tmp_path):
        s = _make(tmp_path)
        route = s.plan([SHOTS[0]])[0]
        p = s._editor_prompt(SHOTS[0], route)
        assert "Jerry in a blazer" not in p.split("take ONLY")[0].split("this scene:")[0]
        assert "blazer" in p and STYLE in p and JERRY in p


# ------------------------------------- fix 2: sheets must not be look-alikes

class TestDistinctSheets:
    def run(self, names, descs, gen, emb, cache=None, **kw):
        return make_sheets(names, descs, gen, emb, cache, log=lambda *_: None, **kw)

    def test_distinct_people_are_all_kept_on_the_first_draw(self):
        gen, emb = FakeGenerator({"AAA": [b"face:a"], "BBB": [b"face:b"]}), FakeEmbedder(
            {"a": [1, 0, 0], "b": [0, 1, 0]})
        out = self.run(["a", "b"], {"a": "AAA", "b": "BBB"}, gen, emb)
        assert set(out.sheets) == {"a", "b"} and not out.dropped
        assert len(gen.sheet_calls) == 2
        assert out.sheets["b"].max_cosine == pytest.approx(0.0)

    def test_a_lookalike_is_reseeded_until_it_is_distinct(self):
        # b's first two draws are the same face as a (cosine 0.78 was Jerry vs Kramer)
        gen = FakeGenerator({"AAA": [b"face:a"], "BBB": [b"face:b_same", b"face:b_same", b"face:b_new"]})
        emb = FakeEmbedder({"a": [1, 0, 0], "b_same": [0.78, 0.626, 0], "b_new": [0, 1, 0]})
        out = self.run(["a", "b"], {"a": "AAA", "b": "BBB"}, gen, emb)
        assert "b" in out.sheets and out.sheets["b"].png == b"face:b_new"
        seeds = [s for p, s in gen.sheet_calls if "BBB" in p]
        assert len(seeds) == 3 and len(set(seeds)) == 3, "each retry uses a new seed"
        assert seeds[0] == sheet_seed("b", 0)

    def test_a_sheet_that_stays_a_lookalike_is_dropped_not_shipped(self):
        gen = FakeGenerator({"AAA": [b"face:a"], "BBB": [b"face:b_same"]})
        emb = FakeEmbedder({"a": [1, 0, 0], "b_same": [0.9, 0.436, 0]})
        out = self.run(["a", "b"], {"a": "AAA", "b": "BBB"}, gen, emb)
        assert "b" not in out.sheets and "a" in out.sheets
        assert "look-alike of a" in out.dropped["b"]
        assert len([1 for p, _ in gen.sheet_calls if "BBB" in p]) == SHEET_ATTEMPTS

    def test_the_more_prominent_character_keeps_the_first_draw(self):
        # order is the caller's: the first name is compared with nobody
        gen = FakeGenerator({"AAA": [b"face:a"], "BBB": [b"face:b"]})
        emb = FakeEmbedder({"a": [1, 0, 0], "b": [1, 0.01, 0]})
        out = self.run(["a", "b"], {"a": "AAA", "b": "BBB"}, gen, emb)
        assert out.sheets["a"].seed == sheet_seed("a", 0) and out.sheets["a"].max_cosine is None

    def test_threshold_is_the_research_values(self):
        assert keyframes.LOOKALIKE_COSINE == 0.5

    def test_a_sheet_with_no_detectable_face_is_redrawn(self):
        gen = FakeGenerator({"AAA": [b"face:none", b"face:a"]})
        out = self.run(["a"], {"a": "AAA"}, gen, FakeEmbedder({"a": [1, 0, 0]}))
        assert out.sheets["a"].png == b"face:a" and len(gen.sheet_calls) == 2

    def test_a_failed_generation_uses_an_attempt_and_can_still_succeed(self):
        gen = FakeGenerator({"AAA": [KeyframeError("boom"), b"face:a"]})
        out = self.run(["a"], {"a": "AAA"}, gen, FakeEmbedder({"a": [1, 0, 0]}))
        assert "a" in out.sheets

    def test_every_attempt_failing_drops_the_character_without_raising(self):
        gen = FakeGenerator({"AAA": [KeyframeError("boom")]})
        out = self.run(["a"], {"a": "AAA"}, gen, FakeEmbedder({}))
        assert out.dropped["a"].startswith("generation failed")

    def test_a_spent_budget_stops_drawing_for_everyone_left(self):
        gen = FakeGenerator({"AAA": [KeyframeBudgetExceeded("cap")], "BBB": [b"face:b"]})
        out = self.run(["a", "b"], {"a": "AAA", "b": "BBB"}, gen, FakeEmbedder({"b": [0, 1, 0]}))
        assert set(out.dropped) == {"a", "b"} and len(gen.sheet_calls) == 1

    def test_sheets_are_cached_and_reused_without_a_call(self, tmp_path):
        cache = DiskCache(str(tmp_path / "characters"))
        emb = FakeEmbedder({"a": [1, 0, 0]})
        first = FakeGenerator({"AAA": [b"face:a"]})
        self.run(["a"], {"a": "AAA"}, first, emb, cache)
        again = FakeGenerator({"AAA": [b"face:other"]})
        out = self.run(["a"], {"a": "AAA"}, again, emb, cache)
        assert again.sheet_calls == [] and out.sheets["a"].png == b"face:a"
        assert os.path.isfile(tmp_path / "characters" / "a_sheet.png")

    def test_a_changed_description_does_not_reuse_the_old_sheet(self, tmp_path):
        cache = DiskCache(str(tmp_path / "characters"))
        emb = FakeEmbedder({"a": [1, 0, 0]})
        self.run(["a"], {"a": "AAA one"}, FakeGenerator({"AAA": [b"face:a"]}), emb, cache)
        gen = FakeGenerator({"AAA": [b"face:a"]})
        self.run(["a"], {"a": "AAA two"}, gen, emb, cache)
        assert len(gen.sheet_calls) == 1


# ---------------------------------------------------------- keyframe pass

class TestMakeKeyframes:
    def sheets(self):
        gen = FakeGenerator({"AAA": [b"face:a"], "BBB": [b"face:b"]})
        emb = FakeEmbedder({"a": [1, 0, 0], "b": [0, 1, 0]})
        return make_sheets(["a", "b"], {"a": "AAA", "b": "BBB"}, gen, emb, log=lambda *_: None)

    def run(self, jobs, gen, cache, sheets=None):
        return make_keyframes(jobs, sheets or self.sheets(), gen, cache, (1280, 704), log=lambda *_: None)

    def test_the_editor_gets_the_shots_sheets_in_order_and_the_shot_as_seed(self, tmp_path):
        gen = FakeGenerator()
        paths, failed = self.run([KeyframeJob(7, ("b", "a"), "P")], gen, DiskCache(str(tmp_path)))
        prompt, pngs, seed, size = gen.keyframe_calls[0]
        assert (prompt, pngs, seed, size) == ("P", [b"face:b", b"face:a"], 7, (1280, 704))
        assert failed == {} and open(paths[7], "rb").read() == b"\x89PNG-keyframe-7"

    def test_a_failed_edit_leaves_the_shot_for_text_and_does_not_raise(self, tmp_path):
        gen = FakeGenerator(fail_keyframes={2})
        paths, failed = self.run([KeyframeJob(1, ("a",), "P"), KeyframeJob(2, ("a",), "P")],
                                 gen, DiskCache(str(tmp_path)))
        assert list(paths) == [1] and failed == {2: Fallback.KEYFRAME_FAILED}

    def test_a_character_without_a_sheet_leaves_the_shot_for_text_with_no_call(self, tmp_path):
        gen = FakeGenerator()
        paths, failed = self.run([KeyframeJob(3, ("a", "zed"), "P")], gen, DiskCache(str(tmp_path)))
        assert paths == {} and failed == {3: Fallback.SHEET_UNAVAILABLE} and gen.keyframe_calls == []

    def test_a_spent_budget_stops_calls_and_the_rest_use_text(self, tmp_path):
        gen = FakeGenerator(budget_after=1)
        jobs = [KeyframeJob(i, ("a",), "P") for i in (1, 2, 3)]
        paths, failed = self.run(jobs, gen, DiskCache(str(tmp_path)))
        assert list(paths) == [1] and set(failed) == {2, 3} and len(gen.keyframe_calls) == 2

    def test_cached_frames_are_reused_until_the_prompt_changes(self, tmp_path):
        cache, sheets = DiskCache(str(tmp_path)), self.sheets()
        self.run([KeyframeJob(1, ("a",), "P")], FakeGenerator(), cache, sheets)
        again = FakeGenerator()
        self.run([KeyframeJob(1, ("a",), "P")], again, cache, sheets)
        assert again.keyframe_calls == []
        self.run([KeyframeJob(1, ("a",), "P changed")], again, cache, sheets)
        assert len(again.keyframe_calls) == 1


# --------------------------------------------- the strategy end to end (mock)

def _prepared(tmp_path, gen=None, shots=SHOTS, **kw):
    gen = gen or FakeGenerator()
    s = _make(tmp_path, gen, _embedder(), **kw)
    s.prepare(shots)
    return s, gen


class TestStrategy:
    def test_registered_by_name_and_opt_in_only(self):
        spec = config.VIDEO_STRATEGIES["runpod-ltx25-keyframe"]
        assert spec.cls is RunPodLtx25KeyframeStrategy and not spec.ab_manifest_only
        assert issubclass(RunPodLtx25KeyframeStrategy, RunPodLtx25Strategy)

    def test_no_default_changed(self):
        assert RunPodWanStrategy._first_part_start_image(object(), 1) is None
        assert RunPodLtx25Strategy._first_part_start_image(object(), 1) is None
        assert config.GenerationStrategy.prepare(object(), SHOTS) is None
        assert "prepare" not in vars(RunPodLtx25Strategy)
        assert not config.VIDEO_STRATEGIES["runpod-ltx25"].ab_manifest_only

    def test_sheets_only_for_people_in_keyframe_shots_most_prominent_first(self, tmp_path):
        s, gen = _prepared(tmp_path)
        assert JERRY in gen.sheet_calls[0][0], "jerry (4 shots) is drawn first"
        assert len(gen.sheet_calls) == 3     # jerry, elaine, kramer: not the droid, not nodesc
        assert not any("gold humanoid droid" in p for p, _ in gen.sheet_calls)

    def test_keyframe_is_the_start_image_of_the_shots_first_part(self, tmp_path):
        s, _ = _prepared(tmp_path)
        s._ensure_pod = MagicMock()
        s._upload_image = MagicMock(side_effect=lambda p: os.path.basename(p))
        s._generate_one_clip = MagicMock(return_value=ClipResult(path="x.mp4", actual_duration_s=2.0, cost=0.0))
        out = s.generate(s.format_prompt(SHOTS[0]), str(tmp_path / "clips"), 1, 2.0, entry=SHOTS[0])
        assert out
        assert s._generate_one_clip.call_args.kwargs["start_image"] == "kf_0001.png"
        s._upload_image.assert_called_once_with(os.path.join(str(tmp_path), "keyframes", "kf_0001.png"))

    def test_only_part_zero_takes_the_keyframe_later_parts_chain(self, tmp_path):
        long_shot = _shot(1, dur=11.0)
        s, _ = _prepared(tmp_path, shots=[long_shot])
        s._ensure_pod = MagicMock()
        s._upload_image = MagicMock(return_value="kf_0001.png")
        s._upload_start_frame = MagicMock(return_value="chain_0001_00.png")
        s._generate_one_clip = MagicMock(return_value=ClipResult(path="x.mp4", actual_duration_s=4.0, cost=0.0))
        s.generate(s.format_prompt(long_shot), str(tmp_path / "clips"), 1, 11.0, entry=long_shot)
        starts = [c.kwargs["start_image"] for c in s._generate_one_clip.call_args_list]
        assert starts[0] == "kf_0001.png" and all(x == "chain_0001_00.png" for x in starts[1:])
        assert len(starts) > 1

    @pytest.mark.parametrize("shot,why", [(5, Fallback.DESIGN_CHARACTER), (6, Fallback.DESIGN_CHARACTER),
                                          (8, Fallback.NO_DESCRIPTION), (9, Fallback.NO_CHARACTERS)])
    def test_shots_without_a_sheeted_person_decode_from_text(self, tmp_path, shot, why):
        s, gen = _prepared(tmp_path)
        assert s.fallbacks[shot] is why
        assert s._first_part_start_image(shot) is None
        assert shot not in [c[2] for c in gen.keyframe_calls], "no editor call for a fallback shot"

    def test_a_keyframe_failure_falls_back_to_text_and_the_shot_still_generates(self, tmp_path):
        s, _ = _prepared(tmp_path, FakeGenerator(fail_keyframes={1}))
        assert s.fallbacks[1] is Fallback.KEYFRAME_FAILED
        s._ensure_pod = MagicMock()
        s._upload_image = MagicMock()
        s._generate_one_clip = MagicMock(return_value=ClipResult(path="x.mp4", actual_duration_s=2.0, cost=0.0))
        out = s.generate(s.format_prompt(SHOTS[0]), str(tmp_path / "clips"), 1, 2.0, entry=SHOTS[0])
        assert out, "the shot is not failed"
        assert s._generate_one_clip.call_args.kwargs["start_image"] is None
        s._upload_image.assert_not_called()

    def test_a_failed_upload_falls_back_to_text_not_to_a_failed_shot(self, tmp_path, capsys):
        s, _ = _prepared(tmp_path)
        s._ensure_pod = MagicMock()
        s._upload_image = MagicMock(return_value=None)
        s._generate_one_clip = MagicMock(return_value=ClipResult(path="x.mp4", actual_duration_s=2.0, cost=0.0))
        out = s.generate(s.format_prompt(SHOTS[0]), str(tmp_path / "clips"), 1, 2.0, entry=SHOTS[0])
        assert out and s._generate_one_clip.call_args.kwargs["start_image"] is None
        assert s.fallbacks[1] is Fallback.UPLOAD_FAILED

    def test_a_dropped_lookalike_sheet_sends_that_characters_shots_to_text(self, tmp_path):
        # kramer's sheet looks like jerry's on every draw
        gen = FakeGenerator()
        emb = _embedder({"kramer": [0.9, 0, 0.436]})    # face cosine 0.9 with jerry on every draw
        s = _make(tmp_path, gen, emb)
        s.prepare(SHOTS)
        assert s.fallbacks[4] is Fallback.SHEET_UNAVAILABLE      # kramer's only shot
        assert {1, 2, 3, 7}.isdisjoint(s.fallbacks), "jerry and elaine are unaffected"

    def test_the_text_prompt_is_unchanged_by_the_keyframe(self, tmp_path):
        s, _ = _prepared(tmp_path)
        plain = RunPodLtx25Strategy(
            output_dir=str(tmp_path), character_shot_map=decode.build_character_shot_map(_chars()),
            characters_data=_chars(), style_prefix=STYLE)
        for shot in SHOTS:
            assert s.format_prompt(shot) == plain.format_prompt(shot)

    def test_graph_is_the_proven_i2v_graph(self, tmp_path):
        s, _ = _prepared(tmp_path)
        wf = s._build_workflow("p", 1, length=49, start_image="kf_0001.png")
        assert wf == RunPodLtx25Strategy._build_workflow(s, "p", 1, length=49, start_image="kf_0001.png")
        assert wf["41"]["class_type"] == "LTXVImgToVideo" and wf["42"]["class_type"] == "LTXVImgToVideoInplace"

    def test_missing_prerequisites_stop_the_run_before_any_pod(self, tmp_path):
        gen = FakeGenerator()
        gen.ok = False
        s = _make(tmp_path, gen, _embedder())
        with pytest.raises(KeyframeSetupError, match="FAL_KEY"):
            s.prepare(SHOTS)
        emb = _embedder()
        emb.ok = False
        with pytest.raises(KeyframeSetupError, match="no models"):
            _make(tmp_path, FakeGenerator(), emb).prepare(SHOTS)
        s._session.ensure_pod.assert_not_called()

    def test_a_film_with_no_keyframe_shots_needs_no_prerequisites(self, tmp_path):
        gen = FakeGenerator()
        gen.ok = False
        s = _make(tmp_path, gen, _embedder())
        s.prepare([SHOTS[4], SHOTS[8]])     # droid-only and nobody
        assert s._keyframes == {} and gen.sheet_calls == []

    def test_a_rerun_reuses_sheets_and_frames_without_new_calls(self, tmp_path):
        _, first = _prepared(tmp_path)
        s, again = _prepared(tmp_path)
        assert again.sheet_calls == [] and again.keyframe_calls == []
        assert set(s._keyframes) == {1, 2, 3, 4, 7}


class TestDecodeLoopIntegration:
    def test_run_decode_calls_prepare_once_with_only_the_unfinished_shots(self, tmp_path):
        out = tmp_path
        json.dump({"format": "v2", "shots": SHOTS, "dialog": []}, open(manifest.shots_path(str(out)), "w"))
        json.dump({"completed": [1], "failed": [], "total_cost_estimate": 0.0, "clips": {}},
                  open(manifest.decode_progress_path(str(out), "fake"), "w"))
        seen = []

        class Fake(config.GenerationStrategy):
            name = "fake"

            def prepare(self, entries):
                seen.append([e["index"] for e in entries])

            def generate(self, *a, **k):
                return []

        args = types.SimpleNamespace(output_dir=str(out), start_index=None, limit=3)
        monkey_sleep = pytest.MonkeyPatch()
        monkey_sleep.setattr(decode.time, "sleep", lambda *_: None)
        try:
            decode.run_decode(args, Fake())
        finally:
            monkey_sleep.undo()
        assert seen == [[2, 3]]


# --------------------------------------------------------------- fal klein

class FakeFal:
    def __init__(self, nsfw=False, fail=0):
        self.calls, self.uploads, self.nsfw, self.fail = [], [], nsfw, fail

    def subscribe(self, model, arguments, with_logs=False):
        self.calls.append((model, arguments))
        if self.fail:
            self.fail -= 1
            raise RuntimeError("fal 503")
        return {"images": [{"url": "https://x/img.png"}], "has_nsfw_concepts": [self.nsfw]}

    def upload_file(self, path):
        self.uploads.append(open(path, "rb").read())
        return f"https://fal/{len(self.uploads)}"


@pytest.fixture
def fal(monkeypatch):
    f = FakeFal()
    mod = types.ModuleType("fal_client")
    mod.subscribe, mod.upload_file = f.subscribe, f.upload_file
    http = types.ModuleType("httpx")
    http.get = lambda url, **k: types.SimpleNamespace(content=b"PNGDATA", raise_for_status=lambda: None)
    monkeypatch.setitem(sys.modules, "fal_client", mod)
    monkeypatch.setitem(sys.modules, "httpx", http)
    monkeypatch.setattr(keyframes.time, "sleep", lambda *_: None)
    return f


class TestFalKleinGenerator:
    def test_uses_the_trials_models_and_arguments(self, fal, tmp_path):
        g = FalKleinGenerator(10, cost_log=str(tmp_path / "cost.jsonl"))
        assert g.make_sheet("S", 5, (1024, 1024)) == b"PNGDATA"
        assert g.make_keyframe("K", [b"s1", b"s2"], 9, (1280, 704)) == b"PNGDATA"
        (m1, a1), (m2, a2) = fal.calls
        assert m1 == "fal-ai/flux-2/klein/4b" and a1["seed"] == 5
        assert m2 == "fal-ai/flux-2/klein/4b/edit" and a2["seed"] == 9
        assert a2["image_urls"] == ["https://fal/1", "https://fal/2"]
        assert a2["image_size"] == {"width": 1280, "height": 704}

    def test_same_sheet_is_uploaded_once(self, fal):
        g = FalKleinGenerator(10)
        g.make_keyframe("K", [b"s1"], 1, (1280, 704))
        g.make_keyframe("K2", [b"s1"], 2, (1280, 704))
        assert fal.uploads == [b"s1"]

    def test_spend_is_logged_and_capped(self, fal, tmp_path):
        log = tmp_path / "k" / "cost.jsonl"
        one = FalKleinGenerator.price_keyframe(1, (1280, 704))
        g = FalKleinGenerator(one * 1.5, cost_log=str(log))
        g.make_keyframe("K", [b"s"], 1, (1280, 704))
        with pytest.raises(KeyframeBudgetExceeded):
            g.make_keyframe("K", [b"s"], 2, (1280, 704))
        assert len(fal.calls) == 1, "the call over the cap is never made"
        rows = [json.loads(line) for line in open(log)]
        assert len(rows) == 1 and rows[0]["usd"] == pytest.approx(one, abs=1e-4)

    def test_a_restarted_run_counts_spend_already_logged(self, fal, tmp_path):
        log = tmp_path / "cost.jsonl"
        one = FalKleinGenerator.price_keyframe(1, (1280, 704))
        FalKleinGenerator(one * 1.5, cost_log=str(log)).make_keyframe("K", [b"s"], 1, (1280, 704))
        restarted = FalKleinGenerator(one * 1.5, cost_log=str(log))     # capped_run's retry
        assert restarted.spent == pytest.approx(one, abs=1e-4)
        with pytest.raises(KeyframeBudgetExceeded):
            restarted.make_keyframe("K", [b"s"], 2, (1280, 704))

    def test_a_blanked_image_is_an_error_not_a_frame(self, fal):
        fal.nsfw = True
        with pytest.raises(KeyframeError, match="safety checker"):
            FalKleinGenerator(10).make_sheet("S", 1, (1024, 1024))

    def test_transient_errors_are_retried_and_billed_once(self, fal):
        fal.fail = 2
        g = FalKleinGenerator(10)
        assert g.make_sheet("S", 1, (1024, 1024)) == b"PNGDATA"
        assert len(fal.calls) == 3 and g.spent == pytest.approx(FalKleinGenerator.price_sheet())

    def test_preflight_names_the_missing_key_without_reading_it(self, monkeypatch):
        monkeypatch.delenv("FAL_KEY", raising=False)
        with pytest.raises(KeyframeSetupError, match="FAL_KEY"):
            FalKleinGenerator(1).preflight()

    def test_face_models_missing_is_a_setup_error(self, tmp_path):
        with pytest.raises(KeyframeSetupError, match="yunet.onnx, sface.onnx"):
            OpenCVFaceEmbedder(str(tmp_path)).preflight()


# ---------------------------------------------------------------- dry run

def _write_output_dir(path):
    json.dump({"format": "v2", "style_prefix": STYLE, "shots": SHOTS, "dialog": []},
              open(manifest.shots_path(str(path)), "w"))
    json.dump(_chars(), open(manifest.characters_path(str(path)), "w"))


class TestDryRun:
    def args(self, path, **kw):
        return types.SimpleNamespace(output_dir=str(path), start_index=None, limit=None,
                                     keyframe_fal_cap=1.0, style_prefix=None,
                                     **{"keyframe_generator": "fal", **kw})

    def test_prints_the_plan_with_no_network_and_writes_nothing(self, tmp_path, capsys, monkeypatch):
        _write_output_dir(tmp_path)
        before = sorted(os.listdir(tmp_path))
        for mod in ("fal_client", "httpx"):   # importing either would mean a network path
            monkeypatch.setitem(sys.modules, mod, None)
        import socket
        monkeypatch.setattr(socket.socket, "connect", lambda *a: pytest.fail("network call"))
        decode.dry_run_keyframes(self.args(tmp_path))
        out = capsys.readouterr().out
        assert "shot 1     KEYFRAME  jerry" in out
        assert "shot 2     KEYFRAME  jerry+elaine" in out
        assert "shot 5     text      design_character" in out
        assert "shot 8     text      no_description" in out
        assert "shot 9     text      no_characters" in out
        assert "5 keyframe shots, 4 text shots" in out
        assert "capped at $1.00" in out
        assert sorted(os.listdir(tmp_path)) == before

    def test_start_index_and_limit_scope_the_plan(self, tmp_path, capsys):
        _write_output_dir(tmp_path)
        args = self.args(tmp_path)
        args.start_index, args.limit = 4, 2
        decode.dry_run_keyframes(args)
        out = capsys.readouterr().out
        assert "shot 4 " in out and "shot 5 " in out and "shot 1 " not in out

    def test_cli_dry_run_through_main(self, tmp_path, capsys, monkeypatch):
        _write_output_dir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["decode.py", str(tmp_path), "--strategy",
                                          "runpod-ltx25-keyframe", "--dry-run"])
        decode.main()
        assert "Dry run OK" in capsys.readouterr().out

    def test_dry_run_flag_is_refused_for_other_strategies_instead_of_spending(self, tmp_path, monkeypatch):
        _write_output_dir(tmp_path)
        # LTX strategies have their own dry run (research 0030); a strategy with
        # none must refuse rather than start a real decode.
        monkeypatch.setattr(sys, "argv", ["decode.py", str(tmp_path), "--strategy", "runpod-wan", "--dry-run"])
        with pytest.raises(SystemExit, match="--dry-run needs"):
            decode.main()


# ------------------------------------------------------------ klein on the pod

def _refs_ok(graph):
    """Every [node, slot] input points at a node in the graph."""
    for node in graph.values():
        for v in node["inputs"].values():
            if isinstance(v, list):
                assert v[0] in graph, f"dangling link {v}"


class TestKleinGraph:
    def test_a_sheet_is_plain_text_to_image(self):
        g = klein_graph("a face", 7, (1024, 1024))
        types_ = {n["class_type"] for n in g.values()}
        assert "LoadImage" not in types_ and "ReferenceLatent" not in types_
        assert g["9"]["inputs"]["conditioning"] == ["4", 0]
        assert g["8"]["inputs"]["noise_seed"] == 7
        assert g["5"]["inputs"] == {"width": 1024, "height": 1024, "batch_size": 1}
        assert g["2"]["inputs"]["type"] == "flux2"
        _refs_ok(g)

    def test_a_keyframe_chains_one_reference_latent_per_sheet(self):
        g = klein_graph("scene", 3, (1280, 704), ["a.png", "b.png"])
        loads = [n["inputs"]["image"] for n in g.values() if n["class_type"] == "LoadImage"]
        assert loads == ["a.png", "b.png"]
        refs = [k for k, n in g.items() if n["class_type"] == "ReferenceLatent"]
        assert len(refs) == 2
        assert g[refs[0]]["inputs"]["conditioning"] == ["4", 0]
        assert g[refs[1]]["inputs"]["conditioning"] == [refs[0], 0]
        assert g["9"]["inputs"]["conditioning"] == [refs[1], 0], "the guider sees both sheets"
        assert g["6"]["inputs"] == {"steps": 4, "width": 1280, "height": 704}
        _refs_ok(g)

    def test_a_size_the_latent_cannot_take_is_refused(self):
        with pytest.raises(ValueError, match="16"):
            klein_graph("x", 1, (1000, 704))


class FakeRunner:
    def __init__(self, images=None, upload_ok=True, boom=False):
        self.ready_calls, self.uploads, self.graphs = 0, [], []
        self.images, self.upload_ok, self.boom = images, upload_ok, boom

    def ensure_ready(self):
        self.ready_calls += 1

    def upload(self, png):
        self.uploads.append(png)
        return f"ref{len(self.uploads)}.png" if self.upload_ok else None

    def run(self, graph):
        self.graphs.append(graph)
        if self.boom:
            raise ConnectionError("pod went away")
        return b"PNG%d" % len(self.graphs) if self.images is None else self.images


class TestPodKleinGenerator:
    def test_boots_the_pod_once_on_first_use_and_logs_each_images_time(self, tmp_path):
        r = FakeRunner()
        g = PodKleinGenerator(r, timing_log=str(tmp_path / "kf" / "timing.jsonl"))
        assert r.ready_calls == 0 and not g.used_pod, "constructing it boots nothing"
        assert g.make_sheet("S", 1, (1024, 1024)) == b"PNG1"
        assert g.make_keyframe("K", [b"s1"], 9, (1280, 704)) == b"PNG2"
        assert r.ready_calls == 1 and g.used_pod
        rows = [json.loads(l) for l in open(tmp_path / "kf" / "timing.jsonl")]
        assert [x["kind"] for x in rows] == ["sheet", "keyframe"] and all(x["seconds"] >= 0 for x in rows)

    def test_a_sheet_is_uploaded_once_and_reused_by_name(self):
        r = FakeRunner()
        g = PodKleinGenerator(r)
        g.make_keyframe("K", [b"s1", b"s2"], 1, (1280, 704))
        g.make_keyframe("K", [b"s1"], 2, (1280, 704))
        assert r.uploads == [b"s1", b"s2"]
        loads = [n["inputs"]["image"] for n in r.graphs[1].values() if n["class_type"] == "LoadImage"]
        assert loads == ["ref1.png"]

    @pytest.mark.parametrize("runner", [FakeRunner(images=b""), FakeRunner(boom=True),
                                        FakeRunner(upload_ok=False)])
    def test_a_failed_image_is_a_keyframe_error_so_the_shot_falls_back(self, runner):
        with pytest.raises(KeyframeError):
            PodKleinGenerator(runner).make_keyframe("K", [b"s1"], 1, (1280, 704))

    def test_id_differs_from_fal_so_caches_never_mix(self):
        assert PodKleinGenerator.id != FalKleinGenerator.id

    def test_preflight_needs_the_runpod_key(self, monkeypatch):
        monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
        with pytest.raises(KeyframeSetupError, match="RUNPOD_API_KEY"):
            PodKleinGenerator(FakeRunner()).preflight()


class TestPodKeyframePass:
    def _pod(self, tmp_path, monkeypatch, **kw):
        monkeypatch.setenv("RUNPOD_API_KEY", "x")
        s = _make(tmp_path, None, _embedder(), keyframe_generator="pod", **kw)
        order = []
        s._ensure_pod = MagicMock(side_effect=lambda: order.append("boot"))
        s._session.unload_models.side_effect = lambda: order.append("unload")
        return s, order

    def _fake_pod_images(self, s, order):
        """Answer klein graphs as the pod would: a sheet's face key from its prompt."""
        def submit(graph, timeout):
            text = graph["4"]["inputs"]["text"]
            order.append("image")
            if "reference sheet" in text and "Character" in text:
                who = "elaine" if "woman" in text else "kramer" if "tall and slim" in text else "jerry"
                return {"outputs": {"12": {"images": [{"filename": f"face:{who}"}]}}}
            return {"outputs": {"12": {"images": [{"filename": "kf"}]}}}
        s._session.submit_workflow.side_effect = submit
        s._session.download_output.side_effect = lambda img: (
            img["filename"].encode() if img["filename"].startswith("face:") else b"\x89PNG-kf")
        s._upload_image = MagicMock(side_effect=os.path.basename)

    def test_pod_option_downloads_klein_beside_ltx_and_fal_option_does_not(self, tmp_path, monkeypatch):
        s, _ = self._pod(tmp_path, monkeypatch)
        klein = {m.dest for m in RunPodLtx25KeyframeStrategy.KLEIN_MODELS}
        assert klein <= {m.dest for m in s.PINNED_MODELS}
        assert {m.dest for m in RunPodLtx25Strategy.PINNED_MODELS} <= {m.dest for m in s.PINNED_MODELS}
        fal_s = _make(tmp_path, FakeGenerator(), _embedder())
        assert klein.isdisjoint({m.dest for m in fal_s.PINNED_MODELS})
        assert all(len(m.sha256) == 64 and "/resolve/" in m.url and not m.gate_url
                   for m in RunPodLtx25KeyframeStrategy.KLEIN_MODELS)

    def test_keyframes_are_drawn_after_boot_then_klein_is_unloaded_before_any_video(self, tmp_path, monkeypatch):
        s, order = self._pod(tmp_path, monkeypatch)
        self._fake_pod_images(s, order)
        s.prepare(SHOTS)
        assert order[0] == "boot" and order[-1] == "unload"
        assert order.count("boot") == 1 and order.count("unload") == 1
        assert set(s._keyframes) == {1, 2, 3, 4, 7}
        assert isinstance(s._get_generator(), PodKleinGenerator)
        assert os.path.exists(os.path.join(str(tmp_path), "keyframes", "timing.jsonl"))

    def test_a_fully_cached_run_boots_nothing_in_prepare(self, tmp_path, monkeypatch):
        s, order = self._pod(tmp_path, monkeypatch)
        self._fake_pod_images(s, order)
        s.prepare(SHOTS)
        again, order2 = self._pod(tmp_path, monkeypatch)
        again.prepare(SHOTS)
        assert order2 == [], "no boot, no unload: every sheet and frame came from the cache"
        assert set(again._keyframes) == {1, 2, 3, 4, 7}

    def test_klein_is_unloaded_even_when_the_pass_raises(self, tmp_path, monkeypatch):
        s, order = self._pod(tmp_path, monkeypatch)
        self._fake_pod_images(s, order)
        s._get_embedder().faces = MagicMock(side_effect=[[np.array([1.0, 0, 0])], RuntimeError("cv2")])
        with pytest.raises(RuntimeError):
            s.prepare(SHOTS)
        assert order[-1] == "unload"

    def test_an_unknown_generator_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="keyframe_generator"):
            _make(tmp_path, keyframe_generator="replicate")

    def test_runner_returns_the_first_saved_image(self):
        st = MagicMock()
        st._session.submit_workflow.return_value = {"outputs": {"3": {}, "12": {"images": [{"filename": "k.png"}]}}}
        st._session.download_output.return_value = b"IMG"
        assert _PodComfyRunner(st).run({}) == b"IMG"
        st._session.submit_workflow.return_value = None
        assert _PodComfyRunner(st).run({}) is None

    def test_cli_flag_reaches_the_strategy_and_defaults_to_fal(self, tmp_path, monkeypatch):
        _write_output_dir(tmp_path)
        made = []
        monkeypatch.setattr(decode, "_create_ltx2_strategy", lambda args, cls, **kw: made.append(kw))
        args = types.SimpleNamespace(keyframe_fal_cap=1.0, keyframe_generator="pod")
        decode._create_keyframe_strategy(args)
        assert made[-1]["keyframe_generator"] == "pod"
        monkeypatch.setattr(sys, "argv", ["decode.py", str(tmp_path), "--strategy", "runpod-ltx25-keyframe",
                                          "--keyframe-generator", "pod", "--dry-run"])
        decode.main()

    def test_dry_run_prints_pod_time_not_fal_dollars_for_the_pod_option(self, tmp_path, capsys):
        _write_output_dir(tmp_path)
        decode.dry_run_keyframes(TestDryRun().args(tmp_path, keyframe_generator="pod"))
        out = capsys.readouterr().out
        assert "Pod klein:" in out and "fal estimate" not in out
