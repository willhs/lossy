"""tools/rd_curve.py: the bit-rate meter, the variant writer and the decode plan.

No pod, no network, no Gemini: the rewriter is a stand-in and the prompt builder is a lambda.
"""

import json
import os
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import rd_curve as rd  # noqa: E402


def shot(i, words=20, start=None, dur=2.0):
    start = i * 3.0 if start is None else start
    text = " ".join(f"w{i}x{n}" for n in range(words))
    return {"index": i, "start_s": start, "end_s": start + dur, "duration_s": dur,
            "dialogue": [{"text": "hello"}],
            "description": {"shot_type": "wide", "camera_movement": "static", "subjects": text,
                            "action": "", "setting": "a desert", "sound": "wind"}}


def scene(n=8, words=20):
    return [shot(100 + i, words) for i in range(n)]


# --- words -----------------------------------------------------------------

def test_count_words_covers_video_fields_only():
    d = {"shot_type": "wide", "camera_movement": "slow pan", "subjects": ["a man", "a droid"],
         "sound": "ten words of sound that must not count toward the axis at all"}
    assert rd.count_words(d) == 1 + 2 + 4


def test_count_words_tolerates_dicts_and_missing():
    assert rd.count_words({"subjects": {"subject": "a man", "description": "tall"}}) == 3
    assert rd.count_words({}) == 0


# --- bits per second ---------------------------------------------------------

def test_xz_size_matches_the_xz_binary():
    if not shutil.which("xz"):
        pytest.skip("no xz binary")
    data = json.dumps([{"a": i, "b": "repeat " * 30} for i in range(200)]).encode()
    cli = subprocess.run(["xz", "-9e", "--check=none", "-T1", "-c"], input=data, capture_output=True, check=True).stdout
    assert rd.xz_size(data) == len(cli)


def test_bits_per_second_is_xz_bits_over_scene_seconds():
    shots = scene()
    m = rd.measure_variant(shots)
    assert m["duration_s"] == pytest.approx(shots[-1]["end_s"] - shots[0]["start_s"])
    assert m["bits_per_s"] == pytest.approx(8 * rd.xz_size(rd.stream_bytes(shots)) / m["duration_s"], abs=0.01)
    assert m["words_total"] == sum(rd.count_words(s["description"]) for s in shots)


def test_stream_ignores_audio_fields_and_dialogue():
    a, b = scene(3), scene(3)
    b[0]["description"]["sound"] = "something entirely different"
    b[0]["dialogue"] = []
    assert rd.stream_bytes(a) == rd.stream_bytes(b)


def test_fewer_words_cost_fewer_bits():
    shots = scene(20, words=60)
    short = rd.build_variant(shots, rd.BY_LABEL["w0"], {})
    assert rd.measure_variant(short)["bits_per_s"] < rd.measure_variant(shots)["bits_per_s"]


def test_zero_word_stream_is_timing_plus_one_constant():
    zero = rd.build_variant(scene(10, words=50), rd.BY_LABEL["w0"], {})
    assert {s["description"]["subjects"] for s in zero} == {rd.GENERIC_TEXT}
    assert rd.count_words(zero[0]["description"]) == len(rd.GENERIC_TEXT.split())
    # the generic text repeats, so xz makes it nearly free: only timing is left
    full = rd.measure_variant(scene(10, words=50))["xz_bytes"]
    assert 0 < rd.measure_variant(zero)["xz_bytes"] < full / 2


# --- variant writer ----------------------------------------------------------

def test_full_variant_is_the_scene_unchanged():
    s = scene()
    assert rd.build_variant(s, rd.BY_LABEL["full"], {}) == s


def test_zero_variant_keeps_timing_and_audio_fields():
    s = scene(3)
    z = rd.build_variant(s, rd.BY_LABEL["w0"], {})
    assert [(x["index"], x["start_s"], x["end_s"]) for x in z] == [(x["index"], x["start_s"], x["end_s"]) for x in s]
    assert z[0]["description"]["sound"] == "wind"
    assert "setting" not in z[0]["description"]


def test_short_descriptions_are_kept_not_rewritten():
    s = [shot(1, words=5), shot(2, words=200)]
    assert [rd.needs_rewrite(x, rd.BY_LABEL["w32"]) for x in s] == [False, True]
    out = rd.build_variant(s, rd.BY_LABEL["w32"], {"2": {"subjects": "short"}})
    assert out[0] == s[0] and out[1]["description"] == {"subjects": "short"}


def test_build_variant_names_the_missing_rewrite():
    with pytest.raises(KeyError, match="101"):
        rd.build_variant([shot(101, words=200)], rd.BY_LABEL["w8"], {})


def test_rewrite_must_fit_the_budget_and_keep_audio_fields():
    s = shot(1, words=100)
    ok = rd.check_rewrite(s["description"], {"subjects": "a man", "setting": "desert"}, 8)
    assert ok["sound"] == "wind" and rd.count_words(ok) == 3
    for bad in ({"subjects": "word " * 9}, {"sound": "x"}, {}, ["not", "a", "dict"]):
        with pytest.raises(rd.RewriteError):
            rd.check_rewrite(s["description"], bad, 8)


def test_rewrite_shot_retries_with_feedback_then_fails_closed():
    s = shot(7, words=100)
    seen = []

    def rewriter(desc, words, feedback):
        seen.append(feedback)
        return {"subjects": "word " * (20 if feedback is None else 4)}

    out = rd.rewrite_shot(s, rd.BY_LABEL["w8"], rewriter)
    assert seen[0] is None and "20 words, budget 8" in seen[1]
    assert rd.count_words(out) == 4
    with pytest.raises(rd.RewriteError, match="shot 7 at w8"):
        rd.rewrite_shot(s, rd.BY_LABEL["w8"], lambda d, w, f: {"subjects": "word " * 50}, tries=2)


def test_write_variant_is_a_v2_shots_file(tmp_path):
    path = rd.write_variant(str(tmp_path), rd.BY_LABEL["w0"], rd.build_variant(scene(3), rd.BY_LABEL["w0"], {}))
    raw = json.load(open(path))
    assert raw["format"] == "v2" and len(raw["shots"]) == 3 and raw["rd"]["budget"] == "w0"


# --- scene and steps ---------------------------------------------------------

def test_select_scene_is_contiguous_and_fails_on_a_gap():
    shots = [shot(i) for i in range(100, 160) if i != 120]
    assert [s["index"] for s in rd.select_scene(shots, 100, 10)] == list(range(100, 110))
    with pytest.raises(ValueError, match="120"):
        rd.select_scene(shots, 115, 10)


def test_steps_match_the_protocol():
    sc = [shot(i) for i in range(rd.SCENE_FIRST, rd.SCENE_FIRST + rd.SCENE_COUNT)]
    pilot, chart, identity = rd.make_steps(sc)
    assert len(rd.core_indices(sc)) == 10
    assert pilot.n_clips() == 30            # 10 shots x (8, 32, full) x 1 seed
    assert chart.n_clips() == 300           # 40 x 5 x 1 seed + 10 x 5 x 2 more seeds
    assert identity.n_clips() == 30
    pilot_keys = {(a.name, i) for a, shots in pilot.cells for i in shots}
    chart_keys = {(a.name, i) for a, shots in chart.cells for i in shots}
    assert pilot_keys <= chart_keys         # the chart resumes over the pilot's clips
    assert not any(a.identity for a, _ in chart.cells)


def test_arm_seed_one_is_the_shot_index_and_seeds_differ():
    a1, a2, a3 = (rd.Arm(rd.BY_LABEL["full"], k) for k in (1, 2, 3))
    assert a1.seed(450) == 450 and len({a1.seed(450), a2.seed(450), a3.seed(450)}) == 3
    assert rd.Arm(rd.BY_LABEL["w32"], 1, identity=True).name == "w32_s1_id"


# --- jobs, cost, commands ----------------------------------------------------

def test_frames_match_ltx25_frames():
    from strategies_video import ltx25_frames
    for d in (0.3, 0.8, 1.0, 2.34, 4.8, 9.2, 30):
        assert rd.frames_for(d) == ltx25_frames(d)


def test_build_jobs_shape_and_span():
    sc = scene(4)
    sc[1]["duration_s"] = sc[1]["end_s"] = 9.0   # a long shot renders only its first clip
    sc[1]["end_s"] = sc[1]["start_s"] + 9.0
    variants = {b.label: {s["index"]: s for s in rd.build_variant(sc, b, {})} for b in (rd.BY_LABEL["w0"], rd.BY_LABEL["full"])}
    step = rd.Step("t", ((rd.Arm(rd.BY_LABEL["w0"], 2), (101,)), (rd.Arm(rd.BY_LABEL["full"], 1, True), (100, 101))))
    jobs = rd.build_jobs(step, variants, lambda e, ident: f"{e['description']['subjects'][:6]}|{ident}")
    assert [(j["arm"], j["shot"]) for j in jobs] == [("w0_s2", 101), ("full_s1_id", 100), ("full_s1_id", 101)]
    assert jobs[0]["seed"] == 101 + 10000 and jobs[1]["seed"] == 100
    assert jobs[0]["prompt"] == "a shot|False" and jobs[1]["prompt"].endswith("|True")
    assert jobs[0]["frames"] == 121 and jobs[0]["span_s"] == [sc[1]["start_s"], round(sc[1]["start_s"] + 121 / 25, 3)]
    from decode import AB_JOB_KEYS
    assert all(k in j for j in jobs for k in AB_JOB_KEYS)


def test_project_cost_uses_the_larger_estimate_and_a_cap_with_headroom():
    jobs = [{"frames": 57}] * 30
    c = rd.project_cost(jobs)
    assert c["projected_usd"] == max(c["model_usd"], c["flat_usd"]) == c["flat_usd"] == pytest.approx(0.72)
    assert c["cap_usd"] >= 1.5 * c["projected_usd"] + 0.5 and c["cap_usd"] * 2 == int(c["cap_usd"] * 2)
    long = rd.project_cost([{"frames": 121}] * 30)
    assert long["model_usd"] > rd.project_cost([{"frames": 25}] * 30)["model_usd"]


def test_decode_command_is_capped_and_points_at_the_run_dir():
    cmd = rd.decode_command("output/rd_curve", "output/rd_curve/manifests/pilot.json", 2.0)
    assert cmd.startswith("uv run python tools/capped_run.py --cap 2 ")
    assert "--output-dir output/rd_curve -- uv run python decode.py output/rd_curve" in cmd
    assert "--ab-manifest output/rd_curve/manifests/pilot.json" in cmd
    dry = rd.decode_command("r", "m.json", 2.0, dry_run=True)
    assert dry.endswith("--dry-run") and "capped_run" not in dry


def test_av1_commands_encode_the_scene_span_two_pass(tmp_path):
    sc = scene(4)
    p1, p2 = rd.av1_command("src.mp4", sc, 10, str(tmp_path))
    assert "-b:v" in p1 and p1[p1.index("-b:v") + 1] == "10k" and p1[p1.index("-pass") + 1] == "1"
    assert p2[p2.index("-pass") + 1] == "2" and p2[-1].endswith("av1_10k.mkv")
    assert p1[p1.index("-ss") + 1] == f"{sc[0]['start_s']:.3f}"
    assert p1[p1.index("-t") + 1] == f"{rd.scene_duration(sc):.3f}"
    assert "-an" in p1


def test_measure_av1_reads_measured_file_size(tmp_path):
    (tmp_path / "av1").mkdir()
    (tmp_path / "av1" / "av1_10k.mkv").write_bytes(b"x" * 1250)
    m = rd.measure_av1(str(tmp_path), 100.0)
    assert m == {"av1_10k": {"target_kbps": 10, "bytes": 1250, "bits_per_s": 100.0}}


def test_gemini_rewriter_sends_budget_and_feedback_without_a_network(monkeypatch):
    import encode

    calls = []

    class FakeModels:
        def generate_content(self, model, contents, config):
            calls.append((model, contents[0].parts[0].text, config.system_instruction))
            return type("R", (), {"text": '{"subjects": "a man"}'})()

    monkeypatch.setattr(encode, "_gemini_client", lambda key: type("C", (), {"models": FakeModels()})())
    rewriter, model = rd.gemini_rewriter()
    out = rewriter({"subjects": "a tall man", "sound": "wind"}, 8, "12 words, budget 8")
    assert out == {"subjects": "a man"}
    _, text, system = calls[0]
    assert '"a tall man"' in text and "wind" not in text      # audio fields are not sent
    assert "12 words, budget 8" in text and "at most 8 words" in system
    assert model
