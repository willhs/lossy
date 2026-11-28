"""faces.py: which face in a frame is the character, and the swap-engine seam.

YuNet and SFace are faked (a test image lists its own faces), so no models are needed; one test
runs the real models when LOSSY_FACE_MODELS points at them.
"""

import os

import cv2
import numpy as np
import pytest

import faces


def _face(h, k=0, w=None):
    e = np.zeros(4)
    e[k] = 1.0
    return faces.Face((0, 0, w or h, h), 0.9, e)


class _Det:
    """YuNet stand-in: the image's top-left pixel row says which faces it holds."""

    def __init__(self, rows):
        self.rows = rows

    def setInputSize(self, size):
        self.size = size

    def detect(self, img):
        return 1, (np.array(self.rows, np.float32) if self.rows else None)


class _Rec:
    """SFace stand-in: the 'aligned crop' is the face row; its embedding is unnormalised."""

    def alignCrop(self, img, f):
        return f

    def feature(self, crop):
        return np.array([[crop[0] + 1.0, 2.0 * crop[1] + 1.0, 3.0]], np.float32)


def _identity(rows):
    fi = faces.FaceIdentity("unused")
    fi._det, fi._rec = _Det(rows), _Rec()
    return fi


def _row(x, y, w, h, score=0.9):
    return [x, y, w, h] + [0.0] * 10 + [score]


class TestFind:
    def test_keeps_detection_order_and_drops_short_faces(self):
        # 0034's labels index this list, so a larger later face must stay later
        fi = _identity([_row(10, 0, 30, 50), _row(20, 0, 30, 30), _row(30, 0, 90, 90)])
        found = fi.find(np.zeros((100, 100, 3), np.uint8), min_h=40)
        assert [f.box for f in found] == [(10, 0, 30, 50), (30, 0, 90, 90)]

    def test_embeddings_are_unit_length_and_score_is_yunets(self):
        found = _identity([_row(1, 2, 50, 50, score=0.77)]).find(np.zeros((10, 10, 3), np.uint8))
        assert np.linalg.norm(found[0].emb) == pytest.approx(1.0)
        assert found[0].score == pytest.approx(0.77)

    def test_no_faces(self):
        assert _identity([]).find(np.zeros((10, 10, 3), np.uint8)) == []

    def test_faces_of_png_bytes(self):
        fi = _identity([_row(1, 2, 50, 50), _row(3, 4, 60, 60)])
        png = cv2.imencode(".png", np.zeros((8, 8, 3), np.uint8))[1].tobytes()
        embs = fi.faces(png)
        assert len(embs) == 2 and all(np.linalg.norm(e) == pytest.approx(1.0) for e in embs)
        assert fi.faces(b"not an image") == []


class TestPick:
    def test_unlabelled_frame_takes_the_tallest(self):
        a, b = _face(50, 0), _face(80, 1, w=10)
        assert faces.pick([a, b], {}, "7/first") is b

    def test_tallest_tie_takes_the_first(self):
        a, b = _face(60, 0), _face(60, 1)
        assert faces.largest([a, b]) is a

    def test_label_overrides_size(self):
        a, b = _face(50, 0), _face(80, 1)
        assert faces.pick([a, b], {"7/first": 0}, "7/first") is a

    def test_null_label_drops_the_frame(self):
        assert faces.pick([_face(50)], {"7/first": None}, "7/first") is None

    def test_no_faces_unlabelled(self):
        assert faces.pick([], {}, "7/first") is None

    def test_by_area_is_largest_first_and_stable(self):
        a, b, c = _face(10, 0, w=10), _face(20, 1, w=4), _face(10, 2, w=10)   # areas 100, 80, 100
        assert faces.by_area([a, b, c]) == [a, c, b]

    def test_most_like(self):
        a, b = _face(50, 0), _face(50, 1)
        got, sim = faces.most_like([a, b], np.array([0.0, 1.0, 0.0, 0.0]))
        assert got is b and sim == pytest.approx(1.0)
        assert faces.most_like([], np.ones(4)) is None

    def test_signature_is_the_unit_centroid(self):
        sig = faces.signature([np.array([1.0, 0.0]), np.array([0.0, 1.0])])
        assert sig == pytest.approx(np.array([1.0, 1.0]) / np.sqrt(2))
        assert faces.signature([]) is None

    def test_shot_times_stay_off_the_cuts(self):
        t = faces.shot_times(10.0, 12.0)
        assert t == pytest.approx({"first": 10.06, "middle": 11.0, "last": 11.94})


class TestModels:
    def test_missing_models_name_the_files(self, tmp_path):
        with pytest.raises(faces.FaceModelsMissing, match="yunet.onnx, sface.onnx"):
            faces.FaceIdentity(str(tmp_path)).preflight()

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("LOSSY_FACE_MODELS", "/x/face")
        assert faces.FaceIdentity.from_env().models_dir == "/x/face"
        monkeypatch.delenv("LOSSY_FACE_MODELS")
        assert faces.FaceIdentity.from_env().models_dir == "models/face"

    @pytest.mark.skipif(not os.path.isfile(os.path.join(os.environ.get("LOSSY_FACE_MODELS", "models/face"),
                                                        "yunet.onnx")), reason="needs LOSSY_FACE_MODELS")
    def test_real_models_run(self):
        fi = faces.FaceIdentity.from_env()
        fi.preflight()
        assert fi.find(np.zeros((200, 200, 3), np.uint8)) == []


ENGINE_OK = '''
from dataclasses import dataclass

@dataclass
class Face:
    box: tuple

class OnnxEngine:
    def __init__(self, models_dir, threads=2):
        self.models_dir, self.threads = models_dir, threads
    def detect(self, img): return []
    def embed(self, img, face): return None
    def swap_aligned(self, img, landmarks, identity): return (img,)
'''


class TestSwapEngine:
    def test_unset_names_the_variable(self, monkeypatch):
        monkeypatch.delenv(faces.SWAP_ENGINE_ENV, raising=False)
        with pytest.raises(faces.SwapEngineUnavailable, match=faces.SWAP_ENGINE_ENV):
            faces.load_swap_engine("m")

    def test_missing_file(self, tmp_path):
        with pytest.raises(faces.SwapEngineUnavailable, match="no such file"):
            faces.load_swap_engine("m", path=str(tmp_path / "nope.py"))

    def test_wrong_module(self, tmp_path):
        p = tmp_path / "engine.py"
        p.write_text("class OnnxEngine:\n    def detect(self, img): return []\n")
        with pytest.raises(faces.SwapEngineUnavailable, match="embed, swap_aligned"):
            faces.load_swap_engine("m", path=str(p))

    def test_loads_the_engine_from_the_env_var(self, tmp_path, monkeypatch):
        p = tmp_path / "face_swap_engine.py"
        p.write_text(ENGINE_OK)    # a dataclass at module level: needs sys.modules registration
        monkeypatch.setenv(faces.SWAP_ENGINE_ENV, str(p))
        engine = faces.load_swap_engine(str(tmp_path), threads=3)
        assert engine.threads == 3 and str(engine.models_dir) == str(tmp_path)
