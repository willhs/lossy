"""Face identity: who is in this frame, measured one way everywhere.

YuNet finds faces and SFace embeds them (OpenCV-zoo ``yunet.onnx`` and ``sface.onnx`` in
``LOSSY_FACE_MODELS``); a character is the face that matches its reference sheet. Used by the
keyframe sheet gate (``keyframes.make_sheets``), the face scorers in ``docs/research/`` and the
post-render swap (``tools/face_swap.py``).

Two rules every caller relies on:

* ``FaceIdentity.find`` keeps YuNet's detection order. Research 0034's by-eye labels
  (``labels.json``: ``{"<shot>/<pos>": k}``) index into it, so it must never be re-sorted here;
  a caller that wants another order (research 0029: largest area first) sorts the result.
* Picking a character's face is research 0034's rule (``pick``): the face labelled by eye, else
  the largest by height. Never "the first face YuNet finds": in Star Wars that is often a
  stormtrooper helmet or Chewbacca.

The swap engine (screenquote's HyperSwap ``face_swap_engine.py``) needs onnxruntime, which lossy
does not depend on, so it stays outside the repo: ``load_swap_engine`` imports the file named by
``LOSSY_FACE_SWAP_ENGINE`` and checks it has the calls the swap uses. This module imports only
cv2, numpy and the stdlib, so it also runs in screenquote's venv.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

MODELS_ENV = "LOSSY_FACE_MODELS"
DEFAULT_MODELS_DIR = "models/face"
SCORE_THRESHOLD = 0.6        # YuNet confidence; helmets reach 0.88, so it does not reject them


class FaceModelsMissing(Exception):
    """yunet.onnx / sface.onnx are not in the models dir."""


@dataclass(frozen=True, eq=False)
class Face:
    """One face found in one image."""

    box: tuple[int, int, int, int]   # x, y, w, h in px of the image it was found in
    score: float                     # YuNet confidence
    emb: np.ndarray                  # unit-length SFace embedding

    @property
    def height(self) -> int:
        return self.box[3]

    @property
    def area(self) -> int:
        return self.box[2] * self.box[3]


class FaceIdentity:
    """YuNet + SFace on CPU, loaded on first use."""

    FILES = ("yunet.onnx", "sface.onnx")

    def __init__(self, models_dir: str):
        self.models_dir = models_dir
        self._det = self._rec = None

    @classmethod
    def from_env(cls) -> "FaceIdentity":
        return cls(os.environ.get(MODELS_ENV, DEFAULT_MODELS_DIR))

    def preflight(self) -> None:
        missing = [f for f in self.FILES if not os.path.isfile(os.path.join(self.models_dir, f))]
        if missing:
            raise FaceModelsMissing(
                f"face models missing in {self.models_dir!r}: {', '.join(missing)}. Download "
                "face_detection_yunet and face_recognition_sface from github.com/opencv/opencv_zoo "
                f"and set {MODELS_ENV} to that directory.")

    def _load(self) -> None:
        if self._det is None:
            import cv2
            self._det = cv2.FaceDetectorYN.create(
                os.path.join(self.models_dir, "yunet.onnx"), "", (320, 320), SCORE_THRESHOLD)
            self._rec = cv2.FaceRecognizerSF.create(os.path.join(self.models_dir, "sface.onnx"), "")

    def find(self, img: np.ndarray, min_h: int = 0) -> list[Face]:
        """Every face at least ``min_h`` px tall, in YuNet's detection order."""
        self._load()
        self._det.setInputSize((img.shape[1], img.shape[0]))
        _, found = self._det.detect(img)
        out = []
        for f in (found if found is not None else []):
            if f[3] < min_h:
                continue
            e = self._rec.feature(self._rec.alignCrop(img, f)).flatten()
            out.append(Face(tuple(int(v) for v in f[:4]), float(f[-1]), e / np.linalg.norm(e)))
        return out

    def faces(self, png: bytes) -> list[np.ndarray]:
        """Unit embedding of every face in an encoded image (empty if none or undecodable)."""
        import cv2
        img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        return [] if img is None else [f.emb for f in self.find(img)]


# ------------------------------------------------------------------ picking

def largest(faces: Sequence[Face]) -> Face | None:
    """The tallest face (the first of equals), or None."""
    return max(faces, key=lambda f: f.height) if faces else None


def by_area(faces: Sequence[Face]) -> list[Face]:
    """Largest area first (research 0029's order), detection order among equals."""
    return sorted(faces, key=lambda f: -f.area)


def pick(faces: Sequence[Face], labels: Mapping[str, int | None], key: str) -> Face | None:
    """The character's face in frame ``key``: research 0034's rule.

    A frame labelled by eye uses candidate ``labels[key]`` (None: no face of that character in
    it); an unlabelled frame uses the largest face.
    """
    if key in labels:
        k = labels[key]
        return None if k is None else faces[k]
    return largest(faces)


def sheet_face(faces: Sequence[Face]) -> Face | None:
    """The face a reference sheet stands for: its largest."""
    return largest(faces)


def most_like(faces: Sequence[Face], identity: np.ndarray) -> tuple[Face, float] | None:
    """The face most like ``identity`` (a unit embedding) and its cosine, or None."""
    if not faces:
        return None
    sims = [float(f.emb @ identity) for f in faces]
    k = int(np.argmax(sims))
    return faces[k], sims[k]


def signature(embs: Sequence[np.ndarray]) -> np.ndarray | None:
    """Unit centroid of some embeddings, or None if there are none."""
    if not len(embs):
        return None
    m = np.mean(embs, axis=0)
    return m / np.linalg.norm(m)


# ------------------------------------------------------- frames for scoring

EDGE = 0.06                               # s in from a cut, so a frame is never the neighbour's
POSITIONS = ("first", "middle", "last")
WINDOW_SIZE = (1280, 704)


def shot_times(t0: float, t1: float) -> dict[str, float]:
    """Where in a shot ``[t0, t1)`` its first, middle and last frames are sampled."""
    return {"first": t0 + EDGE, "middle": (t0 + t1) / 2, "last": t1 - EDGE}


def frame_at(video: str, t: float, size: tuple[int, int] = WINDOW_SIZE) -> np.ndarray:
    """The frame at ``t`` s, Lanczos-scaled to ``size`` (BGR)."""
    import cv2
    png = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(t, 0):.3f}", "-i", video, "-vf",
                          f"scale={size[0]}:{size[1]}:flags=lanczos", "-frames:v", "1", "-f",
                          "image2pipe", "-vcodec", "png", "-"], capture_output=True, check=True).stdout
    return cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)


# --------------------------------------------------------------- swap seam

SWAP_ENGINE_ENV = "LOSSY_FACE_SWAP_ENGINE"
# The engine this seam was checked against: screenquote commit 1e09a04,
# screenquote_web/face_swap_engine.py (OnnxEngine: SCRFD, ArcFace, HyperSwap 1b, BiSeNet).
SWAP_ENGINE_CHECKED_AT = "screenquote 1e09a04"
_SWAP_CALLS = ("detect", "embed", "swap_aligned")


class SwapEngineUnavailable(Exception):
    """The swap engine file is not configured, missing, or not the expected engine."""


class SwapEngine(Protocol):
    """What the post-render swap needs. Its faces are the engine's own (``box`` is x0, y0, x1, y1;
    ``landmarks`` for alignment), not ``Face``; its embeddings are ArcFace, not SFace."""

    def detect(self, img: np.ndarray) -> list[Any]: ...

    def embed(self, img: np.ndarray, face: Any) -> np.ndarray: ...

    def swap_aligned(self, img: np.ndarray, landmarks: Any, identity: np.ndarray) -> tuple: ...


def load_swap_engine(models_dir: str, threads: int = 6, path: str | None = None) -> SwapEngine:
    """screenquote's ``OnnxEngine`` from the file in ``path`` or ``LOSSY_FACE_SWAP_ENGINE``.

    Run it with an interpreter that has onnxruntime (screenquote's venv). Nothing is written
    into the engine's checkout if run with ``PYTHONDONTWRITEBYTECODE=1``.
    """
    path = path or os.environ.get(SWAP_ENGINE_ENV)
    if not path:
        raise SwapEngineUnavailable(
            f"{SWAP_ENGINE_ENV} is not set: point it at screenquote's "
            f"screenquote_web/face_swap_engine.py (checked against {SWAP_ENGINE_CHECKED_AT}).")
    if not os.path.isfile(path):
        raise SwapEngineUnavailable(f"{SWAP_ENGINE_ENV}={path!r}: no such file")
    spec = importlib.util.spec_from_file_location("face_swap_engine", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["face_swap_engine"] = module   # its dataclasses look themselves up here
    spec.loader.exec_module(module)
    cls = getattr(module, "OnnxEngine", None)
    lacking = [c for c in _SWAP_CALLS if not callable(getattr(cls, c, None))]
    if cls is None or lacking:
        raise SwapEngineUnavailable(
            f"{path!r} is not the expected engine: needs OnnxEngine with {', '.join(_SWAP_CALLS)} "
            f"(missing {', '.join(lacking) if cls else 'OnnxEngine'}; checked against "
            f"{SWAP_ENGINE_CHECKED_AT})")
    return cls(Path(models_dir), threads=threads)
