"""tools/face_swap.py: a shot is swapped on every frame or on none (research 0035's flicker).

The swap engine is faked: a test frame names its own faces (left edge and similarity to the
sheet), and "swapping" a face records it on the frame. No models, no video files.
"""

import os
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

import face_swap  # noqa: E402

IDENTITY = np.array([1.0, 0.0])


@dataclass
class _Face:
    box: tuple
    landmarks: float   # the fake's stand-in: the face's left edge
    sim: float


class _Engine:
    def detect(self, frame):
        return [_Face((x, 100, x + 80, 200), x, sim) for x, sim in frame["faces"]]

    def embed(self, frame, face):
        return np.array([face.sim, 0.0])

    def swap_aligned(self, frame, landmarks, identity):
        return {**frame, "swapped": landmarks}, None


def _swap(*frames):
    """Each frame is a list of (left edge, sim) faces; returns the left edge swapped per frame, or None."""
    out, _ = face_swap.swap_shot([{"faces": f} for f in frames], _Engine(), IDENTITY)
    return [f.get("swapped") for f in out]


def test_face_hovering_around_the_line_is_swapped_on_every_frame():
    got = _swap([(100, 0.05)], [(102, 0.12)], [(104, 0.08)], [(106, 0.11)], [(108, 0.02)])
    assert got == [100, 102, 104, 106, 108]


def test_face_that_never_clears_the_line_is_left_alone():
    got = _swap([(100, 0.05)], [(102, 0.09)], [(104, 0.01)])
    assert got == [None, None, None]


def test_shot_where_the_face_is_lost_partway_is_left_alone():
    got = _swap([(100, 0.5)], [(102, 0.5)], [], [(106, 0.5)])
    assert got == [None, None, None, None]


def test_the_same_person_is_swapped_when_another_face_outscores_them_for_a_frame():
    got = _swap([(100, 0.3), (600, 0.0)], [(102, 0.1), (602, 0.2)], [(104, 0.3), (604, 0.0)])
    assert got == [100, 102, 104]


def test_cut_is_the_biggest_change_near_each_expected_boundary():
    # Shot times say frames 10 and 20; the stitched video really cuts at 13 and 19.
    diffs = [1.0] * 30
    diffs[13], diffs[19] = 60.0, 40.0
    assert face_swap.find_cuts(diffs, [10, 20]) == [13, 19]
