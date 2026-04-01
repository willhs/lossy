"""Tests for eval.py comparison logic."""

import pytest
from eval import (
    CATEGORICAL_FIELDS,
    TEXT_FIELDS,
    compare_audio,
    compare_categorical,
    jaccard_similarity,
)


class TestCompareCategorical:
    def test_exact_match(self):
        assert compare_categorical("wide", "wide") == 1.0

    def test_case_insensitive(self):
        assert compare_categorical("Wide", "wide") == 1.0

    def test_mismatch(self):
        assert compare_categorical("wide", "close-up") == 0.0

    def test_both_empty(self):
        assert compare_categorical("", "") == 1.0

    def test_one_empty(self):
        assert compare_categorical("wide", "") == 0.0


class TestJaccardSimilarity:
    def test_identical(self):
        assert jaccard_similarity(["Music", "Orchestra"], ["Music", "Orchestra"]) == 1.0

    def test_disjoint(self):
        assert jaccard_similarity(["Music"], ["Explosion"]) == 0.0

    def test_partial_overlap(self):
        assert jaccard_similarity(["Music", "Orchestra"], ["Music", "Piano"]) == pytest.approx(1 / 3)

    def test_both_empty(self):
        assert jaccard_similarity([], []) == 1.0

    def test_one_empty(self):
        assert jaccard_similarity(["Music"], []) == 0.0

    def test_case_insensitive(self):
        assert jaccard_similarity(["Music"], ["music"]) == 1.0


class TestCompareAudio:
    def test_matching_bucket(self):
        orig = {"bucket": "music", "labels": ["Music", "Orchestra"]}
        recon = {"bucket": "music", "labels": ["Music", "Piano"]}
        result = compare_audio(orig, recon)
        assert result["bucket_match"] == 1.0
        assert result["label_similarity"] == pytest.approx(1 / 3)

    def test_different_bucket(self):
        orig = {"bucket": "music", "labels": ["Music"]}
        recon = {"bucket": "effects", "labels": ["Explosion"]}
        result = compare_audio(orig, recon)
        assert result["bucket_match"] == 0.0
        assert result["label_similarity"] == 0.0

    def test_both_none(self):
        result = compare_audio(None, None)
        assert result["bucket_match"] == 1.0

    def test_one_none(self):
        result = compare_audio({"bucket": "music", "labels": []}, None)
        assert result["bucket_match"] == 0.0


class TestFieldConstants:
    def test_categorical_fields_defined(self):
        assert "shot_type" in CATEGORICAL_FIELDS
        assert "camera_movement" in CATEGORICAL_FIELDS

    def test_text_fields_defined(self):
        assert "subjects" in TEXT_FIELDS
        assert "action" in TEXT_FIELDS
        assert "sound" in TEXT_FIELDS
        assert len(TEXT_FIELDS) == 7
