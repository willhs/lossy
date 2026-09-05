"""Tests for prompt_format helpers."""

import pytest

from prompt_format import (
    format_duration_hms,
    is_palindrome_word,
    pluralize_count,
)


class TestFormatDurationHms:
    @pytest.mark.parametrize(
        "seconds,expected",
        [
            (0, "0:00"),
            (1, "0:01"),
            (3.6, "0:04"),  # rounds to nearest second
            (9.4, "0:09"),
            (10, "0:10"),
            (59, "0:59"),
            (60, "1:00"),
            (62, "1:02"),
            (600, "10:00"),
            (3599, "59:59"),
            (3600, "1:00:00"),
            (3662, "1:01:02"),
            (3730, "1:02:10"),
            (36000, "10:00:00"),
            (123.49, "2:03"),  # sub-second precision dropped
            (123.51, "2:04"),
        ],
    )
    def test_formats(self, seconds, expected):
        assert format_duration_hms(seconds) == expected

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            format_duration_hms(-0.1)


class TestPluralizeCount:
    @pytest.mark.parametrize(
        "noun,count,expected",
        [
            ("shot", 0, "0 shots"),
            ("shot", 1, "1 shot"),
            ("shot", 2, "2 shots"),
            ("droid", 0, "0 droids"),
            ("droid", 1, "1 droid"),
            ("droid", 2, "2 droids"),
        ],
    )
    def test_counts(self, noun, count, expected):
        assert pluralize_count(noun, count) == expected


class TestIsPalindromeWord:
    @pytest.mark.parametrize(
        "word,expected",
        [
            # A straightforward palindrome.
            ("level", True),
            # A non-palindrome.
            ("droid", False),
            # Single characters read the same both ways.
            ("a", True),
            ("Z", True),
            # Case-insensitive.
            ("Level", True),
            ("Anna", True),
            ("Droid", False),
            # Longer palindromes and near-misses.
            ("racecar", True),
            ("racecars", False),
            # Empty string is not a word.
            ("", False),
            # Not a single alphabetic word.
            ("a a", False),
            ("level!", False),
            ("a1a", False),
        ],
    )
    def test_words(self, word, expected):
        assert is_palindrome_word(word) == expected

    def test_returns_bool(self):
        assert is_palindrome_word("level") is True
        assert is_palindrome_word("droid") is False
