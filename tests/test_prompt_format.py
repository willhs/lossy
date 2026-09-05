"""Tests for prompt_format helpers."""

import pytest

from prompt_format import (
    format_duration_hms,
    is_leap_year,
    is_palindrome_word,
    ordinal_suffix,
    pluralize_count,
    sanitize_filename,
    slugify_title,
    wrap_bullet,
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


class TestIsLeapYear:
    @pytest.mark.parametrize(
        "year,expected",
        [
            (2000, True),   # divisible by 400
            (1900, False),  # divisible by 100 but not by 400
            (2024, True),   # divisible by 4, not by 100
            (2023, False),  # not divisible by 4
        ],
    )
    def test_years(self, year, expected):
        assert is_leap_year(year) is expected


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


class TestOrdinalSuffix:
    @pytest.mark.parametrize(
        "n,expected",
        [
            (0, "th"),
            (1, "st"),
            (2, "nd"),
            (3, "rd"),
            (4, "th"),
            (11, "th"),
            (12, "th"),
            (13, "th"),
            (14, "th"),
            (20, "th"),
            (21, "st"),
            (22, "nd"),
            (23, "rd"),
            (100, "th"),
            (101, "st"),
            (111, "th"),
            (112, "th"),
            (113, "th"),
            (121, "st"),
            (123, "rd"),
        ],
    )
    def test_suffixes(self, n, expected):
        assert ordinal_suffix(n) == expected

    @pytest.mark.parametrize("n", [-1, -11])
    def test_negative_raises(self, n):
        with pytest.raises(ValueError):
            ordinal_suffix(n)

    def test_non_int_raises(self):
        with pytest.raises(ValueError):
            ordinal_suffix("1")  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            ordinal_suffix(1.5)  # type: ignore[arg-type]


class TestSanitizeFilename:
    def test_replaces_multiple_unsafe_chars(self):
        assert sanitize_filename('shot: 01/review?') == "shot_ 01_review_"

    def test_clean_name_unchanged(self):
        assert sanitize_filename("scene-3 final") == "scene-3 final"

    def test_strips_leading_and_trailing_whitespace(self):
        assert sanitize_filename("  scene 3  ") == "scene 3"

    @pytest.mark.parametrize(
        "name,expected",
        [
            ('a<b', "a_b"),
            ('a>b', "a_b"),
            ('a:b', "a_b"),
            ('a"b', "a_b"),
            ('a/b', "a_b"),
            ('a\\b', "a_b"),
            ('a|b', "a_b"),
            ('a?b', "a_b"),
            ('a*b', "a_b"),
        ],
    )
    def test_each_unsafe_char(self, name, expected):
        assert sanitize_filename(name) == expected

    def test_empty_string(self):
        assert sanitize_filename("") == ""


class TestWrapBullet:
    def test_short_text_under_width_stays_on_one_line(self):
        assert wrap_bullet("a short line", 20) == ["a short line"]

    def test_text_fitting_exactly_on_one_line(self):
        assert wrap_bullet("a short line", 12) == ["a short line"]

    def test_wraps_into_exactly_two_lines(self):
        # "one two" is 7 chars; adding " three" would make 13 > 10.
        assert wrap_bullet("one two three", 10) == ["one two", "three"]

    def test_wraps_into_exactly_two_lines_multi_word(self):
        assert wrap_bullet("alpha beta gamma delta", 11) == ["alpha beta", "gamma delta"]

    def test_single_word_longer_than_width_overflows_whole(self):
        # A word longer than width is never broken mid-word.
        assert wrap_bullet("extraordinarily", 5) == ["extraordinarily"]

    def test_long_word_with_surrounding_words(self):
        assert wrap_bullet("hi supersupercalifragilistic hi", 10) == [
            "hi",
            "supersupercalifragilistic",
            "hi",
        ]

    def test_all_lines_no_wider_than_width(self):
        text = "the quick brown fox jumps over the lazy dog again and again"
        for width in (5, 10, 20, 40):
            for line in wrap_bullet(text, width):
                assert len(line) <= width

    def test_multiple_spaces_collapse(self):
        assert wrap_bullet("a    b   c", 5) == ["a b c"]

    def test_empty_text(self):
        assert wrap_bullet("", 10) == []
        assert wrap_bullet("   ", 10) == []

    def test_width_one(self):
        assert wrap_bullet("ab cd", 1) == ["ab", "cd"]

    def test_zero_width_raises(self):
        with pytest.raises(ValueError):
            wrap_bullet("text", 0)


class TestSlugifyTitle:
    def test_normal_title(self):
        assert slugify_title("The Empire Strikes Back") == "the-empire-strikes-back"

    def test_punctuation_is_replaced_with_hyphens(self):
        assert slugify_title("Star Wars: Episode IV") == "star-wars-episode-iv"
        assert slugify_title("Hello, World!") == "hello-world"

    def test_multiple_consecutive_spaces_collapse_to_one_hyphen(self):
        assert slugify_title("A   New   Hope") == "a-new-hope"

    def test_lowercases_input(self):
        assert slugify_title("Attack Of The Clones") == "attack-of-the-clones"

    def test_strips_leading_and_trailing_hyphens(self):
        assert slugify_title("  Return of the Jedi!  ") == "return-of-the-jedi"
        assert slugify_title("--Revenge--") == "revenge"

    def test_digits_are_kept(self):
        assert slugify_title("Episode 7") == "episode-7"

    def test_already_slugified_unchanged(self):
        assert slugify_title("the-force-awakens") == "the-force-awakens"

    @pytest.mark.parametrize("title", ["", "   ", "!!!", "---"])
    def test_empty_result(self, title):
        assert slugify_title(title) == ""
