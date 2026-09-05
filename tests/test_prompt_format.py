"""Tests for prompt_format helpers."""

import pytest

from prompt_format import format_duration_hms


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
