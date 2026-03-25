"""Shared dataclasses for clip results."""

from dataclasses import dataclass


@dataclass
class ClipResult:
    """Result of generating a single video clip."""
    path: str
    actual_duration_s: float
    cost: float


@dataclass
class AudioClipResult:
    """Result of generating a single audio clip."""
    path: str
    actual_duration_s: float
    cost: float


@dataclass
class SpeechClipResult:
    """Result of generating a single speech clip."""
    path: str
    duration_s: float
    offset_s: float  # shot-relative offset for placement
    cost: float
