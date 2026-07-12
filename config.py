"""Shared config surface: .env loading, Gemini model constants, strategy registry.

Single source of truth so CLIs don't each reimplement the same env-file
parsing, hardcode model ids, or duplicate strategy name lists.
"""

import os
from dataclasses import dataclass

from strategies_video import (
    FalSeedanceProStrategy,
    FalSeedanceStrategy,
    GenerationStrategy,
    ReplicateWanStrategy,
    RunPodVaceStrategy,
    RunPodWan22Strategy,
    RunPodWanEnrichedStrategy,
    RunPodWanStrategy,
)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_env() -> None:
    """Load .env into os.environ, without overwriting already-set vars.

    Call this once from a CLI entry point (a script's ``main()``), never
    from library functions — those get exercised directly by tests, and a
    disk read there would silently re-populate variables a test just
    deleted via monkeypatch, defeating test isolation.
    """
    env_path = os.path.join(ROOT_DIR, ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path) as ef:
        for line in ef:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


# Gemini model ids. These are two intentionally different models for two
# different concerns: ENCODE_MODEL is the pinned, stable model used for
# production shot description at encode time; EVAL_MODEL is a newer preview
# model used only by the offline eval harness to score reconstruction
# quality, where staying on the latest model matters more than pinning.
ENCODE_MODEL = "gemini-2.5-flash-lite"
EVAL_MODEL = "gemini-3.1-flash-lite-preview"


@dataclass(frozen=True)
class StrategySpec:
    name: str
    cls: type[GenerationStrategy]
    supports_concurrent_audio: bool


VIDEO_STRATEGIES: dict[str, StrategySpec] = {
    spec.name: spec
    for spec in (
        StrategySpec("replicate-wan", ReplicateWanStrategy, supports_concurrent_audio=False),
        StrategySpec("fal-seedance", FalSeedanceStrategy, supports_concurrent_audio=False),
        StrategySpec("fal-seedance-pro", FalSeedanceProStrategy, supports_concurrent_audio=False),
        StrategySpec("runpod-wan", RunPodWanStrategy, supports_concurrent_audio=True),
        StrategySpec("runpod-wan22", RunPodWan22Strategy, supports_concurrent_audio=True),
        StrategySpec("runpod-wan-enriched", RunPodWanEnrichedStrategy, supports_concurrent_audio=True),
        StrategySpec("runpod-vace", RunPodVaceStrategy, supports_concurrent_audio=True),
    )
}

VIDEO_STRATEGY_NAMES = list(VIDEO_STRATEGIES)

CONCURRENT_AUDIO_STRATEGIES = frozenset(
    name for name, spec in VIDEO_STRATEGIES.items() if spec.supports_concurrent_audio
)

AUDIO_STRATEGIES = ["elevenlabs", "mmaudio", "runpod-mmaudio"]
MUSIC_STRATEGIES = ["musicgen", "runpod-musicgen"]
DECODE_AUDIO_STRATEGIES = AUDIO_STRATEGIES + MUSIC_STRATEGIES
