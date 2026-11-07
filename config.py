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
    RunPodLtx2Strategy,
    RunPodLtx25Strategy,
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
#
# ENCODE_MODEL decision (2026-07-25, see task "Sticky wrong character names"):
# gemini-2.5-flash-lite is confirmed for the full Star Wars IV run. It matches
# the sw_r2_leia baseline that got the Leia hologram right, and a clean A/B on
# the BCS control clip (identical model, previous-shot-context propagation on
# vs. off) showed the earlier naming instability (Saul Goodman/Jimmy McGill/
# Howard Hamlin flipping across shots) was NOT caused by 2.5-flash-lite itself
# or by context propagation — it was the describe-pass prompt inviting Gemini
# to name characters at all. With the identity-stripped continuity context and
# reworded system prompt (see SYSTEM_PROMPT in encode.py), both the on and off
# runs produced zero character-name hallucinations. The star_wars_iv_v2 full
# encode used gemini-3.1-flash-lite-preview (EVAL_MODEL) instead; it must be
# re-encoded with ENCODE_MODEL before the real full run so the film is encoded
# with the same model throughout.
#
# Backend (2026-11-08): the AI Studio API-key path is project-throttled (429 on
# every model, "API requests per minute per region"). LOSSY_GEMINI_BACKEND=vertex
# sends the same calls through Vertex AI on the same GCP project instead, which
# has its own quota. Vertex does not serve gemini-2.5-flash-lite or the
# *-preview ids, so also set LOSSY_ENCODE_MODEL (e.g. gemini-3.1-flash-lite) --
# that is a model change, so check descriptions against the pinned model first.
ENCODE_MODEL = os.environ.get("LOSSY_ENCODE_MODEL", "gemini-2.5-flash-lite")
EVAL_MODEL = "gemini-3.1-flash-lite-preview"

GEMINI_BACKEND = os.environ.get("LOSSY_GEMINI_BACKEND", "aistudio")
VERTEX_PROJECT = os.environ.get("LOSSY_VERTEX_PROJECT", "gen-lang-client-0146943418")
VERTEX_LOCATION = os.environ.get("LOSSY_VERTEX_LOCATION", "global")


def gemini_client(timeout_ms: int | None = None):
    """Build a Gemini client for the configured backend (aistudio | vertex).

    aistudio needs GEMINI_API_KEY; vertex uses Application Default Credentials
    with VERTEX_PROJECT as the quota/billing project, so no key is needed.
    Exits with a message if the chosen backend is not usable.
    """
    from google import genai
    from google.genai import types

    http_options = types.HttpOptions(timeout=timeout_ms) if timeout_ms else None
    if GEMINI_BACKEND == "vertex":
        import google.auth

        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        creds = creds.with_quota_project(VERTEX_PROJECT)
        return genai.Client(vertexai=True, project=VERTEX_PROJECT, location=VERTEX_LOCATION,
                            credentials=creds, http_options=http_options)
    if GEMINI_BACKEND != "aistudio":
        raise SystemExit(f"Error: LOSSY_GEMINI_BACKEND must be 'aistudio' or 'vertex', got {GEMINI_BACKEND!r}")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        print("Get one at https://aistudio.google.com/apikey")
        raise SystemExit(1)
    return genai.Client(api_key=api_key, http_options=http_options)


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
        StrategySpec("runpod-ltx2", RunPodLtx2Strategy, supports_concurrent_audio=False),
        StrategySpec("runpod-ltx25", RunPodLtx25Strategy, supports_concurrent_audio=False),
    )
}

VIDEO_STRATEGY_NAMES = list(VIDEO_STRATEGIES)

CONCURRENT_AUDIO_STRATEGIES = frozenset(
    name for name, spec in VIDEO_STRATEGIES.items() if spec.supports_concurrent_audio
)

AUDIO_STRATEGIES = ["elevenlabs", "mmaudio", "runpod-mmaudio"]
MUSIC_STRATEGIES = ["musicgen", "runpod-musicgen"]
DECODE_AUDIO_STRATEGIES = AUDIO_STRATEGIES + MUSIC_STRATEGIES
