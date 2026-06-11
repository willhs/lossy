"""On-disk contract for the lossy pipeline.

Single source of truth for:
  * The ``shots.json`` v2 schema (see ``load_shots`` / ``save_shots``)
  * Filename and directory conventions for clips, audio, speech, and progress
    files (see the ``*_path`` / ``*_dir`` helpers)

Stages communicate by files on disk (ADR-002). Historically the v1/v2 format
check and ``f"{idx:04d}.mp4"`` string literals were reinvented in every
reader, so a schema tweak meant grepping five modules. This module removes
the duplication.

The two top-level encode artifacts:
  * ``shot_index.json`` — shot boundaries + keyframe filenames (stage 1, from
    PySceneDetect). Where the shots are.
  * ``shots.json`` — per-shot descriptions + the dialog timeline (stage 2).
    What the shots are: the compressed film itself.

v1 ``shots.json`` (a flat list of shots) is no longer supported on the
decode side. Encode stage 2 has emitted v2 since the format was introduced,
so the only places v1 still lives are stage 3's fixture tests.
"""

import json
import os


PROMPTS_FORMAT_VERSION = "v2"


# ---------------------------------------------------------------------------
# Top-level file paths
# ---------------------------------------------------------------------------

def shot_index_path(output_dir: str) -> str:
    return os.path.join(output_dir, "shot_index.json")


def shots_path(output_dir: str) -> str:
    return os.path.join(output_dir, "shots.json")


def characters_path(output_dir: str) -> str:
    return os.path.join(output_dir, "characters.json")


def decode_progress_path(output_dir: str, strategy: str) -> str:
    return os.path.join(output_dir, f"decode_progress_{strategy}.json")


def audio_progress_path(output_dir: str, strategy: str) -> str:
    return os.path.join(output_dir, f"audio_progress_{strategy}.json")


def speech_progress_path(output_dir: str) -> str:
    return os.path.join(output_dir, "speech_progress.json")


def reconstructed_path(output_dir: str, strategy: str, audio_strategy: str | None = None) -> str:
    """Final stitched video path, prefixed with the film name for VLC distinguishability."""
    film = os.path.basename(os.path.normpath(output_dir))
    suffix = f"+{audio_strategy}" if audio_strategy else ""
    return os.path.join(output_dir, f"{film}_reconstructed_{strategy}{suffix}.mp4")


# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------

def clips_dir(output_dir: str, strategy: str) -> str:
    return os.path.join(output_dir, "clips", strategy)


def audio_dir(output_dir: str, strategy: str) -> str:
    return os.path.join(output_dir, "audio", strategy)


def speech_dir(output_dir: str) -> str:
    return os.path.join(output_dir, "speech")


# ---------------------------------------------------------------------------
# Clip / audio / speech filename conventions
# ---------------------------------------------------------------------------
#
# Single-part shots use ``{idx:04d}.ext``. Split shots (long shots broken
# into N parts by a strategy) use ``{idx:04d}-{part:02d}.ext`` with part
# numbering starting at 1. Speech clips are one per global dialog line:
# ``{line_idx:04d}-{sub:02d}.mp3``, where sub is always 0 in v2.

def clip_filename(shot_idx: int, part: int | None = None, ext: str = ".mp4") -> str:
    if part is None:
        return f"{shot_idx:04d}{ext}"
    return f"{shot_idx:04d}-{part:02d}{ext}"


def clip_path(clips_dir_path: str, shot_idx: int, part: int | None = None,
              ext: str = ".mp4") -> str:
    return os.path.join(clips_dir_path, clip_filename(shot_idx, part, ext))


def audio_clip_filename(shot_idx: int, part: int | None = None,
                        ext: str = ".flac") -> str:
    return clip_filename(shot_idx, part, ext)


def audio_clip_path(audio_dir_path: str, shot_idx: int,
                    part: int | None = None, ext: str = ".flac") -> str:
    return os.path.join(audio_dir_path, audio_clip_filename(shot_idx, part, ext))


def speech_clip_filename(line_idx: int, sub_idx: int = 0) -> str:
    return f"{line_idx:04d}-{sub_idx:02d}.mp3"


def speech_clip_path(speech_dir_path: str, line_idx: int, sub_idx: int = 0) -> str:
    return os.path.join(speech_dir_path, speech_clip_filename(line_idx, sub_idx))


def clip_exists_for_shot(clips_dir_path: str, shot_idx: int,
                         ext: str = ".mp4") -> bool:
    """True if either the single-part clip or the first split part exists."""
    primary = clip_path(clips_dir_path, shot_idx, None, ext)
    first_part = clip_path(clips_dir_path, shot_idx, 1, ext)
    return os.path.exists(primary) or os.path.exists(first_part)


# ---------------------------------------------------------------------------
# shots.json (v2 format)
# ---------------------------------------------------------------------------
#
# Shape:
#   {
#     "format": "v2",
#     "shots": [ { "index", "start_s", "end_s", "duration_s",
#                  "description": { "shot_type", "camera_movement", ...,
#                                   "sound" },
#                  "dialogue": [...],
#                  "camera_motion_detected": ..., "audio_detected": ...,
#                  "temporal_segments"?: [...] },
#                ... ],
#     "dialog":  [ { "text", "start_s", "end_s" }, ... ]
#   }
#
# ``shots`` is the per-shot list. ``dialog`` is a global subtitle timeline
# used by speech TTS so that lines spanning shot boundaries are only spoken
# once.


def load_shots(output_dir: str) -> tuple[list[dict], list[dict]]:
    """Read shots.json (v2) and return ``(shots, dialog)``.

    Raises ``FileNotFoundError`` if the file is missing, ``ValueError`` if
    it is not in v2 format. v1 flat-list shots are no longer supported on
    the decode side — re-run ``encode stage2`` to regenerate.
    """
    path = shots_path(output_dir)
    with open(path) as f:
        raw = json.load(f)

    if not (isinstance(raw, dict) and raw.get("format") == PROMPTS_FORMAT_VERSION):
        raise ValueError(
            f"{path}: expected v2 shots.json "
            f"({{'format': 'v2', 'shots': [...], 'dialog': [...]}}). "
            f"Re-run `python encode.py stage2 {output_dir}` to regenerate."
        )

    return raw["shots"], raw.get("dialog", [])


def save_shots(output_dir: str, shots: list[dict], dialog: list[dict]) -> str:
    path = shots_path(output_dir)
    data = {"format": PROMPTS_FORMAT_VERSION, "shots": shots, "dialog": dialog}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path
