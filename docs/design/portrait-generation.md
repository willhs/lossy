---
type: spec
spec_id: SPEC-210
status: Active
purpose: Requirements for character portrait generation and character-to-shot mapping.
---

# Portrait Generation

**ID**: SPEC-210

## Overview

Before VACE clip generation, the decode flow generates a canonical portrait image per character from `characters.json`. Portraits are stored as PNG files and uploaded to the RunPod pod as VACE reference images. A character-to-shot mapping enables the decode loop to look up which portrait(s) to use for each shot.

## Portrait Generation

<!-- REQ-001 to REQ-009 -->
**REQ-001**: The system shall generate one portrait image per character in `characters.json`.
**REQ-002**: Portraits shall be stored as PNG files in `{output_dir}/characters/{name}.png`.
**REQ-003**: If a portrait file already exists, it shall be reused without regeneration.
**REQ-004**: The return value shall be a dict mapping character name to portrait file path.
**REQ-005**: A failed portrait generation shall not abort the entire run; the character shall be skipped.

## Character-to-Shot Mapping

<!-- REQ-010 to REQ-019 -->
**REQ-010**: `build_character_shot_map` shall return a dict mapping shot index to a list of character names.
**REQ-011**: A shot with multiple characters shall list all of them.
**REQ-012**: A shot with no characters shall not appear in the mapping.
**REQ-013**: Character order in each shot's list shall match the order characters appear in `characters.json` (most prominent first).
