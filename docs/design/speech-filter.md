---
type: spec
spec_id: SPEC-100
status: Active
purpose: Requirements for filtering speech/dialogue cues from MMAudio sound descriptions.
---

# Speech-Cue Filter for MMAudio

**ID**: SPEC-100

## Overview

MMAudio is an SFX/ambiance model that produces poor output when sound descriptions include speech, voice, or dialogue references. The pipeline has a separate ElevenLabs TTS track for speech. The filter preprocesses sound descriptions before MMAudio, stripping speech cues while preserving ambient, SFX, and music content.

## Keyword Detection

<!-- REQ-001 to REQ-009 -->
**REQ-001**: The filter shall detect "voice" and "voices" as speech keywords.
**REQ-002**: The filter shall detect "speaking", "speaks", "spoken", and "speech" as speech keywords.
**REQ-003**: The filter shall detect "dialogue", "talking", "talks", and "conversation" as speech keywords.
**REQ-004**: The filter shall detect "says" and "saying" as speech keywords.
**REQ-005**: The filter shall detect "shout/shouting", "whisper/whispering", "scream/screaming", and "yell/yelling" as speech keywords.
**REQ-006**: The filter shall detect "murmur/murmuring", "narrate/narrates/narrating/narration" as speech keywords.
**REQ-007**: The filter shall detect "vocal" as a speech keyword.
**REQ-008**: Keyword detection shall be case-insensitive.
**REQ-009**: The filter shall NOT flag non-speech audio terms: "breathing", "beeps", "chirps", "footsteps", "hum", "explosion".

## Filtering Behavior

<!-- REQ-010 to REQ-019 -->
**REQ-010**: Descriptions containing no speech keywords shall pass through unchanged.
**REQ-011**: Descriptions containing only speech cues shall return None.
**REQ-012**: Empty input or input where the filtered remainder is fewer than 5 characters shall return None.
**REQ-013**: Mixed descriptions shall preserve SFX/ambient/music content and remove speech content.
**REQ-014**: When speech is the main subject of a clause, the entire clause shall be dropped rather than partially removed.
**REQ-015**: When a speech keyword appears as a comma-separated list item, only that item shall be removed; other list items shall be preserved.

## Structural Integrity

<!-- REQ-020 to REQ-029 -->
**REQ-020**: The filter shall not leave dangling adjectives from possessive noun phrases (e.g., "C-3PO's anxious," with no noun).
**REQ-021**: "and [possessive] [adjectives] voice [verb]" constructions shall be removed while preserving the content before "and".
**REQ-022**: Clause boundaries (". ", "; ", ", accompanied by", ", followed by", etc.) shall be respected; content in non-speech clauses shall be preserved.
**REQ-023**: When filtering removes the start of the text, the first letter of the result shall be capitalized.
**REQ-024**: Punctuation artifacts (double commas, semicolon-comma sequences, double periods, orphaned "The sound of") shall be cleaned up.
