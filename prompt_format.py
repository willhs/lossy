"""Prompt formatting for video generation models.

Model-specific formatters that convert structured shot descriptions
into optimized text prompts for different video generation backends.
"""


def format_duration_hms(seconds: float) -> str:
    """Format a duration in seconds as a human-readable timestamp.

    Returns "H:MM:SS" for durations of an hour or more (minutes and seconds
    zero-padded to two digits) and "M:SS" below an hour, matching the
    timestamp convention players use:

      0        -> "0:00"
      3.6      -> "0:04"
      62       -> "1:02"
      3730     -> "1:02:10"

    Sub-second precision is intentionally dropped: callers use this for
    context lines and progress output, not for frame-accurate timing.
    Raises ValueError for negative durations, which callers should never
    produce (shot durations come from end_s - start_s).
    """
    if seconds < 0:
        raise ValueError(f"duration must be non-negative, got {seconds!r}")
    total = round(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def pluralize_count(noun: str, count: int) -> str:
    """Render '<count> <noun>' with a naive plural for counts other than 1.

    Purely additive helper for composing subject counts in prompts:

      pluralize_count("shot", 1)  -> "1 shot"
      pluralize_count("shot", 3)  -> "3 shots"

    Pluralization is the plain 's' suffix (no irregular forms), which is
    fine for the nouns used here; callers with irregular nouns should pass
    an already-pluralized string and a count of 1 or handle it themselves.
    """
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def is_leap_year(year: int) -> bool:
    """Check whether a year is a Gregorian leap year.

    Purely additive helper, not wired into any formatter:

      is_leap_year(2000) -> True   # divisible by 400
      is_leap_year(2024) -> True   # divisible by 4, not by 100
      is_leap_year(1900) -> False  # divisible by 100 but not by 400
      is_leap_year(2023) -> False  # not divisible by 4
    """
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def is_palindrome_word(word: str) -> bool:
    """Case-insensitively check whether a single alphabetic word is a palindrome.

    Purely additive helper, not wired into any formatter:

      is_palindrome_word("level")   -> True
      is_palindrome_word("Level")   -> True
      is_palindrome_word("droid")   -> False
      is_palindrome_word("a")       -> True

    Only single alphabetic words count: anything containing spaces,
    punctuation, digits, or non-letter characters returns False, as does
    an empty string. Case is ignored, so "Anna" is a palindrome.
    """
    if not isinstance(word, str) or not word.isalpha():
        return False
    lowered = word.lower()
    return lowered == lowered[::-1]


def ordinal_suffix(n: int) -> str:
    """Return the ordinal suffix for a non-negative integer.

    Purely additive helper for composing position labels ("Part 2 of 5"):

      ordinal_suffix(1)   -> "st"
      ordinal_suffix(2)   -> "nd"
      ordinal_suffix(3)   -> "rd"
      ordinal_suffix(4)   -> "th"
      ordinal_suffix(11)  -> "th"   # 11th-13th break the 1st/2nd/3rd rule
      ordinal_suffix(21)  -> "st"
      ordinal_suffix(111) -> "th"

    Only the last two digits matter, except teens (11-13) always take
    "th". Raises ValueError for negative or non-integer input, which
    callers should never produce (positions come from enumerate-style
    indices).
    """
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise ValueError(f"ordinal position must be a non-negative int, got {n!r}")
    if 11 <= n % 100 <= 13:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def wrap_bullet(text: str, width: int) -> list[str]:
    """Word-wrap text into lines no wider than ``width`` using greedy wrapping.

    Purely additive helper, not wired into any formatter or call site:

      wrap_bullet("a short line", 20)  -> ["a short line"]
      wrap_bullet("one two three four", 7)
          -> ["one two", "three", "four"]
      wrap_bullet("extraordinarily", 5) -> ["extraordinarily"]

    Words are packed onto the current line until adding the next word would
    exceed ``width``. A single word longer than ``width`` is placed on its
    own line and overflows -- it is never broken mid-word. Multiple spaces
    between words collapse to one. Raises ValueError for width < 1.
    """
    if width < 1:
        raise ValueError(f"width must be at least 1, got {width!r}")
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        if len(current) + 1 + len(word) <= width:
            current += " " + word
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def sanitize_filename(name: str) -> str:
    """Strip characters that are unsafe in filenames from a name.

    Purely additive helper, not wired into any formatter or call site:

      sanitize_filename('shot: 01/review?') -> 'shot_ 01_review_'
      sanitize_filename('  scene 3  ')     -> 'scene 3'
      sanitize_filename('scene-3')         -> 'scene-3'

    Any of the characters <>:"/\\|?* is replaced with an underscore, and
    leading/trailing whitespace is stripped. Everything else passes through
    unchanged; safe characters (letters, digits, spaces, hyphens, etc.) are
    never touched.
    """
    for ch in '<>:"/\\|?*':
        name = name.replace(ch, "_")
    return name.strip()


CAMERA_TERMS = {
    "slow zoom out": "slow dolly out",
    "slow zoom in": "slow dolly in",
    "zoom out": "dolly out",
    "zoom in": "dolly in",
    "follows": "tracking shot follows",
    "moves left": "pan left",
    "moves right": "pan right",
    "moves up": "crane up",
    "moves down": "crane down",
    "shaky": "handheld",
    "smooth movement": "steadicam",
}


def format_prompt(entry: dict) -> str:
    """Convert a structured prompt entry into a flat text prompt for video generation.

    Combines the description fields into a single cinematic prompt string
    that video generation models respond well to.
    """
    desc = entry["description"]

    parts = []

    # Lead with shot type and camera movement
    shot_type = desc.get("shot_type", "")
    camera = desc.get("camera_movement", "")
    if shot_type and camera:
        parts.append(f"Cinematic {shot_type} shot, {camera}.")
    elif shot_type:
        parts.append(f"Cinematic {shot_type} shot.")

    # Core content: action is the most important for video gen
    action = desc.get("action", "")
    if action:
        parts.append(action)

    # Subjects if not already covered by action
    subjects = desc.get("subjects", "")
    if subjects:
        if isinstance(subjects, list):
            subjects = ", ".join(subjects)
        # Only add if action doesn't already describe subjects well
        if len(str(action)) < 50:
            parts.append(subjects)

    # Visual style
    lighting = desc.get("lighting", "")
    if lighting:
        parts.append(lighting)

    palette = desc.get("color_palette", "")
    if palette:
        if isinstance(palette, list):
            palette = ", ".join(palette)
        parts.append(f"Color palette: {palette}.")

    mood = desc.get("mood", "")
    if mood:
        parts.append(f"Mood: {mood}.")

    setting = desc.get("setting", "")
    if setting:
        parts.append(setting)

    return " ".join(parts)


def _subject_to_str(s) -> str:
    """Coerce a subject entry to text. Gemini sometimes emits subjects as
    dicts (e.g. {"subject": ..., "description": ...}) instead of plain
    strings; flatten those into a single descriptive clause."""
    if isinstance(s, str):
        return s
    if isinstance(s, dict):
        for key in ("description", "subject", "name", "text"):
            if s.get(key):
                return str(s[key])
        return ", ".join(str(v) for v in s.values() if v)
    return str(s)


def _format_prompt_wan(entry: dict) -> str:
    """Wan-optimized prompt: Subject > Action > Camera > Style.

    Uses professional cinematography vocabulary, no metadata labels,
    front-loaded content. Targets ~150-200 words to stay within
    Wan's T5 encoder sweet spot (~320 tokens).
    """
    desc = entry["description"]
    parts = []

    # 1. Subject (front-loaded for T5 attention)
    subjects = desc.get("subjects", "")
    if subjects:
        if isinstance(subjects, list):
            subjects = ", ".join(_subject_to_str(s) for s in subjects)
        elif isinstance(subjects, dict):
            subjects = _subject_to_str(subjects)
        parts.append(subjects.rstrip(".") + ".")

    # 2. Action (core content)
    action = desc.get("action", "")
    if action:
        parts.append(action)

    # 3. Camera (professional terms)
    shot_type = desc.get("shot_type", "")
    camera = desc.get("camera_movement", "")
    if camera:
        camera_lower = camera.lower()
        for casual, pro in CAMERA_TERMS.items():
            if casual in camera_lower:
                camera = camera_lower.replace(casual, pro)
                break
    if shot_type and camera:
        parts.append(f"{shot_type.title()} shot, {camera}.")
    elif shot_type:
        parts.append(f"{shot_type.title()} shot.")
    elif camera:
        parts.append(f"{camera}.")

    # 4. Style/Atmosphere (no labels, just descriptive text)
    setting = desc.get("setting", "")
    if setting:
        parts.append(setting)

    lighting = desc.get("lighting", "")
    if lighting:
        parts.append(lighting)

    palette = desc.get("color_palette", "")
    if palette:
        if isinstance(palette, list):
            palette = " and ".join(palette)
        parts.append(f"{palette} tones.")

    mood = desc.get("mood", "")
    if mood:
        parts.append(f"{mood} atmosphere.")

    return " ".join(parts)


def vary_prompt_for_part(prompt: str, part_index: int, total_parts: int) -> str:
    """Fallback: append a generic temporal progression cue for split-shot prompts.

    Used when temporal_segments are not available in the encoded entry.
    Returns prompt unchanged if total_parts <= 1.
    """
    if total_parts <= 1:
        return prompt
    if part_index == 0:
        cue = "Beginning of the action."
    elif part_index == total_parts - 1:
        cue = "The action concludes."
    else:
        cue = "The action continues."
    return f"{prompt} {cue}"


def _format_prompt_seedance(entry: dict) -> str:
    """Seedance-optimized prompt: ~30-60 words, single action, intensity adverbs."""
    desc = entry["description"]
    parts = []

    # Subject + single action verb (Seedance responds best to concise actions)
    subjects = desc.get("subjects", "")
    if subjects:
        if isinstance(subjects, list):
            subjects = subjects[0] if subjects else ""
        parts.append(subjects.rstrip("."))

    action = desc.get("action", "")
    if action:
        # Take just the first sentence for brevity
        first_sentence = action.split(".")[0].strip()
        if first_sentence:
            parts.append(first_sentence.rstrip(".") + ".")

    # Camera as a brief modifier
    camera = desc.get("camera_movement", "")
    if camera and camera.lower() != "static":
        parts.append(camera.rstrip(".") + ".")

    # One atmosphere phrase combining mood + setting
    mood = desc.get("mood", "")
    setting = desc.get("setting", "")
    if mood and setting:
        parts.append(f"{mood.split(',')[0].strip()} {setting.rstrip('.')}")
    elif setting:
        parts.append(setting)
    elif mood:
        parts.append(mood.split(",")[0].strip())

    return " ".join(parts)


# Character names must not reach the video model.
#
# Naming a character invites the model to reconstruct it from world knowledge
# instead of from the compressed description -- the effect research 0021
# documents, and the one that undermines the reconstruction claim on a film
# this heavily memorised. The names are kept in the *encode* (shots.json
# `subjects`) on purpose, because stage 3 text-matches the TMDB cast against
# that text to work out who is in each shot. They are stripped here instead,
# at the last moment before the prompt goes to the model, so character
# assignment keeps working and the model still never sees a name.
#
# Measured on the star_wars_iv_v2 encode: 389 of 2069 shots (18.8%) name a
# character in `subjects`, but 1828 (88.4%) had a name reaching the model,
# because the identity block prepended "<display_name>: <description>" to
# every shot with a mapped character.

def slugify_title(title: str) -> str:
    """Slugify a title into a URL/filename-friendly string.

    Purely additive helper, not wired into any formatter or call site:

      slugify_title("The Empire Strikes Back")  -> "the-empire-strikes-back"
      slugify_title("Star Wars: Episode IV")    -> "star-wars-episode-iv"
      slugify_title("  A   New   Hope!  ")      -> "a-new-hope"

    Lowercases the title, replaces every run of non-alphanumeric characters
    (spaces, punctuation, etc.) with a single hyphen, and strips leading and
    trailing hyphens. Empty or all-punctuation input yields an empty string.
    """
    import re

    slug = re.sub(r"[^a-z0-9]+", "-", title.lower())
    return slug.strip("-")


def strip_character_names(text: str, names) -> str:
    """Remove character names from a composed prompt, keeping it readable.

    Handles the three forms the encode actually produces:
      "A golden humanoid droid (C-3PO) and..."  -> parenthetical gloss
      "A golden humanoid droid, C-3PO, stands"  -> comma appositive
      "the golden droid C-3PO is visible"       -> bare, after a noun

    The description around the name is what carries the identity, so deleting
    the name leaves the shot fully specified.
    """
    import re

    if not text:
        return text
    # Longest first, so "Luke Skywalker" goes before "Luke" and never strands
    # a surname.
    ordered = sorted({n for n in names if n}, key=len, reverse=True)
    if not ordered:
        return text
    alt = "|".join(re.escape(n) for n in ordered)
    # Nothing to do -- and importantly, no tidying either. The cleanup below
    # would otherwise recapitalise and reflow prompts that never contained a
    # name, silently changing text it has no business touching.
    if not re.search(rf"\b(?:{alt})\b", text):
        return text

    # A parenthetical that is only a name is a gloss -- drop it whole.
    text = re.sub(rf"\s*\(\s*(?:{alt})\s*\)", "", text)
    # Comma appositive: drop the name and both its commas, so "a droid,
    # C-3PO, stands" reads "a droid stands" rather than "a droid, stands".
    # This also collapses list items cleanly.
    text = re.sub(rf",\s*(?:{alt})\s*,", " ", text)
    # A name governed by a preposition would otherwise strand it, leaving
    # "similar in appearance to but silver".
    text = re.sub(rf"\b(?:to|like|as|of|beside|behind|near)\s+(?:{alt})\b", "", text)
    # The identity block's "<name>: <description>" label.
    text = re.sub(rf"\b(?:{alt})\s*:\s*", "", text)
    # Anything left over.
    text = re.sub(rf"\b(?:{alt})\b", "", text)

    # Tidy the seams the deletions leave behind.
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = re.sub(r"([,;:])\1+", r"\1", text)
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"^\s*[:,;]\s*", "", text)
    # Canonical descriptions are written "C-3PO is a tall golden droid", so
    # removing the name strands the copula. Drop it at any sentence start and
    # recapitalise, turning "is a tall golden droid" into "A tall golden droid".
    text = re.sub(r"(^|(?<=\. ))\s*(?:is|are|was|were)\s+", "", text)
    text = re.sub(r"(^|(?<=\. ))([a-z])",
                  lambda m: m.group(1) + m.group(2).upper(), text)
    return text.strip().strip(",").strip()
