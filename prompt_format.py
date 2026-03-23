"""Prompt formatting for video generation models.

Model-specific formatters that convert structured shot descriptions
into optimized text prompts for different video generation backends.
"""

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
            subjects = ", ".join(subjects)
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
