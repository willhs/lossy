"""Reference sheets for the LTX-2.5 Ingredients IC-LoRA (research 0029, arm I).

The adapter conditions every frame of a clip on ONE composite image (a "reference
sheet": characters, props and settings on a black background) and expects a prompt
in two parts. The model card's example prompts all use this form::

    ### Reference Sheet Description **Top Row Left (Character):** <what that panel
    shows> **Top Row Right (Character):** <...> ### Target Description <the clip>

so the panel descriptions name their position on the sheet and the model can tie
each described panel to the pixels. This module owns that one body of knowledge:
what a panel is, where panels sit, how the sheet is drawn, and how the prompt
refers to it. Everything here is pure (no pod, no network), so the graph and the
prompt can be checked without spending anything.

Sheets hold at most three panels, in a single row. The research note describes the
adapter as designed for 1-3 characters per composite; one row keeps the position
words to the ones the card uses ("Top Row Left", "Top Row Middle", "Top Row Right").
"""
from dataclasses import dataclass

import cv2
import numpy as np

MAX_PANELS = 3

# Position words by panel count. A lone panel is just "Top Row": the card's own
# single-character example is "**Top Row (Character):**".
_POSITIONS = {
    1: ("Top Row",),
    2: ("Top Row Left", "Top Row Right"),
    3: ("Top Row Left", "Top Row Middle", "Top Row Right"),
}

REFERENCE_HEADING = "### Reference Sheet Description"
TARGET_HEADING = "### Target Description"


@dataclass(frozen=True)
class SheetPanel:
    """One cell of the composite: an image plus the sentence that describes it."""

    kind: str          # the card's labels: "Character", "Prop", "Setting"
    description: str   # what the panel shows; the model matches it to the pixels
    image: np.ndarray  # BGR uint8, as cv2.imread returns

    def __post_init__(self):
        if not self.description.strip():
            raise ValueError("a sheet panel needs a description")
        if self.image.ndim != 3 or self.image.shape[2] != 3:
            raise ValueError(f"panel image must be HxWx3, got shape {self.image.shape}")


def panel_labels(n: int) -> tuple[str, ...]:
    """Position words for a sheet of n panels, left to right."""
    if n not in _POSITIONS:
        raise ValueError(f"a reference sheet holds 1 to {MAX_PANELS} panels, got {n}")
    return _POSITIONS[n]


def format_sheet_prompt(panels: list[SheetPanel], target: str) -> str:
    """The adapter's two-part prompt: the sheet described panel by panel, then the clip."""
    if not target.strip():
        raise ValueError("the target description is empty")
    labels = panel_labels(len(panels))
    sheet = " ".join(f"**{pos} ({p.kind}):** {p.description.strip()}"
                     for pos, p in zip(labels, panels))
    return f"{REFERENCE_HEADING} {sheet} {TARGET_HEADING} {target.strip()}"


def compose_sheet(panels: list[SheetPanel], width: int, height: int) -> np.ndarray:
    """Draw the panels left to right on a black canvas of the clip's frame size.

    The canvas has the clip's aspect ratio on purpose: ComfyUI's guide node
    resizes the image to the latent with a centre crop, so any other aspect would
    lose the edges of the outer panels. Each panel is scaled to fit its cell
    without distortion and centred.
    """
    labels = panel_labels(len(panels))
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    cell_w = width // len(labels)
    for i, panel in enumerate(panels):
        h, w = panel.image.shape[:2]
        scale = min(cell_w / w, height / h)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        resized = cv2.resize(panel.image, (nw, nh), interpolation=cv2.INTER_AREA)
        x0 = i * cell_w + (cell_w - nw) // 2
        y0 = (height - nh) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas
