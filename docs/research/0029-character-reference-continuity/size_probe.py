"""Measure what one character reference image costs in the encoded payload.

Free, local, no API. Takes photographic crops from an existing research
contact sheet (real LTX-2.5 render frames, so texture and noise are
representative of the images a sheet generator would produce), resizes them
to the sizes a reference-conditioned video model actually consumes, and
encodes them at a few WebP/AVIF qualities.

Usage: uv run python docs/research/0029-character-reference-continuity/size_probe.py
"""

import io
import statistics
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).parent
SHEET = HERE.parent / "0025-video-model-refresh-nov-2026" / "contact_sheet_1.jpg"

# (label, output size). 512x512 is a single head-and-shoulders ref; a 2x2
# four-view sheet at 1024x1024 is what an "Ingredients"-style IC-LoRA takes.
TARGETS = [("single 512x512", (512, 512)), ("single 768x768", (768, 768)),
           ("4-view sheet 1024x1024", (1024, 1024))]
CODECS = [("webp", 60), ("webp", 80), ("avif", 40), ("avif", 60)]


def crops(img: Image.Image, n: int = 8) -> list[Image.Image]:
    """n square crops spread across the sheet (avoids the label strip)."""
    w, h = img.size
    side = min(w, h) // 3
    out = []
    for i in range(n):
        x = (w - side) * i // max(n - 1, 1)
        y = (h - side) * ((i * 3) % n) // max(n - 1, 1)
        out.append(img.crop((x, y, x + side, y + side)))
    return out


def encoded_bytes(img: Image.Image, fmt: str, q: int) -> int:
    buf = io.BytesIO()
    img.save(buf, format=fmt.upper(), quality=q)
    return buf.tell()


def main() -> int:
    src = Image.open(SHEET).convert("RGB")
    samples = crops(src)
    print(f"source: {SHEET.name} {src.size}, {len(samples)} crops")
    for label, size in TARGETS:
        if size[0] > 512:
            # Larger targets: tile 4 crops into a 2x2 sheet, as a real
            # multi-view sheet would be, rather than upscaling one crop.
            half = (size[0] // 2, size[1] // 2)
            imgs = []
            for k in range(0, len(samples) - 3, 2):
                sheet = Image.new("RGB", size)
                for j, c in enumerate(samples[k:k + 4]):
                    sheet.paste(c.resize(half, Image.LANCZOS), ((j % 2) * half[0], (j // 2) * half[1]))
                imgs.append(sheet)
            if label.startswith("single"):
                imgs = [c.resize(size, Image.LANCZOS) for c in samples]
        else:
            imgs = [c.resize(size, Image.LANCZOS) for c in samples]
        for fmt, q in CODECS:
            sizes = [encoded_bytes(i, fmt, q) for i in imgs]
            print(f"{label:24s} {fmt} q{q:<3d} median {statistics.median(sizes) / 1024:6.1f} KB"
                  f"  (min {min(sizes) / 1024:.1f}, max {max(sizes) / 1024:.1f}, n={len(sizes)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
