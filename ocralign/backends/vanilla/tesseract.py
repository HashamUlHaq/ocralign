"""
Tesseract adapter: image -> List[Word] (canonical schema).

Uses pytesseract.image_to_data, which runs the exact same recognition pass
as the hOCR output (same config, same engine) but returns word-level
text/bbox/confidence as a table, with no HTML parsing required.
"""

from __future__ import annotations

from typing import Any, List

import pytesseract
from pytesseract import Output

from ocralign.core.schema import Word

DEFAULT_CONFIG = "--psm 12 -c preserve_interword_spaces=1"


def extract_words(
    page_image: Any,
    tesseract_config: str = DEFAULT_CONFIG,
) -> List[Word]:
    """
    OCR a page image and return word boxes with normalized 0-1 coordinates.

    page_image: PIL Image, numpy array, or path accepted by pytesseract.
    """
    data = pytesseract.image_to_data(
        page_image,
        config=tesseract_config,
        output_type=Output.DICT,
    )

    # Page dimensions for normalization: level 1 is the page entry.
    try:
        page_idx = data["level"].index(1)
        page_w = float(data["width"][page_idx])
        page_h = float(data["height"][page_idx])
    except (ValueError, KeyError):
        # Fall back to the image size if no page-level row is present.
        page_w = float(getattr(page_image, "width", 0)) or 1.0
        page_h = float(getattr(page_image, "height", 0)) or 1.0

    words: List[Word] = []
    n = len(data["text"])
    for i in range(n):
        if data["level"][i] != 5:  # 5 = word level
            continue
        text = " ".join(data["text"][i].split()).strip()
        if not text:
            continue
        x0 = data["left"][i]
        y0 = data["top"][i]
        x1 = x0 + data["width"][i]
        y1 = y0 + data["height"][i]
        conf = float(data["conf"][i])
        words.append(
            Word(
                text=text,
                bbox=(x0 / page_w, y0 / page_h, x1 / page_w, y1 / page_h),
                confidence=conf if conf >= 0 else None,
            )
        )
    return words
