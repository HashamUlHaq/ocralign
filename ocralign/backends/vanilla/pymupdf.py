"""
PyMuPDF adapter: born-digital PDF page -> List[Word] (canonical schema).

No OCR involved; word boxes come straight from the PDF text layer.
"""

from __future__ import annotations

from typing import List

import fitz  # PyMuPDF

from ocralign.core.schema import Word


def extract_words(page: "fitz.Page") -> List[Word]:
    """
    Return the page's text-layer words with normalized 0-1 coordinates.
    """
    rect = page.rect
    page_w = float(rect.width) or 1.0
    page_h = float(rect.height) or 1.0

    words: List[Word] = []
    for x0, y0, x1, y1, text, *_ in page.get_text("words"):
        text = " ".join(text.split()).strip()
        if not text:
            continue
        words.append(
            Word(
                text=text,
                bbox=(x0 / page_w, y0 / page_h, x1 / page_w, y1 / page_h),
                confidence=None,
            )
        )
    return words
