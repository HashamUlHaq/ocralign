"""
Resolve character spans / substrings in rendered text back to page
coordinates, for drawing overlays on the original document.

All returned boxes are (x0, y0, x1, y1) fractions of page width/height
(0-1, origin top-left). Convert to pixels for any render size with
to_pixels(); this works identically for a PIL debug image, a react-pdf
page, or anything else that knows its rendered page dimensions.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from ocralign.core.schema import BBox, Document, Page, Word


def locate(page: Page, char_start: int, char_end: int) -> List[BBox]:
    """
    Map a character span of page.text to bounding box(es) on the page.

    Inclusion is overlap-based: any word whose span intersects
    [char_start, char_end) is included, so spans that clip mid-word
    (e.g. from tokenizer boundary mismatches) still highlight the whole
    word. Returns one merged box per visual line (a span wrapping two
    lines yields two boxes). Empty list if nothing matches.
    """
    if char_end <= char_start:
        return []

    hits = [
        w for w in page.words
        if w.char_start is not None and w.char_end is not None
        and w.char_start < char_end and w.char_end > char_start
    ]
    if not hits:
        return []

    boxes: dict = {}  # line_no -> merged bbox
    for w in hits:
        key = w.line_no
        if key in boxes:
            x0, y0, x1, y1 = boxes[key]
            boxes[key] = (min(x0, w.x0), min(y0, w.y0), max(x1, w.x1), max(y1, w.y1))
        else:
            boxes[key] = w.bbox

    return [boxes[k] for k in sorted(boxes, key=lambda k: (k is None, k))]


def locate_substring(
    page: Page,
    query: str,
    occurrence: Optional[int] = None,
) -> List[List[BBox]]:
    """
    Convenience wrapper: find `query` in page.text and resolve each
    occurrence to boxes. Returns a list of box-lists (one entry per
    occurrence); pass occurrence=n to get just [nth match].

    Matching is whitespace-flexible: any whitespace in the query matches
    any run of whitespace (including newlines) in the layout text, since
    the renderer inserts alignment spacing that literal matching would
    trip over.

    Prefer locate() with exact offsets when you have them (e.g. from an
    NER span) — substrings can repeat and are ambiguous.
    """
    tokens = query.split()
    if not tokens:
        return []
    pattern = re.compile(r"\s+".join(re.escape(t) for t in tokens))

    results: List[List[BBox]] = []
    for m in pattern.finditer(page.text):
        results.append(locate(page, m.start(), m.end()))

    if occurrence is not None:
        return [results[occurrence]] if 0 <= occurrence < len(results) else []
    return results


def locate_in_document(
    doc: Document,
    char_start: int,
    char_end: int,
    add_marker: bool = True,
) -> List[Tuple[int, BBox]]:
    """
    Like locate(), but for offsets into Document.text(add_marker=...)
    (the concatenated multi-page string). Returns (page_number, bbox)
    tuples so callers know which page each box belongs on.

    add_marker must match the value used to produce the text the
    offsets refer to.
    """
    results: List[Tuple[int, BBox]] = []
    offset = 0
    for p in doc.pages:
        head = f"-- Page {p.page_number} --\n" if add_marker else ""
        tail = "\n\n" if add_marker else ""
        body_start = offset + len(head)
        body_end = body_start + len(p.text)
        # Overlap of the query span with this page's body
        s = max(char_start, body_start)
        e = min(char_end, body_end)
        if s < e:
            for bbox in locate(p, s - body_start, e - body_start):
                results.append((p.page_number, bbox))
        offset = body_end + len(tail)
    return results


def to_pixels(
    bbox: BBox,
    page_width: float,
    page_height: float,
) -> Tuple[float, float, float, float]:
    """
    Convert a normalized bbox to pixel coordinates for a page rendered
    at page_width x page_height.
    """
    x0, y0, x1, y1 = bbox
    return (x0 * page_width, y0 * page_height, x1 * page_width, y1 * page_height)
