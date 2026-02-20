"""
Tesseract hOCR -> layout-aligned text (PSM 12) WITHOUT newline explosion.

Key fixes vs your original:
1) Stop using ocr_line as the primitive (psm 12 fragments lines). Use ocrx_word boxes.
2) Build lines with a stable sweep algorithm (sort by y, cluster by y-center tolerance).
3) Use robust stats (median) for char width + line height (mean is easily skewed by tables).
4) Convert vertical gaps to newlines with thresholds + caps (no round()+1 runaway).
5) Avoid O(n^2) row grouping and "seed-only overlap" problems.

Output:
- Attempts to preserve columns/tables using whitespace alignment
- Produces a readable linear text stream (not pixel-perfect)
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytesseract
from bs4 import BeautifulSoup


# -----------------------------
# Types / helpers
# -----------------------------
BBox = Tuple[int, int, int, int]  # (x0, y0, x1, y1)


@dataclass(frozen=True)
class WordBox:
    bbox: BBox
    text: str

    @property
    def x0(self) -> int: return self.bbox[0]

    @property
    def y0(self) -> int: return self.bbox[1]

    @property
    def x1(self) -> int: return self.bbox[2]

    @property
    def y1(self) -> int: return self.bbox[3]

    @property
    def w(self) -> int: return max(1, self.x1 - self.x0)

    @property
    def h(self) -> int: return max(1, self.y1 - self.y0)

    @property
    def y_center(self) -> float: return (self.y0 + self.y1) / 2.0


@dataclass
class Line:
    words: List[WordBox]
    top: int
    bottom: int

    @property
    def height(self) -> int:
        return max(1, self.bottom - self.top)

    @property
    def y_center(self) -> float:
        return (self.top + self.bottom) / 2.0


def parse_bbox(title: str) -> BBox:
    """
    Extracts bbox from a hOCR title string.
    Expected title fragment: "bbox x0 y0 x1 y1; ..."
    """
    title = (title or "").strip()
    if not title.startswith("bbox"):
        # Sometimes bbox appears later in the title; be defensive
        if "bbox" not in title:
            raise ValueError(f"Error parsing bbox: {title!r}")
        title = title[title.index("bbox") :]

    bbox_part = title.split(";")[0].replace("bbox", "").strip()
    parts = [p for p in bbox_part.split() if p]
    if len(parts) != 4:
        raise ValueError(f"Error parsing bbox: {title!r}")
    x0, y0, x1, y1 = (int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]))
    return (x0, y0, x1, y1)


def _norm_text(s: str) -> str:
    # Keep it simple: collapse internal newlines, strip ends
    return " ".join((s or "").replace("\n", " ").split()).strip()


def _robust_char_width(words: Sequence[WordBox]) -> float:
    """
    Estimate a global average character width (pixels per char) from word boxes.
    Median is robust against long tokens and tables.
    """
    vals: List[float] = []
    for w in words:
        n = len(w.text)
        if n > 0:
            vals.append(w.w / n)
    return float(median(vals)) if vals else 7.0


def _robust_line_height(lines: Sequence[Line]) -> float:
    """
    Estimate a typical line height (pixels) from line bands.
    Median is robust against tall table rows / headers.
    """
    hs = [ln.height for ln in lines if ln.height > 0]
    return float(median(hs)) if hs else 15.0


# -----------------------------
# hOCR extraction (word-level)
# -----------------------------
def extract_words_from_hocr(hocr_bytes: bytes) -> List[WordBox]:
    """
    Parse Tesseract hOCR and return word-level boxes (ocrx_word).
    This is more stable than relying on ocr_line output in psm 12.
    """
    soup = BeautifulSoup(hocr_bytes, "html.parser")

    # Word nodes are typically: <span class="ocrx_word" ...>text</span>
    word_spans = soup.find_all("span", class_="ocrx_word")

    words: List[WordBox] = []
    for sp in word_spans:
        txt = _norm_text(sp.get_text())
        if not txt:
            continue
        title = sp.get("title", "")
        try:
            bbox = parse_bbox(title)
        except Exception:
            continue
        words.append(WordBox(bbox=bbox, text=txt))

    # Sometimes some PDFs yield few/no ocrx_word. Fallback: use ocr_line as a last resort.
    if not words:
        line_spans = soup.find_all("span", class_="ocr_line")
        for sp in line_spans:
            txt = _norm_text(sp.get_text())
            if not txt:
                continue
            title = sp.get("title", "")
            try:
                bbox = parse_bbox(title)
            except Exception:
                continue
            # Split line text into tokens and approximate each as one word box
            # (Fallback only; not ideal but prevents empty output.)
            words.append(WordBox(bbox=bbox, text=txt))

    return words


# -----------------------------
# Line construction (fast sweep)
# -----------------------------
def group_words_into_lines(words: List[WordBox], y_tol: Optional[float] = None) -> List[Line]:
    """
    Cluster words into lines using y-center proximity (sweep).
    This avoids O(n^2) overlap grouping and is robust to psm 12 fragmentation.

    y_tol: tolerance in pixels. If None, computed from median word height.
    """
    if not words:
        return []

    # Sort by y-center, then x0
    words_sorted = sorted(words, key=lambda w: (w.y_center, w.x0))

    # Use a tolerance derived from typical word height
    if y_tol is None:
        h_med = median([w.h for w in words_sorted])
        # 0.55*h_med tends to group same line while separating adjacent lines
        y_tol = max(2.0, 0.55 * float(h_med))

    lines: List[Line] = []
    cur_words: List[WordBox] = []
    cur_top: int = 0
    cur_bottom: int = 0
    cur_y: Optional[float] = None

    for w in words_sorted:
        if cur_y is None:
            cur_words = [w]
            cur_top, cur_bottom = w.y0, w.y1
            cur_y = w.y_center
            continue

        if abs(w.y_center - cur_y) <= y_tol:
            cur_words.append(w)
            cur_top = min(cur_top, w.y0)
            cur_bottom = max(cur_bottom, w.y1)
            # Track drift smoothly
            cur_y = (cur_y * 0.85) + (w.y_center * 0.15)
        else:
            # finalize current line
            cur_words.sort(key=lambda ww: ww.x0)
            lines.append(Line(words=cur_words, top=cur_top, bottom=cur_bottom))
            # start next
            cur_words = [w]
            cur_top, cur_bottom = w.y0, w.y1
            cur_y = w.y_center

    # finalize last
    if cur_words:
        cur_words.sort(key=lambda ww: ww.x0)
        lines.append(Line(words=cur_words, top=cur_top, bottom=cur_bottom))

    # Sort top-to-bottom (stable)
    lines.sort(key=lambda ln: ln.top)
    return lines


# -----------------------------
# Horizontal alignment renderer
# -----------------------------
def render_line_aligned(line: Line, char_w: float) -> str:
    """
    Paint words onto a monospaced line using their x0 / char_w to compute columns.
    """
    if not line.words:
        return ""

    out = ""
    for w in line.words:
        col = int(round(w.x0 / char_w))
        # Ensure at least one space between tokens in case of collisions
        target = max(col, len(out) + (0 if not out else 1))
        if len(out) < target:
            out += " " * (target - len(out))
        out += w.text
    return out.rstrip()


# -----------------------------
# Vertical spacing logic (no runaway newlines)
# -----------------------------
def gap_to_newlines(gap_px: float, line_h: float) -> int:
    """
    Convert pixel gap to newline count.
    - Always at least 1 newline between lines.
    - Use thresholds for paragraph breaks.
    - Cap to avoid huge whitespace from noisy bbox gaps.
    """
    if gap_px <= 0:
        return 1

    # Normal inter-line gap: 1 newline
    if gap_px <= 0.35 * line_h:
        return 1

    # Larger gaps: scale, but cap
    n = 1 + int(gap_px / line_h)
    return min(max(1, n), 3)


# -----------------------------
# Main entry point
# -----------------------------
def process_page(
    page_image: Any,
    *,
    tesseract_config: str = "--psm 12 -c preserve_interword_spaces=1",
) -> str:
    """
    Extract and align text from a single page image using Tesseract hOCR (PSM 12).
    page_image can be a PIL Image, numpy array, or image bytes supported by pytesseract.
    """
    # 1) OCR -> hOCR
    hocr = pytesseract.image_to_pdf_or_hocr(
        page_image,
        extension="hocr",
        config=tesseract_config,
    )

    # 2) Parse words with bboxes
    words = extract_words_from_hocr(hocr)
    if not words:
        return ""

    # 3) Robust global char width
    char_w = _robust_char_width(words)

    # 4) Group words into lines
    lines = group_words_into_lines(words)

    # 5) Robust line height
    line_h = _robust_line_height(lines)

    # 6) Render with controlled vertical gaps
    out_parts: List[str] = []
    prev_bottom: Optional[int] = None

    for ln in lines:
        txt = render_line_aligned(ln, char_w)
        if not txt:
            continue

        if prev_bottom is None:
            out_parts.append(txt)
            prev_bottom = ln.bottom
            continue

        gap_px = float(ln.top - prev_bottom)
        n_new = gap_to_newlines(gap_px, line_h)
        out_parts.append("\n" * n_new + txt)
        prev_bottom = ln.bottom

    return "".join(out_parts)

    