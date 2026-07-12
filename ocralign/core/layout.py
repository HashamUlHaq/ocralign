"""
Layout renderer: List[Word] -> formatted monospace text + char-offset mapping.

Engine-agnostic: operates only on the canonical schema (normalized 0-1
coordinates), so the same renderer serves Tesseract, PyMuPDF and any
future adapter.

Ported from the tess_align_normalized implementation:
- words clustered into lines by y-center sweep with median-derived tolerance
- median-based char width / line height (robust to tables and headers)
- horizontal placement by x0 / char_width with collision push-right
- vertical gaps converted to newlines under a configurable policy

While rendering, each word's final character span in the output text is
recorded on the Word itself (char_start/char_end/line_no). This is done
inside the renderer because collision handling can push a word right of
its bbox-derived column, so offsets are only knowable at paste time.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import List, Optional

from ocralign.core.schema import Word


@dataclass
class _Line:
    words: List[Word]
    top: float
    bottom: float

    @property
    def height(self) -> float:
        return max(1e-9, self.bottom - self.top)


def _robust_char_width(words: List[Word]) -> float:
    """Median pixels-per-character (in normalized x units)."""
    vals = [w.width / len(w.text) for w in words if w.text]
    return float(median(vals)) if vals else 0.005


def _group_words_into_lines(words: List[Word], y_tol: Optional[float] = None) -> List[_Line]:
    """Cluster words into lines by y-center proximity (single sweep)."""
    if not words:
        return []

    words_sorted = sorted(words, key=lambda w: (w.y_center, w.x0))

    if y_tol is None:
        h_med = float(median([w.height for w in words_sorted]))
        y_tol = max(1e-4, 0.55 * h_med)

    lines: List[_Line] = []
    cur: List[Word] = []
    cur_top = cur_bottom = 0.0
    cur_y: Optional[float] = None

    for w in words_sorted:
        if cur_y is None:
            cur = [w]
            cur_top, cur_bottom = w.y0, w.y1
            cur_y = w.y_center
            continue

        if abs(w.y_center - cur_y) <= y_tol:
            cur.append(w)
            cur_top = min(cur_top, w.y0)
            cur_bottom = max(cur_bottom, w.y1)
            cur_y = (cur_y * 0.85) + (w.y_center * 0.15)  # track drift smoothly
        else:
            cur.sort(key=lambda ww: ww.x0)
            lines.append(_Line(words=cur, top=cur_top, bottom=cur_bottom))
            cur = [w]
            cur_top, cur_bottom = w.y0, w.y1
            cur_y = w.y_center

    if cur:
        cur.sort(key=lambda ww: ww.x0)
        lines.append(_Line(words=cur, top=cur_top, bottom=cur_bottom))

    lines.sort(key=lambda ln: ln.top)
    return lines


def _gap_to_newlines(gap: float, line_h: float, policy: str) -> int:
    """
    Convert a vertical gap (normalized units) to a newline count.

    policy="capped": readable stream, paragraph gaps capped at 3 newlines.
    policy="proportional": gaps scale with distance (absolute-ish layout),
    generously capped to avoid runaway whitespace from noisy boxes.
    """
    if gap <= 0:
        return 1
    if policy == "capped":
        if gap <= 0.35 * line_h:
            return 1
        return min(1 + int(gap / line_h), 3)
    # proportional
    return min(max(1, 1 + int(round(gap / line_h))), 10)


def render_page(words: List[Word], gap_policy: str = "capped") -> str:
    """
    Render words into layout-aligned text and record each word's
    char_start/char_end/line_no in place.

    Returns the formatted page text. `words` list order is untouched;
    offsets refer to the returned string.
    """
    if gap_policy not in ("capped", "proportional"):
        raise ValueError(f"Invalid gap_policy: {gap_policy}. Use 'capped' or 'proportional'.")

    for w in words:
        w.line_no = None
        w.char_start = None
        w.char_end = None

    if not words:
        return ""

    char_w = _robust_char_width(words)
    lines = _group_words_into_lines(words)
    line_h = float(median([ln.height for ln in lines])) if lines else 0.01

    parts: List[str] = []
    offset = 0
    prev_bottom: Optional[float] = None

    for line_no, ln in enumerate(lines):
        if prev_bottom is None:
            prefix = ""
        else:
            n_new = _gap_to_newlines(ln.top - prev_bottom, line_h, gap_policy)
            prefix = "\n" * n_new

        # Paint words onto the line, recording spans as they are pasted.
        out = ""
        for w in ln.words:
            col = int(round(w.x0 / char_w))
            target = max(col, len(out) + (0 if not out else 1))
            if len(out) < target:
                out += " " * (target - len(out))
            w.line_no = line_no
            w.char_start = offset + len(prefix) + len(out)
            out += w.text
            w.char_end = offset + len(prefix) + len(out)

        parts.append(prefix + out)
        offset += len(prefix) + len(out)
        prev_bottom = ln.bottom

    return "".join(parts)
