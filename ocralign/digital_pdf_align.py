from __future__ import annotations
import fitz  # PyMuPDF
from statistics import median
from typing import List, Tuple


def _estimate_line_height(words: List[Tuple[float, float, float, float, str]]) -> float:
    """
    Estimate a good line height (in PDF points) from word boxes.
    """
    heights = [(y1 - y0) for (x0, y0, x1, y1, w) in words if (y1 - y0) > 0]
    if not heights:
        return 12.0
    h = median(heights)
    # Add a bit of spacing to reduce accidental line collisions
    return max(10.0, h * 1.15)


def _group_words_into_lines(
    words: List[Tuple[float, float, float, float, str]],
    line_height: float,
) -> List[List[Tuple[float, float, float, float, str]]]:
    """
    Group words into lines by y-center proximity (tolerant to minor drift).
    """
    if not words:
        return []

    # Sort by y, then x
    words_sorted = sorted(words, key=lambda t: ((t[1] + t[3]) / 2.0, t[0]))
    lines: List[List[Tuple[float, float, float, float, str]]] = []
    current: List[Tuple[float, float, float, float, str]] = []
    current_y = None
    tol = line_height * 0.45

    for x0, y0, x1, y1, w in words_sorted:
        y_center = (y0 + y1) / 2.0
        if current_y is None:
            current_y = y_center
            current = [(x0, y0, x1, y1, w)]
            continue

        if abs(y_center - current_y) <= tol:
            current.append((x0, y0, x1, y1, w))
            # Slowly track drift
            current_y = (current_y * 0.85) + (y_center * 0.15)
        else:
            # finalize previous line
            lines.append(sorted(current, key=lambda t: t[0]))
            current = [(x0, y0, x1, y1, w)]
            current_y = y_center

    if current:
        lines.append(sorted(current, key=lambda t: t[0]))

    return lines


def page_to_layout_text(
    page: fitz.Page,
    cols: int = 140,
    keep_page_marker: bool = True,
) -> str:
    """
    Convert a PDF page (born-digital) to a monospaced layout-ish text output.

    - cols controls horizontal resolution (more cols => better column fidelity, more whitespace).
    - Uses PDF word bboxes to place text into a grid.
    """
    # Extract words: each tuple is (x0, y0, x1, y1, word, block_no, line_no, word_no)
    raw = page.get_text("words")
    if not raw:
        return f"[PAGE {page.number + 1}]\n" if keep_page_marker else ""

    # Keep (x0,y0,x1,y1,text)
    words = [(x0, y0, x1, y1, w) for (x0, y0, x1, y1, w, *_rest) in raw if w.strip()]

    rect = page.rect
    page_w = float(rect.width)
    page_h = float(rect.height)

    line_height = _estimate_line_height(words)
    lines = _group_words_into_lines(words, line_height=line_height)

    # Vertical resolution: approximate number of text rows
    # Use page height / line height, add a bit of slack
    rows = max(1, int(page_h / line_height) + 3)

    # Create empty canvas
    canvas = [list(" " * cols) for _ in range(rows)]

    def x_to_col(x: float) -> int:
        # Map PDF x coordinate into [0, cols-1]
        return max(0, min(cols - 1, int((x / page_w) * (cols - 1))))

    def y_to_row(y: float) -> int:
        # Map PDF y coordinate into [0, rows-1]
        return max(0, min(rows - 1, int((y / page_h) * (rows - 1))))

    for line in lines:
        # Use the line's y-center for row placement
        y_center = median([(y0 + y1) / 2.0 for (x0, y0, x1, y1, w) in line])
        r = y_to_row(y_center)

        # Place each word at its left edge; insert at least one space between words
        for (x0, y0, x1, y1, w) in line:
            c = x_to_col(x0)

            # Ensure we don't overwrite too aggressively if collisions happen
            # Find next available spot if current spot is already non-space
            while c < cols and canvas[r][c] != " ":
                c += 1
            if c >= cols:
                continue

            # Write word (truncate if needed)
            for ch in w:
                if c >= cols:
                    break
                canvas[r][c] = ch
                c += 1

            # Add a space after the word (if room)
            if c < cols:
                canvas[r][c] = " "
    
    # Convert canvas to lines, rstrip to keep layout while avoiding infinite trailing spaces
    text_lines = ["".join(row).rstrip() for row in canvas]

    # Drop fully empty rows at top/bottom
    while text_lines and text_lines[0] == "":
        text_lines.pop(0)
    while text_lines and text_lines[-1] == "":
        text_lines.pop()

    body = "\n".join(text_lines)
    
    return body.rstrip()