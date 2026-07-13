"""
Docling adapter: ConversionResult -> Document (canonical schema).

Walks the DoclingDocument's items in reading order (Docling's
reading-order model has already sequenced them, which is what fixes
multi-column interleaving) and renders markdown-style page text
ourselves, claiming word cells from the parsed page as we go and
recording each word's char_start/char_end into the rendered string.

Rendering the text ourselves — instead of reverse-mapping into
Docling's export_to_markdown() output — is what guarantees offset
integrity: a word's span is recorded at the moment it is painted, so
page.text[w.char_start:w.char_end] == w.text always holds, and
locate() works identically to the vanilla backend.

Coordinate notes (verified against docling 2.x):
- parsed-page cells (word_cells/textline_cells) are TOPLEFT-origin in
  page coordinates; item provenance bboxes are BOTTOMLEFT. Every bbox
  is normalized individually via its own coord_origin — never assume
  one origin per page.
- word_cells are true word granularity for born-digital PDFs and
  empty for OCR'd pages, where textline_cells hold the OCR cells
  (word granularity for Tesseract, text-line granularity for
  RapidOCR). Coarser cells just mean coarser overlay boxes; locate()
  overlap semantics are unaffected.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from docling_core.types.doc import DocItemLabel, TableItem
from docling_core.types.doc.page import TextCell

from ocralign.core.schema import BBox, Document, Page, Word

# Vertical overlap ratio for clustering claimed words into visual lines
# (same approach as the vanilla renderer: ratio of the shorter span, so
# it is font-size invariant).
_MIN_LINE_OVERLAP = 0.4

# Fraction of a word box's area that must fall inside a region bbox for
# the word to be claimed by that region.
_MIN_REGION_CONTAINMENT = 0.5

_HEADING_PREFIXES = {
    DocItemLabel.TITLE: "# ",
    DocItemLabel.SECTION_HEADER: "## ",
}


class _PoolWord:
    """A page word cell awaiting assignment to a document item."""

    __slots__ = ("text", "bbox", "confidence", "claimed")

    def __init__(self, text: str, bbox: BBox, confidence: Optional[float]):
        self.text = text
        self.bbox = bbox
        self.confidence = confidence
        self.claimed = False


def _norm_cell_bbox(cell: TextCell, page_w: float, page_h: float) -> BBox:
    bb = cell.rect.to_bounding_box().to_top_left_origin(page_h)
    return (bb.l / page_w, bb.t / page_h, bb.r / page_w, bb.b / page_h)


def _norm_prov_bbox(bbox, page_w: float, page_h: float) -> BBox:
    bb = bbox.to_top_left_origin(page_h)
    return (bb.l / page_w, bb.t / page_h, bb.r / page_w, bb.b / page_h)


def _containment(word_bbox: BBox, region: BBox) -> float:
    """Fraction of the word box's area inside the region box."""
    wx0, wy0, wx1, wy1 = word_bbox
    rx0, ry0, rx1, ry1 = region
    ix = max(0.0, min(wx1, rx1) - max(wx0, rx0))
    iy = max(0.0, min(wy1, ry1) - max(wy0, ry0))
    area = max(1e-12, (wx1 - wx0) * (wy1 - wy0))
    return (ix * iy) / area


def _build_word_pool(parsed_page, page_w: float, page_h: float) -> List[_PoolWord]:
    """
    Extract the finest-granularity cells available from a parsed page.
    """
    cells = []
    if getattr(parsed_page, "has_words", False) and parsed_page.word_cells:
        cells = parsed_page.word_cells
    elif parsed_page.textline_cells:
        cells = parsed_page.textline_cells

    pool: List[_PoolWord] = []
    for c in cells:
        text = " ".join(c.text.split())
        if not text:
            continue
        conf = getattr(c, "confidence", None)
        if conf is not None:
            # docling reports 0-1; canonical schema uses 0-100 (Tesseract-style)
            conf = conf * 100.0 if conf <= 1.0 else conf
        pool.append(_PoolWord(text, _norm_cell_bbox(c, page_w, page_h), conf))
    return pool


def _claim_words(pool: List[_PoolWord], region: BBox) -> List[_PoolWord]:
    """Take (and mark claimed) all unclaimed pool words inside region."""
    hits = [
        w for w in pool
        if not w.claimed and _containment(w.bbox, region) >= _MIN_REGION_CONTAINMENT
    ]
    for w in hits:
        w.claimed = True
    return hits


def _cluster_into_lines(words: List[_PoolWord]) -> List[List[_PoolWord]]:
    """Group words into visual lines by vertical bbox overlap, then x."""
    if not words:
        return []
    ws = sorted(words, key=lambda w: (w.bbox[1], w.bbox[0]))
    lines: List[List[_PoolWord]] = []
    cur: List[_PoolWord] = []
    cur_top = cur_bottom = 0.0
    for w in ws:
        top, bottom = w.bbox[1], w.bbox[3]
        if not cur:
            cur, cur_top, cur_bottom = [w], top, bottom
            continue
        overlap = max(0.0, min(cur_bottom, bottom) - max(cur_top, top))
        shorter = min(cur_bottom - cur_top, bottom - top)
        if shorter > 0 and overlap / shorter >= _MIN_LINE_OVERLAP:
            cur.append(w)
            cur_top = min(cur_top, top)
            cur_bottom = max(cur_bottom, bottom)
        else:
            cur.sort(key=lambda x: x.bbox[0])
            lines.append(cur)
            cur, cur_top, cur_bottom = [w], top, bottom
    if cur:
        cur.sort(key=lambda x: x.bbox[0])
        lines.append(cur)
    return lines


class _PageBuilder:
    """
    Accumulates markdown text for one page, recording Word entries with
    exact char offsets as text is emitted.
    """

    def __init__(self, page_number: int, pool: List[_PoolWord],
                 width_px: Optional[int], height_px: Optional[int]):
        self.page_number = page_number
        self.pool = pool
        self.width_px = width_px
        self.height_px = height_px
        self._parts: List[str] = []
        self._len = 0
        self._line_no = 0
        self.words: List[Word] = []

    def _emit(self, text: str) -> None:
        self._parts.append(text)
        self._len += len(text)

    def _emit_word(self, w: _PoolWord, line_no: int) -> None:
        start = self._len
        self._emit(w.text)
        self.words.append(
            Word(
                text=w.text,
                bbox=w.bbox,
                confidence=w.confidence,
                line_no=line_no,
                char_start=start,
                char_end=self._len,
            )
        )

    def _next_line_no(self) -> int:
        n = self._line_no
        self._line_no += 1
        return n

    def add_block(self, region: Optional[BBox], fallback_text: str,
                  prefix: str = "", code: bool = False) -> None:
        """
        Emit one markdown block (paragraph / heading / list item / code).

        Words inside `region` are claimed from the pool and painted in
        visual order; paragraph lines are soft-wrapped (joined with a
        space) into one flowing block, which reads better for LLMs than
        hard line breaks. If no words match the region (rare), the
        item's own text is emitted unmapped so nothing is lost.
        """
        claimed = _claim_words(self.pool, region) if region is not None else []
        if not claimed:
            text = " ".join(fallback_text.split())
            if not text:
                return
            if code:
                self._emit("```\n" + text + "\n```\n\n")
            else:
                self._emit(prefix + text + "\n\n")
            return

        lines = _cluster_into_lines(claimed)
        if code:
            self._emit("```\n")
            for line in lines:
                line_no = self._next_line_no()
                for i, w in enumerate(line):
                    if i:
                        self._emit(" ")
                    self._emit_word(w, line_no)
                self._emit("\n")
            self._emit("```\n\n")
        else:
            self._emit(prefix)
            first = True
            for line in lines:
                line_no = self._next_line_no()
                for w in line:
                    if not first:
                        self._emit(" ")
                    self._emit_word(w, line_no)
                    first = False
            self._emit("\n\n")

    def add_table(self, item: TableItem, page_w: float, page_h: float) -> None:
        """
        Emit a markdown pipe table. Each grid row is one output line;
        every word claimed within a cell records offsets into that line.
        Cells whose region matches no pool words fall back to the
        table-structure model's own cell text (unmapped).
        """
        data = item.data
        if not data.table_cells:
            return

        nrows = data.num_rows
        ncols = data.num_cols
        # grid[r][c] -> cell placed at its starting position only
        grid: List[List[Optional[object]]] = [[None] * ncols for _ in range(nrows)]
        for cell in data.table_cells:
            r, c = cell.start_row_offset_idx, cell.start_col_offset_idx
            if 0 <= r < nrows and 0 <= c < ncols and grid[r][c] is None:
                grid[r][c] = cell

        for r in range(nrows):
            line_no = self._next_line_no()
            self._emit("|")
            for c in range(ncols):
                self._emit(" ")
                cell = grid[r][c]
                if cell is not None:
                    region = None
                    if cell.bbox is not None:
                        region = _norm_prov_bbox(cell.bbox, page_w, page_h)
                    claimed = _claim_words(self.pool, region) if region else []
                    if claimed:
                        flat = [w for line in _cluster_into_lines(claimed) for w in line]
                        for i, w in enumerate(flat):
                            if i:
                                self._emit(" ")
                            self._emit_word(w, line_no)
                    else:
                        self._emit(" ".join(cell.text.split()))
                self._emit(" |")
            self._emit("\n")
            if r == 0:
                self._emit("|" + "---|" * ncols + "\n")
        self._emit("\n")

    def build(self) -> Page:
        text = "".join(self._parts).rstrip("\n")
        return Page(
            page_number=self.page_number,
            words=self.words,
            text=text,
            width_px=self.width_px,
            height_px=self.height_px,
        )


def convert_result(result) -> Document:
    """
    Convert a docling ConversionResult into a canonical Document with
    markdown page text and word-level offset mapping.

    Requires the conversion to have run with generate_parsed_pages=True
    (the processor sets this).
    """
    # Per-page word pools from the parsed pages.
    pools: Dict[int, List[_PoolWord]] = {}
    sizes: Dict[int, Tuple[float, float]] = {}

    for page in result.pages:
        if page.size is None:
            continue
        # Both result.pages[].page_no and item prov page_no are 1-based
        # (verified empirically against docling 2.x).
        page_no = page.page_no
        w, h = float(page.size.width), float(page.size.height)
        sizes[page_no] = (w, h)
        parsed = page.parsed_page
        pools[page_no] = _build_word_pool(parsed, w, h) if parsed is not None else []

    builders: Dict[int, _PageBuilder] = {}

    def builder_for(page_no: int) -> _PageBuilder:
        if page_no not in builders:
            w, h = sizes.get(page_no, (1.0, 1.0))
            builders[page_no] = _PageBuilder(
                page_number=page_no,
                pool=pools.get(page_no, []),
                width_px=int(w),
                height_px=int(h),
            )
        return builders[page_no]

    doc = result.document
    for item, _level in doc.iterate_items():
        prov = getattr(item, "prov", None)
        if not prov:
            continue
        p = prov[0]
        page_no = p.page_no
        w, h = sizes.get(page_no, (1.0, 1.0))
        b = builder_for(page_no)

        if isinstance(item, TableItem):
            b.add_table(item, w, h)
            continue

        text = getattr(item, "text", "") or ""
        label = getattr(item, "label", None)
        if not text and label != DocItemLabel.CODE:
            continue  # pictures etc.

        region = _norm_prov_bbox(p.bbox, w, h)
        prefix = _HEADING_PREFIXES.get(label, "")
        if label == DocItemLabel.LIST_ITEM:
            prefix = "- "
        b.add_block(
            region=region,
            fallback_text=text,
            prefix=prefix,
            code=(label == DocItemLabel.CODE),
        )

    pages = [builders[k].build() for k in sorted(builders)]
    # Include word-less empty pages so page numbering stays dense.
    for page_no in sorted(sizes):
        if page_no not in builders:
            w, h = sizes[page_no]
            pages.append(Page(page_number=page_no, words=[], text="",
                              width_px=int(w), height_px=int(h)))
    pages.sort(key=lambda p: p.page_number)
    return Document(pages=pages)
