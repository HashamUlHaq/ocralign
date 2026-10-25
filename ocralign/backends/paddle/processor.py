"""
Paddle backend processor: PDF/image -> Document via the PaddleOCR-VL service.

The service returns per-page layout blocks; render.py turns them into markdown page text and this
module assembles the canonical schema:

- Every page of the source is emitted, in order. A page the service returned no record for is
  still there (empty), so page N is always the N-th page of the document.
- Page.text is the rendered text. Page.blocks holds every layout region in reading order, with
  char offsets into page.text (page.text[b.char_start:b.char_end] == b.text) and a normalized
  0-1 top-left bbox.
- Page.words is empty: PaddleOCR-VL recognises whole regions, not words, so there is no word
  geometry to report. locate() falls back to the block boxes.
"""

from __future__ import annotations

import io
import os
from typing import Any, List, Optional

from ocralign.backends.paddle.client import DEFAULT_URL, fetch_payload, post_document
from ocralign.backends.paddle.render import BLOCK_SEPARATOR, render_segments
from ocralign.core.schema import BBox, Block, Document, Page


def _clamp01(v: float) -> float:
    return min(1.0, max(0.0, v))


def _norm_bbox(bbox: Any, width: Any, height: Any) -> Optional[BBox]:
    """Service pixel box (x0, y0, x1, y1) on a width x height page -> normalized 0-1, or None."""
    try:
        x0, y0, x1, y1 = (float(v) for v in bbox)
        w, h = float(width), float(height)
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    # The layout model's boxes stay inside the page, but clamp so the 0-1 convention holds
    # whatever it returns.
    return (_clamp01(x0 / w), _clamp01(y0 / h), _clamp01(x1 / w), _clamp01(y1 / h))


def _int_or_none(v: Any) -> Optional[int]:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _build_page(page_number: int, record: Optional[dict]) -> Page:
    if record is None:  # the service returned nothing for this page
        return Page(page_number=page_number)
    width, height = record.get("width"), record.get("height")
    parts: List[str] = []
    blocks: List[Block] = []
    pos = 0
    for raw, label, segment in render_segments(record.get("blocks")):
        bbox = _norm_bbox(raw.get("bbox"), width, height)
        if segment is None:
            blocks.append(Block(label=label, bbox=bbox))
            continue
        if parts:
            pos += len(BLOCK_SEPARATOR)
        blocks.append(Block(label=label, text=segment, bbox=bbox,
                            char_start=pos, char_end=pos + len(segment)))
        parts.append(segment)
        pos += len(segment)
    return Page(
        page_number=page_number,
        text=BLOCK_SEPARATOR.join(parts),
        width_px=_int_or_none(width),
        height_px=_int_or_none(height),
        blocks=blocks,
    )


def payload_to_document(payload: dict) -> Document:
    """
    Service payload (fetch_payload) -> Document, one Page per page of the source, in order.

    Pure, no network: a stored payload can be re-rendered after a renderer change without
    re-OCR'ing. Raises ValueError if the payload has no pages at all.
    """
    records = payload.get("pages") or []
    by_index = {int(r["page_index"]): r for r in records}
    n = max(int(payload.get("page_count") or 0), (max(by_index) + 1) if by_index else 0)
    if n == 0:
        raise ValueError("PaddleOCR returned no pages for this document")
    return Document(pages=[_build_page(i + 1, by_index.get(i)) for i in range(n)])


def process_pdf(pdf_path: str, url: str = DEFAULT_URL, timeout: float = 3600.0) -> Document:
    """
    Process a document into a Document with markdown page text and block-level geometry.

    Args:
        pdf_path: Path to a PDF. Scanned and born-digital PDFs are handled alike: pages are read
                  from pixels, so a broken embedded text layer can't leak through. Image files
                  the service accepts (png, jpg, tif, bmp) work too; a multi-page TIFF yields one
                  Page per frame.
        url: Base URL of the PaddleOCR service (started by service/stack.sh).
        timeout: Seconds allowed for the whole document.

    Returns:
        Document. page.text is markdown (tagged page header/footer, headings, pipe tables);
        page.blocks holds the layout regions, and ocralign.locate() resolves character spans to
        the boxes of the blocks they fall in.
    """
    return payload_to_document(fetch_payload(pdf_path, url=url, timeout=timeout))


def process_image(page_image: Any, url: str = DEFAULT_URL, timeout: float = 3600.0) -> Page:
    """
    Process a single page image into a Page with markdown text and block-level geometry.

    Args:
        page_image: Path to an image file, or a PIL Image (sent to the service as PNG).
        Other args: as in process_pdf.
    """
    if isinstance(page_image, (str, os.PathLike)):
        payload = fetch_payload(page_image, url=url, timeout=timeout)
    elif hasattr(page_image, "save"):
        buf = io.BytesIO()
        page_image.save(buf, format="PNG")
        payload = post_document(buf.getvalue(), "png", url=url, timeout=timeout)
    else:
        raise TypeError(f"page_image must be a path or a PIL Image, got {type(page_image).__name__}")
    return payload_to_document(payload).pages[0]
