"""
Vanilla backend processor: PDF/image -> Document via Tesseract or the
PDF text layer, rendered with the grid-layout renderer.
"""

import logging
from typing import Any, List, Optional

import fitz  # PyMuPDF
from PIL import Image
from tqdm import tqdm

from ocralign.backends.vanilla import pymupdf as pymupdf_adapter
from ocralign.backends.vanilla import tesseract as tesseract_adapter
from ocralign.backends.vanilla.layout import render_page
from ocralign.core.schema import Document, Page, Word

# No logging.basicConfig here: libraries must not configure the root
# logger (it force-enables INFO output from every dependency). Messages
# below surface only if the application configures logging itself.
logger = logging.getLogger(__name__)

_LAYOUT_TO_GAP_POLICY = {
    "normalized": "capped",
    "absolute": "proportional",
}


def _page_to_pil_image(page: "fitz.Page", dpi: int) -> Image.Image:
    """
    Render a single PyMuPDF page to a PIL.Image at the requested DPI.
    """
    # PDF user space is 72 DPI; scale matrix accordingly.
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)

    mode = "RGBA" if pix.alpha else "RGB"
    img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
    return img


def _build_page(
    page_number: int,
    words: List[Word],
    layout: str,
    width_px: Optional[int],
    height_px: Optional[int],
) -> Page:
    page = Page(
        page_number=page_number,
        words=words,
        width_px=width_px,
        height_px=height_px,
    )
    if layout == "none":
        page.text = " ".join(w.text for w in words)
        # No layout pass ran, so no char offsets are recorded.
    else:
        page.text = render_page(words, gap_policy=_LAYOUT_TO_GAP_POLICY[layout])
    return page


def process_pdf(
    pdf_path: str,
    type: str = "image",
    layout: str = "normalized",
    dpi: int = 300,
) -> Document:
    """
    Process a PDF into a Document: per-page formatted text plus word-level
    bounding boxes (normalized 0-1) with character-offset mapping.

    Args:
        pdf_path: Path to the input PDF file.
        type: "image" for scanned PDFs (OCR via Tesseract) or "digital"
              for PDFs with a retrievable text layer (no OCR).
        layout: "normalized" (readable stream, capped vertical gaps),
                "absolute" (vertical gaps proportional to page distance),
                or "none" (plain word stream, no offsets recorded).
        dpi: Render resolution for OCR ("image" type only).

    Returns:
        Document. Use doc.text() for the full formatted text,
        doc.pages[i].words for the mapping, and ocralign.locate()
        to resolve character spans back to page coordinates.
    """
    if layout not in ("normalized", "absolute", "none"):
        raise ValueError("Invalid layout. Valid layouts are: normalized, absolute, none")
    if type not in ("image", "digital"):
        raise ValueError('Invalid document type. Only "digital" or "image" is allowed')

    doc = None
    try:
        logger.info(f"Starting PDF processing for: {pdf_path}")
        doc = fitz.open(pdf_path)
        logger.info(f"Opened PDF with {doc.page_count} page(s)")

        pages: List[Page] = []
        for page_index in tqdm(range(doc.page_count), desc="Processing Pages"):
            fitz_page = doc.load_page(page_index)
            if type == "image":
                image = _page_to_pil_image(fitz_page, dpi=dpi)
                words = tesseract_adapter.extract_words(image)
                width_px, height_px = image.width, image.height
            else:
                words = pymupdf_adapter.extract_words(fitz_page)
                width_px = int(fitz_page.rect.width)
                height_px = int(fitz_page.rect.height)
            pages.append(_build_page(page_index + 1, words, layout, width_px, height_px))

        return Document(pages=pages)
    except Exception as e:
        logger.error(f"Failed to process PDF: {e}", exc_info=True)
        raise
    finally:
        if doc is not None:
            doc.close()


def process_image(page_image: Any, layout: str = "normalized") -> Page:
    """
    OCR a single image (PIL Image, numpy array, or path) into a Page
    with formatted text and word-level mapping.
    """
    if layout not in ("normalized", "absolute", "none"):
        raise ValueError("Invalid layout. Valid layouts are: normalized, absolute, none")

    if isinstance(page_image, str):
        page_image = Image.open(page_image)

    words = tesseract_adapter.extract_words(page_image)
    width_px = getattr(page_image, "width", None)
    height_px = getattr(page_image, "height", None)
    return _build_page(1, words, layout, width_px, height_px)
