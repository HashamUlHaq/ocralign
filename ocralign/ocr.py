"""
Entry points: process_pdf / process_image with a selectable backend.

Backends
--------
"vanilla" (default)
    Tesseract OCR (scanned pages) or the PDF text layer (born-digital),
    rendered as visually formatted monospace text that mirrors the page
    layout. Fast, light dependencies. No understanding of columns or
    tables — multi-column pages interleave and table rows flatten.

"docling"
    Docling layout analysis (layout-detection + reading-order +
    table-structure models) on top of a selectable OCR engine
    ("tesseract" or "rapidocr", CPU or GPU). Produces structural
    markdown text (headings, pipe tables, reading-order-correct
    paragraphs) meant for LLM/NER consumption rather than visual
    fidelity. Requires the optional docling dependencies:
    pip install "ocralign[docling]"

Both backends emit the same Document/Page/Word schema with word-level
normalized bounding boxes and char offsets into the page text, so
locate() / overlay resolution works identically regardless of backend.
"""

from typing import Any

from ocralign.backends import vanilla
from ocralign.core.schema import Document, Page

_BACKENDS = ("vanilla", "docling")


def _docling_backend():
    try:
        from ocralign.backends import docling as docling_backend
    except ImportError as e:
        raise ImportError(
            "The docling backend requires optional dependencies. "
            "Install them with: pip install 'ocralign[docling]'"
        ) from e
    return docling_backend


def process_pdf(pdf_path: str, backend: str = "vanilla", **kwargs: Any) -> Document:
    """
    Process a PDF into a Document with per-page text and word-level
    coordinate mapping.

    Args:
        pdf_path: Path to the input PDF file.
        backend: "vanilla" or "docling" (see module docstring).
        **kwargs: Backend-specific options.
            vanilla: type="image"|"digital", layout="normalized"|"absolute"|"none", dpi=300
            docling: ocr_engine="tesseract"|"rapidocr", device="cpu"|"cuda"|"auto",
                     tables=True, lang=None, force_ocr=False

    Returns:
        Document. Use doc.pages[i].text for the page text, and
        ocralign.locate() to resolve character spans back to page
        coordinates for overlays.
    """
    if backend not in _BACKENDS:
        raise ValueError(f"Invalid backend: {backend!r}. Valid backends: {', '.join(_BACKENDS)}")
    if backend == "docling":
        return _docling_backend().process_pdf(pdf_path, **kwargs)
    return vanilla.process_pdf(pdf_path, **kwargs)


def process_image(page_image: Any, backend: str = "vanilla", **kwargs: Any) -> Page:
    """
    Process a single page image into a Page with text and word-level
    coordinate mapping.

    Args:
        page_image: PIL Image or path to an image file.
        backend: "vanilla" or "docling" (see module docstring).
        **kwargs: Backend-specific options (see process_pdf).
    """
    if backend not in _BACKENDS:
        raise ValueError(f"Invalid backend: {backend!r}. Valid backends: {', '.join(_BACKENDS)}")
    if backend == "docling":
        return _docling_backend().process_image(page_image, **kwargs)
    return vanilla.process_image(page_image, **kwargs)
