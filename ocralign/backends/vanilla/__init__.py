"""
Vanilla backend: Tesseract OCR (scanned pages) / PyMuPDF text layer
(born-digital PDFs) + the monospace grid-layout renderer.

Produces visually formatted text that mirrors the page's spatial layout.
Best for simple single-column documents; has no concept of columns or
table structure (use the docling backend for those).
"""

from ocralign.backends.vanilla.processor import process_image, process_pdf

__all__ = ["process_pdf", "process_image"]
