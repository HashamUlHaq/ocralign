"""
Docling backend: layout-aware document processing (layout detection,
reading order, table structure) on top of a selectable OCR engine
(tesseract | rapidocr) and device (cpu | cuda | mps | auto).

Produces structural markdown text meant for LLM/NER consumption —
correct multi-column reading order and pipe-rendered tables — while
emitting the same Word/Page/Document schema as the vanilla backend,
so locate() and overlay resolution work unchanged.

Optional dependency: pip install "ocralign[docling]"
"""

from ocralign.backends.docling.processor import process_image, process_pdf

__all__ = ["process_pdf", "process_image"]
