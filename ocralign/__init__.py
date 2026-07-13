from ocralign.core.schema import BBox, Document, Page, Word
from ocralign.core.locate import locate, locate_in_document, locate_substring, to_pixels
from ocralign.ocr import process_pdf, process_image

__version__ = "0.2.0"

__all__ = [
    "BBox",
    "Document",
    "Page",
    "Word",
    "locate",
    "locate_in_document",
    "locate_substring",
    "to_pixels",
    "process_pdf",
    "process_image",
]
