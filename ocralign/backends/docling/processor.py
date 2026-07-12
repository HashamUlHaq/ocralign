"""
Docling backend processor: PDF/image -> Document via Docling's layout
pipeline (layout detection + reading order + table structure) with a
selectable OCR engine and compute device.

OCR engines:
    "tesseract" — Tesseract via its CLI (the system `tesseract` binary,
                  same engine as the vanilla backend). CPU only.
    "rapidocr"  — PP-OCR models on ONNX Runtime. CPU by default; runs
                  on GPU when device="cuda" and onnxruntime-gpu is
                  installed (same models, no code change).

device: "cpu" (default), "cuda", "mps", or "auto". Passed through to
Docling's accelerator options, which govern the layout and table
models; for rapidocr it selects the ONNX execution provider as well.
Tesseract has no GPU path — with ocr_engine="tesseract" the device
only affects the layout/table models.

Note on input resolution: Docling's OCR stage internally re-renders
regions at 3x scale, so feed ~100-150 DPI page images, not the ~300 DPI
you would give raw Tesseract — higher input DPI multiplies OCR cost
for no accuracy gain (measured: 300 DPI input doubles OCR time).
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, List, Optional

from ocralign.backends.docling.adapter import convert_result
from ocralign.core.schema import Document, Page

logger = logging.getLogger(__name__)

_OCR_ENGINES = ("tesseract", "rapidocr")
_DEVICES = ("cpu", "cuda", "mps", "auto")

# Engine-specific default language identifiers.
_DEFAULT_LANGS = {
    "tesseract": ["eng"],
    "rapidocr": ["english"],
}


def _build_converter(
    ocr_engine: str,
    device: str,
    tables: bool,
    lang: Optional[List[str]],
    force_ocr: bool,
):
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        RapidOcrOptions,
        TesseractCliOcrOptions,
    )
    from docling.document_converter import (
        DocumentConverter,
        ImageFormatOption,
        PdfFormatOption,
    )

    langs = lang if lang else _DEFAULT_LANGS[ocr_engine]

    po = PdfPipelineOptions()
    po.do_ocr = True
    po.do_table_structure = tables
    po.generate_parsed_pages = True  # the adapter needs the word cells
    po.accelerator_options = AcceleratorOptions(device=device)
    if tables:
        po.table_structure_options.do_cell_matching = True

    if ocr_engine == "tesseract":
        po.ocr_options = TesseractCliOcrOptions(
            lang=list(langs), force_full_page_ocr=force_ocr
        )
    else:  # rapidocr
        po.ocr_options = RapidOcrOptions(
            lang=list(langs), force_full_page_ocr=force_ocr
        )

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=po),
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=po),
        }
    )


def _validate(ocr_engine: str, device: str) -> None:
    if ocr_engine not in _OCR_ENGINES:
        raise ValueError(
            f"Invalid ocr_engine: {ocr_engine!r}. Valid engines: {', '.join(_OCR_ENGINES)}"
        )
    if device not in _DEVICES:
        raise ValueError(f"Invalid device: {device!r}. Valid devices: {', '.join(_DEVICES)}")


def process_pdf(
    pdf_path: str,
    ocr_engine: str = "tesseract",
    device: str = "cpu",
    tables: bool = True,
    lang: Optional[List[str]] = None,
    force_ocr: bool = False,
) -> Document:
    """
    Process a PDF (scanned or born-digital) into a Document with
    structural markdown page text and word-level coordinate mapping.

    Born-digital PDFs are handled automatically: Docling reads the text
    layer directly and only OCRs bitmap regions, so no `type` parameter
    is needed (unlike the vanilla backend). Pass force_ocr=True to OCR
    everything regardless of an existing text layer.

    Args:
        pdf_path: Path to the input PDF file.
        ocr_engine: "tesseract" or "rapidocr" (see module docstring).
        device: "cpu", "cuda", "mps", or "auto".
        tables: Run table-structure recognition (adds per-table cost on
                pages that contain tables; skipped elsewhere).
        lang: OCR language codes in the chosen engine's convention
              (default: English).
        force_ocr: Force full-page OCR even when a text layer exists.

    Returns:
        Document. page.text is markdown (headings, pipe tables,
        reading-order-correct paragraphs); ocralign.locate() resolves
        character spans back to page coordinates as with any backend.
    """
    _validate(ocr_engine, device)
    logger.info(f"Docling backend processing: {pdf_path} (ocr={ocr_engine}, device={device})")
    converter = _build_converter(ocr_engine, device, tables, lang, force_ocr)
    result = converter.convert(pdf_path)
    return convert_result(result)


def process_image(
    page_image: Any,
    ocr_engine: str = "tesseract",
    device: str = "cpu",
    tables: bool = True,
    lang: Optional[List[str]] = None,
) -> Page:
    """
    Process a single page image into a Page with structural markdown
    text and word-level coordinate mapping.

    Args:
        page_image: Path to an image file, or a PIL Image (written to a
                    temporary file for Docling, which converts by path).
        Other args: as in process_pdf. force_ocr is implied — an image
        has no text layer.
    """
    _validate(ocr_engine, device)
    converter = _build_converter(ocr_engine, device, tables, lang, force_ocr=False)

    if isinstance(page_image, (str, Path)):
        result = converter.convert(str(page_image))
    else:
        # PIL image (or anything with .save): hand to docling via a temp file.
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp_path = f.name
        try:
            page_image.save(tmp_path)
            result = converter.convert(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    doc = convert_result(result)
    if not doc.pages:
        return Page(page_number=1)
    return doc.pages[0]
