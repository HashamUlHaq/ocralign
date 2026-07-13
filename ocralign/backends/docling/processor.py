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

import contextlib
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, List, Optional

from ocralign.backends.docling.adapter import convert_result
from ocralign.core.schema import Document, Page

logger = logging.getLogger(__name__)

# Docling and its OCR engines log per-conversion progress at INFO
# ("Going to convert document batch...", "Finished converting..."),
# which floods the console when the parallel path runs one convert()
# per page. Cap them at WARNING; set OCRALIGN_VERBOSE=1 to keep them.
_NOISY_LOGGERS = ("docling", "docling_ibm_models", "docling_core", "RapidOCR")


def _min_warning_filter(record: logging.LogRecord) -> bool:
    return record.levelno >= logging.WARNING


@contextlib.contextmanager
def _mute_native_stderr():
    """
    Temporarily silence writes to the stderr file descriptor. Needed for
    C++-side messages that bypass Python's sys.stderr entirely.
    """
    import sys

    try:
        fd = sys.stderr.fileno()
    except Exception:
        yield
        return
    saved = os.dup(fd)
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        sys.stderr.flush()
        os.dup2(devnull, fd)
        os.close(devnull)
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved, fd)
        os.close(saved)


def _quiet_dependency_logs() -> None:
    if os.getenv("OCRALIGN_VERBOSE"):
        return
    # A filter, not just setLevel: rapidocr's Logger class calls
    # setLevel(INFO) on its logger every time one of its modules
    # instantiates it (i.e. repeatedly, during engine construction),
    # which would overwrite any level we set here. Filters on the named
    # logger persist across those calls.
    for name in _NOISY_LOGGERS:
        lg = logging.getLogger(name)
        lg.setLevel(logging.WARNING)
        if _min_warning_filter not in lg.filters:
            lg.addFilter(_min_warning_filter)
    # The "Loading weights" bar printed while docling loads its layout
    # model comes from transformers/huggingface_hub; this env var is
    # their documented off switch (read at import, so set it early).
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    # onnxruntime emits C++-side warnings (e.g. GPU device-discovery
    # probes on CPU-only machines) straight to the stderr fd, bypassing
    # Python logging. The probe runs while the ORT environment is being
    # created — i.e. during this very import/call — so the fd must be
    # muted around it; the severity floor then covers everything later.
    try:
        with _mute_native_stderr():
            import onnxruntime

            onnxruntime.set_default_logger_severity(3)
    except Exception:
        pass


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
    num_threads: Optional[int] = None,
):
    # Quiet BEFORE importing docling: huggingface_hub/transformers read
    # the progress-bar env var at import time. Applies in the parent and
    # in each spawned worker — every code path that talks to docling
    # builds its converter through here.
    _quiet_dependency_logs()

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
    # num_threads governs BOTH the torch models (layout/tableformer) and
    # the OCR engine's ONNX Runtime intra-op pool — docling propagates
    # AcceleratorOptions.num_threads to each. Docling's default is 4.
    if num_threads is not None:
        po.accelerator_options = AcceleratorOptions(device=device, num_threads=num_threads)
    else:
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
    num_threads: Optional[int] = None,
    workers: int = 1,
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
        num_threads: Thread cap for the layout/table models (torch) and
                     the OCR engine's ONNX Runtime sessions. None keeps
                     docling's default of 4. With workers > 1 and no
                     explicit value, defaults to cpu_count() // workers.
        workers: Number of worker processes for page-level parallelism
                 (CPU strategy). Each worker loads its own model copies
                 (~1-1.5 GB RSS each) and converts pages of the shared
                 PDF via page ranges; results merge into one Document.
                 Requires device="cpu" — on GPU, workers would contend
                 for the same device; keep workers=1 there.
                 IMPORTANT: workers > 1 starts processes via "spawn",
                 which re-imports the calling script — the caller MUST
                 invoke this from under `if __name__ == "__main__":`
                 (standard multiprocessing requirement), otherwise the
                 pool dies with "A process in the process pool was
                 terminated abruptly".

    Returns:
        Document. page.text is markdown (headings, pipe tables,
        reading-order-correct paragraphs); ocralign.locate() resolves
        character spans back to page coordinates as with any backend.
    """
    _validate(ocr_engine, device)
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    if workers > 1 and device != "cpu":
        raise ValueError(
            'workers > 1 is a CPU parallelism strategy and requires device="cpu"; '
            "on GPU keep workers=1 (processes would contend for the same device)."
        )

    logger.info(
        f"Docling backend processing: {pdf_path} "
        f"(ocr={ocr_engine}, device={device}, workers={workers})"
    )

    if workers > 1:
        from ocralign.backends.docling.parallel import process_pdf_parallel

        if num_threads is None:
            import os
            num_threads = max(1, (os.cpu_count() or workers) // workers)
        converter_config = dict(
            ocr_engine=ocr_engine,
            device=device,
            tables=tables,
            lang=lang,
            force_ocr=force_ocr,
            num_threads=num_threads,
        )
        return process_pdf_parallel(pdf_path, workers, converter_config)

    import sys
    import time

    t0 = time.perf_counter()
    converter = _build_converter(ocr_engine, device, tables, lang, force_ocr, num_threads)
    t_load = time.perf_counter() - t0
    t1 = time.perf_counter()
    result = converter.convert(pdf_path)
    doc = convert_result(result)
    t_convert = time.perf_counter() - t1

    n = max(1, len(doc.pages))
    print(
        f"{len(doc.pages)} pages in {t_convert:.1f}s "
        f"({t_convert / n:.2f}s/page, +{t_load:.1f}s model load)",
        file=sys.stderr,
    )
    return doc


def process_image(
    page_image: Any,
    ocr_engine: str = "tesseract",
    device: str = "cpu",
    tables: bool = True,
    lang: Optional[List[str]] = None,
    num_threads: Optional[int] = None,
) -> Page:
    """
    Process a single page image into a Page with structural markdown
    text and word-level coordinate mapping.

    Args:
        page_image: Path to an image file, or a PIL Image (written to a
                    temporary file for Docling, which converts by path).
        Other args: as in process_pdf. force_ocr is implied — an image
        has no text layer. No workers param: a single page has nothing
        to parallelize over.
    """
    _validate(ocr_engine, device)
    converter = _build_converter(ocr_engine, device, tables, lang, force_ocr=False,
                                 num_threads=num_threads)

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
