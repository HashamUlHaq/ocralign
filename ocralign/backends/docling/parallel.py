"""
Page-level process parallelism for the docling backend (CPU strategy).

N worker processes each build one DocumentConverter (models loaded once
per worker, amortized over the whole job) and convert pages of the
shared PDF via docling's page_range — no manual splitting, no
rasterization, so the born-digital fast path is preserved. Docling
keeps original page numbers within a range (verified: page_range=(2,2)
yields prov page_no 2), so the parent just merges the returned Pages
by page_number.

Workers use the "spawn" start method: forking a process that may
already hold torch/ONNX Runtime thread pools is unsafe.

Sizing: workers x num_threads should not exceed physical cores (the
processor defaults num_threads to cpu_count() // workers). Each worker
holds its own model copies, ~1-1.5 GB RSS.
"""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Tuple

from ocralign.core.schema import Document, Page

# Per-worker-process converter, created once by _init_worker.
_CONVERTER = None


def _init_worker(converter_config: Dict) -> None:
    global _CONVERTER
    from ocralign.backends.docling.processor import _build_converter

    _CONVERTER = _build_converter(**converter_config)


def _convert_range(task: Tuple[str, int, int]) -> List[Page]:
    pdf_path, first, last = task
    from ocralign.backends.docling.adapter import convert_result

    result = _CONVERTER.convert(pdf_path, page_range=(first, last))
    return convert_result(result).pages


def process_pdf_parallel(
    pdf_path: str,
    workers: int,
    converter_config: Dict,
) -> Document:
    """
    Convert a PDF with `workers` processes, one page per task (pages
    vary a lot in cost — a page with tables is several times a plain
    one — so single-page tasks give the best load balance).
    """
    import fitz

    with fitz.open(pdf_path) as pdf:
        n_pages = pdf.page_count
    if n_pages == 0:
        return Document(pages=[])

    workers = min(workers, n_pages)
    tasks = [(pdf_path, page_no, page_no) for page_no in range(1, n_pages + 1)]

    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(converter_config,),
    ) as executor:
        page_lists = list(executor.map(_convert_range, tasks))

    pages = [p for page_list in page_lists for p in page_list]
    pages.sort(key=lambda p: p.page_number)
    return Document(pages=pages)
