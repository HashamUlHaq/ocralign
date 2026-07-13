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
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

from tqdm import tqdm

from ocralign.core.schema import Document, Page

# Per-worker-process converter, created once by _init_worker.
_CONVERTER = None


def _init_worker(converter_config: Dict) -> None:
    global _CONVERTER
    # Quiet dependency logging FIRST — before docling/onnxruntime are
    # imported and initialized — so the ORT native-severity floor is in
    # place before its environment (and its GPU-probe warnings) exists.
    from ocralign.backends.docling.processor import _build_converter, _quiet_dependency_logs

    _quiet_dependency_logs()
    _CONVERTER = _build_converter(**converter_config)


def _convert_range(task: Tuple[str, int, int]) -> Tuple[List[Page], float]:
    pdf_path, first, last = task
    from ocralign.backends.docling.adapter import convert_result

    t0 = time.perf_counter()
    result = _CONVERTER.convert(pdf_path, page_range=(first, last))
    pages = convert_result(result).pages
    return pages, time.perf_counter() - t0


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
    pages: List[Page] = []
    durations: List[float] = []
    t_start = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(converter_config,),
    ) as executor:
        futures = [executor.submit(_convert_range, t) for t in tasks]
        with tqdm(total=n_pages, desc="Processing Pages", unit="page") as bar:
            for future in as_completed(futures):
                pages_part, duration = future.result()
                pages.extend(pages_part)
                durations.append(duration)
                bar.set_postfix_str(
                    f"last {duration:.1f}s, avg {sum(durations) / len(durations):.1f}s/page"
                )
                bar.update(1)
    total = time.perf_counter() - t_start

    print(
        f"{n_pages} pages in {total:.1f}s "
        f"({total / n_pages:.2f}s/page wall with {workers} workers, "
        f"{sum(durations) / n_pages:.2f}s/page compute)",
        file=sys.stderr,
    )

    pages.sort(key=lambda p: p.page_number)
    return Document(pages=pages)
