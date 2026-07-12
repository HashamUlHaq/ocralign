"""
Benchmark: Docling layout/table analysis on top of Tesseract OCR.

Purpose: measure how much overhead Docling's layout model, reading-order
model, and table-structure model (TableFormer) add on top of plain
Tesseract OCR, so we can decide whether/how to fold Docling into ocralign
as a second adapter for table- and multi-column-heavy documents.

How Docling's layout model works (confirmed by reading its source,
docling/models/stages/layout/layout_object_detection_model.py):
  1. The page is rasterized to a plain image (page.get_image()).
  2. That IMAGE (pixels only, not OCR text or OCR boxes) is fed to a
     vision object-detection model, which outputs region clusters with
     labels (text, title, table, picture, ...) and bboxes. OCR
     coordinates are NOT an input to this step.
  3. Separately, OCR (here: Tesseract, via TesseractCliOcrOptions) or
     native PDF text extraction produces text cells with their own
     bboxes. These are assigned into the detected regions by bbox
     overlap (see LayoutPostprocessor) -- this is the only place OCR
     output and layout-model output meet.
  4. A separate reading-order model (docling_ibm_models.reading_order)
     takes the region clusters (not raw OCR cells) and predicts the
     order to traverse them -- this is what fixes multi-column
     interleaving.
  5. For clusters labeled TABLE, TableFormer runs on the cropped table
     image to predict row/col structure, then OCR/PDF cells are matched
     into structure cells by bbox overlap (do_cell_matching).

So: two independent, CPU-hungry vision models (layout detection,
table structure) run on top of whatever Tesseract already costs you.
That's the overhead this script measures.

Usage:
    python examples/docling_tesseract_benchmark.py <path-to-pdf-or-image> [--no-ocr] [--no-tables]

    --no-ocr      Skip OCR entirely (only valid for born-digital PDFs;
                  Docling pulls text straight from the PDF instead).
                  Useful to isolate "layout model cost" from "OCR cost".
    --no-tables   Skip TableFormer (do_table_structure=False).
                  Useful to isolate "layout model cost" from "table
                  structure cost".
"""

import argparse
import json
import sys
import time
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
from docling.datamodel.settings import settings
from docling.document_converter import DocumentConverter, ImageFormatOption, PdfFormatOption


def run(input_path: str, do_ocr: bool, do_table_structure: bool) -> None:
    settings.debug.profile_pipeline_timings = True

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = do_ocr
    pipeline_options.do_table_structure = do_table_structure
    if do_table_structure:
        pipeline_options.table_structure_options.do_cell_matching = True
    if do_ocr:
        pipeline_options.ocr_options = TesseractCliOcrOptions(lang=["eng"])

    fmt_option = PdfFormatOption(pipeline_options=pipeline_options)
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: fmt_option,
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_options),
        }
    )

    t0 = time.time()
    result = converter.convert(input_path)
    wall_time = time.time() - t0

    doc = result.document
    num_pages = len(doc.pages)

    print(f"input:              {input_path}")
    print(f"do_ocr:             {do_ocr}")
    print(f"do_table_structure: {do_table_structure}")
    print(f"pages:              {num_pages}")
    print(f"wall time:          {wall_time:.2f}s  ({wall_time / max(num_pages, 1):.2f}s/page)")
    print(f"tables detected:    {len(doc.tables)}")
    print()
    print("--- per-stage timings (seconds, summed across pages) ---")
    stage_totals = {}
    for key, item in sorted(result.timings.items()):
        total = float(sum(item.times))
        stage_totals[key] = total
        print(f"  {key:30s} count={len(item.times):3d}  total={total:7.2f}s  avg={total / len(item.times):6.3f}s")

    accounted = sum(stage_totals.values())
    print()
    print(f"  sum of measured stages: {accounted:.2f}s  (wall time incl. model load / overhead: {wall_time:.2f}s)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="Path to a PDF or image file")
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR (digital PDFs only)")
    parser.add_argument("--no-tables", action="store_true", help="Disable table-structure recognition")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"error: file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    run(args.input, do_ocr=not args.no_ocr, do_table_structure=not args.no_tables)


if __name__ == "__main__":
    main()
