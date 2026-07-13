"""
Demo: the docling backend end-to-end.

Processes a two-column PDF (the layout that breaks the vanilla
backend's grid renderer), prints a report of what came back, resolves
a substring to overlay coordinates, and draws the boxes onto a render
of the page as visual proof.

Run from the examples/ directory:
    python demo_docling.py

Requires the optional docling dependencies:
    pip install "ocralign[docling]"

Outputs (written to the current directory):
    docling_demo_output.md   - the structural markdown text per page
    docling_demo_doc.json    - full Document (text + words + offsets)
    docling_demo_overlay.png - page 1 render with overlay boxes drawn

NOTE: the __main__ guard below is REQUIRED when using workers > 1.
Worker processes start via "spawn", which re-imports this script; any
top-level process_pdf() call would recurse into spawning more workers.
"""

import time

import fitz  # PyMuPDF, only used here to render the page image for the overlay proof
from PIL import Image, ImageDraw

from ocralign import locate_substring, process_pdf, to_pixels

PDF = "./23092015_Double Column Research Paper Format.pdf"

# A phrase that lives in the RIGHT column of page 1. Under the vanilla
# backend this sentence gets interleaved word-by-word with the left
# column; under docling it is one clean span in the text.
QUERY = "Author names and affiliations are to be centered beneath the title"


def main():
    # ------------------------------------------------------------ process
    t0 = time.time()
    doc = process_pdf(
        PDF,
        backend="docling",
        ocr_engine="rapidocr",  # or "tesseract"
        device="cpu",           # "cuda" on a GPU box (then keep workers=1)
        workers=1,              # >1 = CPU page-parallelism (needs the __main__ guard)
        num_threads=4,          # per-process thread cap (torch + ONNX Runtime)
    )
    elapsed = time.time() - t0

    print("=== processing report ===")
    print(f"file:      {PDF}")
    print(f"pages:     {len(doc.pages)}")
    print(f"time:      {elapsed:.1f}s ({elapsed / len(doc.pages):.1f}s/page, "
          f"incl. model load)")
    for p in doc.pages:
        mapped = sum(1 for w in p.words if w.char_start is not None)
        print(f"  page {p.page_number}: {len(p.words)} words ({mapped} offset-mapped), "
              f"{len(p.text)} chars of markdown")

    # Offset integrity: every mapped word's recorded span must slice back
    # to exactly its own text. This is the invariant locate() depends on.
    bad = [
        (p.page_number, w.text)
        for p in doc.pages
        for w in p.words
        if w.char_start is not None and p.text[w.char_start:w.char_end] != w.text
    ]
    print(f"offset integrity: {'OK' if not bad else f'FAILED {bad[:5]}'}")

    # ---------------------------------------------------------- page text
    print("\n=== page 1 markdown (first 600 chars) ===")
    print(doc.pages[0].text[:600])

    with open("docling_demo_output.md", "w") as f:
        for p in doc.pages:
            f.write(f"<!-- page {p.page_number} -->\n\n{p.text}\n\n")
    doc.save_json("docling_demo_doc.json", indent=2)

    # ----------------------------------------------------- locate overlay
    print(f"\n=== overlay boxes for: {QUERY!r} ===")
    page = doc.pages[0]
    occurrences = locate_substring(page, QUERY)
    print(f"occurrences found: {len(occurrences)}")

    for boxes in occurrences:
        for x0, y0, x1, y1 in boxes:
            # Normalized fractions (0-1, origin top-left): what you store
            # on the entity record and ship to the frontend.
            print(f"  normalized: x0={x0:.4f} y0={y0:.4f} x1={x1:.4f} y1={y1:.4f}")
            # Pixel positions for any render size, e.g. a react-pdf page
            # rendered 900px wide: left = x0 * width, top = y0 * height.
            px = to_pixels((x0, y0, x1, y1), 900, 900 * page.height_px / page.width_px)
            print(f"  at 900px wide render: left={px[0]:.0f} top={px[1]:.0f} "
                  f"right={px[2]:.0f} bottom={px[3]:.0f}")

    # ----------------------------------------------------- visual proof
    # Render page 1 at an arbitrary zoom (deliberately different from any
    # size used during processing - normalized coords don't care) and
    # draw the boxes on it.
    with fitz.open(PDF) as pdf:
        pix = pdf[0].get_pixmap(dpi=150)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

    draw = ImageDraw.Draw(img)
    for boxes in occurrences:
        for bbox in boxes:
            draw.rectangle(to_pixels(bbox, img.width, img.height), outline="red", width=4)

    img.save("docling_demo_overlay.png")
    print("\nwrote docling_demo_output.md, docling_demo_doc.json, docling_demo_overlay.png")


if __name__ == "__main__":
    main()
