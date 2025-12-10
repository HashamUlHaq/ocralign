import logging
from typing import List, Optional

import fitz  # PyMuPDF
from PIL import Image
from tqdm import tqdm

from ocralign.tess_align import process_page

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)


def write_to_txt(output_path: str, text_pages: List[str]) -> None:
    full_doc_text = ""
    for pg_no, img_text in enumerate(text_pages):
        full_doc_text += f"-- Page {pg_no + 1} --\n"
        full_doc_text += img_text.strip() + "\n\n"

    with open(output_path, "w") as f_:
        f_.write(full_doc_text)

    logger.info(f"Output written to file {output_path}.")


def _page_to_pil_image(page: "fitz.Page", dpi: int) -> Image.Image:
    """
    Render a single PyMuPDF page to a PIL.Image at the requested DPI.
    """
    # PDF user space is 72 DPI; scale matrix accordingly.
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)

    mode = "RGBA" if pix.alpha else "RGB"
    img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
    return img


def process_pdf(
    pdf_path: str,
    dpi: int = 300,
    output_path: Optional[str] = None,
) -> Optional[List[str]]:
    """
    Process a PDF file by converting each page to an image and extracting text using OCR.

    Uses PyMuPDF (fitz) instead of pdf2image for significantly lower memory usage
    and better performance on large files.

    Args:
        pdf_path (str): Path to the input PDF file.
        dpi (int): Dots per inch for image rendering. Higher DPI gives better OCR results.
        output_path (str, optional): If provided, writes concatenated text to this file
                                     instead of returning the list of page texts.

    Returns:
        Optional[List[str]]: A list of strings where each string contains the OCR-extracted
                             text from one page. Returns None if output_path is provided.
    """
    doc = None
    try:
        logger.info(f"Starting PDF processing for: {pdf_path}")
        doc = fitz.open(pdf_path)
        page_count = doc.page_count
        logger.info(f"Opened PDF with {page_count} page(s)")

        text_pages: List[str] = []

        # Iterate pages one-by-one to keep memory footprint low
        for page_index in tqdm(range(page_count), desc="Processing Pages"):
            logger.debug(f"Processing page {page_index + 1}")
            page = doc.load_page(page_index)
            image = _page_to_pil_image(page, dpi=dpi)
            text = process_page(image)
            text_pages.append(text)
            logger.debug(f"Extracted text from page {page_index + 1}")

        if output_path:
            write_to_txt(output_path, text_pages)
            return None
        else:
            return text_pages

    except Exception as e:
        logger.error(f"Failed to process PDF: {e}", exc_info=True)
        raise
    finally:
        if doc is not None:
            doc.close()


def process_image(image_path: str) -> str:
    """
    Process a single image file with OCR.

    This keeps the existing behavior intact; if `process_page` already
    accepts a path, this remains a thin passthrough.
    """
    return process_page(image_path)
