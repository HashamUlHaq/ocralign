import logging
from typing import List, Optional

import fitz  # PyMuPDF
from PIL import Image
from tqdm import tqdm

from ocralign.tess_align import process_page
from ocralign.digital_pdf_align import page_to_layout_text

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)

def _add_marker(text_pages: List[str]) -> List[str]:
    return [ f"-- Page {pg_no + 1} --\n{img_text}"+ "\n\n" for pg_no, img_text in enumerate(text_pages)]
        
def write_to_txt(output_path: str, text_pages: List[str]) -> None:

    full_doc_text = "".join(text_pages)

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

def process_image_pdf(
    pdf_path: str,
    add_marker: bool=True,
    dpi: int = 300,
    output_path: Optional[str] = None,
) -> Optional[List[str]]:

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

        if add_marker:
            text_pages = _add_marker(text_pages)

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

def process_digital_pdf(pdf_path: str, enforce_layout: bool, add_marker: bool, output_path: Optional[str] = None) -> Optional[List[str]]:
    doc = None
    try:
        logger.info(f"Starting PDF processing for: {pdf_path}")
        doc = fitz.open(pdf_path)
        page_count = doc.page_count
        logger.info(f"Opened PDF with {page_count} page(s)")
        
        text_pages: List[str] = []
        for i in range(doc.page_count):
            page = doc.load_page(i)
            if enforce_layout: 
                text_pages.append(page_to_layout_text(page, cols=140))
            else:
                text_pages.append(page.get_text())

        if add_marker:
            text_pages = _add_marker(text_pages)

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

def process_pdf(
    pdf_path: str,
    type: str="image",
    enforce_layout: bool = True,
    add_marker: bool = True,
    dpi: int = 300,
    output_path: Optional[str] = None,
) -> Optional[List[str]]:
    """
    Process a PDF file by converting each page to an image and extracting text using OCR.

    Uses PyMuPDF (fitz) instead of pdf2image for significantly lower memory usage
    and better performance on large files.

    Args:
        pdf_path (str): Path to the input PDF file.
        type (str): "image" for scanned or "digital" for pdfs with retrievable text.
        dpi (int): Dots per inch for image rendering. Higher DPI gives better OCR results.
        output_path (str, optional): If provided, writes concatenated text to this file
                                     instead of returning the list of page texts.

    Returns:
        Optional[List[str]]: A list of strings where each string contains the OCR-extracted
                             text from one page. Returns None if output_path is provided.
    """
    
    if type == "image":
        return process_image_pdf(pdf_path, add_marker, dpi, output_path)
    elif type == "digital":
        return process_digital_pdf(pdf_path, enforce_layout, add_marker, output_path)
    else:
        raise Exception ('Invalid document type. Only "digital" or "image" is allowed')

def process_image(image_path: str) -> str:
    """
    Process a single image file with OCR.

    This keeps the existing behavior intact; if `process_page` already
    accepts a path, this remains a thin passthrough.
    """
    return process_page(image_path)
