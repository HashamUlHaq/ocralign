import logging
from pdf2image import convert_from_path
from typing import List
from ocralign.tess_align import process_page
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


def write_to_txt(output_path, text_pages):
    full_doc_text = ""
    for pg_no, img_text in enumerate(text_pages):
        full_doc_text += f"-- Page {pg_no + 1} --\n"
        full_doc_text += img_text.strip() + "\n\n"
    
    with open(output_path, "w") as f_:
        f_.write(full_doc_text)

    logger.info(f"Output written to file {output_path}.")

def process_pdf(pdf_path: str, dpi: int = 300, output_path=None) -> List[str]:
    """
    Process a PDF file by converting each page to an image and extracting text using OCR.

    Args:
        pdf_path (str): Path to the input PDF file.
        dpi (int): Dots per inch setting for image conversion. Higher DPI gives better OCR results.

    Returns:
        List[str]: A list of strings where each string contains the OCR-extracted text from one page.
    """
    try:
        logger.info(f"Starting PDF processing for: {pdf_path}")
        images = convert_from_path(pdf_path, dpi=dpi)
        logger.info(f"Converted PDF to {len(images)} image(s)")

        text_pages = []
        for i, image in enumerate(tqdm(images, desc="Processing Pages")):
            logger.debug(f"Processing page {i + 1}")
            text = process_page(image)
            text_pages.append(text)
            logger.debug(f"Extracted text from page {i + 1}")

        if output_path:
            write_to_txt(output_path, text_pages)

        else:
            return text_pages

    except Exception as e:
        logger.error(f"Failed to process PDF: {e}", exc_info=True)
        raise

def process_image(image_path: str) -> str:
    return process_page(image_path)

