"""
Paddle backend: PaddleOCR-VL (layout detection + a 0.9B vision-language
recogniser) running as a local GPU service, reached over HTTP.

Produces structural markdown meant for LLM consumption: reading-order
blocks, tagged `[page header]` / `[page footer]`, `#` headings, pipe
tables, LaTeX symbols normalized to Unicode. Geometry is recorded per
layout BLOCK (Page.blocks), not per word — the recogniser reads whole
regions — and locate() falls back to those block boxes.

Optional dependency: pip install "ocralign[paddle]" (the HTTP client
only). The engine runs in Docker, started with
`python -m ocralign.backends.paddle up` (service/stack.sh; needs ~13 GB
of free GPU memory).
"""

from ocralign.backends.paddle.client import (
    DEFAULT_URL,
    SERVICE_DIR,
    STACK_SCRIPT,
    PaddleOCRUnavailable,
    check_service,
    fetch_payload,
    post_document,
)
from ocralign.backends.paddle.processor import payload_to_document, process_image, process_pdf
from ocralign.backends.paddle.render import render_page

__all__ = [
    "process_pdf",
    "process_image",
    "payload_to_document",
    "fetch_payload",
    "post_document",
    "check_service",
    "render_page",
    "PaddleOCRUnavailable",
    "DEFAULT_URL",
    "SERVICE_DIR",
    "STACK_SCRIPT",
]
