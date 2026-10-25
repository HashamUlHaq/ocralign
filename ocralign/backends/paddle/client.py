"""
Paddle backend client: talks to the PaddleOCR-VL service over HTTP.

The engine runs as a service (service/service.py beside this module, started by service/stack.sh)
and this package never imports paddle: paddlepaddle-gpu's CUDA wheels conflict with the torch most
ML environments already ship, so the two never share an interpreter.

Fail loud. If the service is down the error says how to start it; there is no silent fallback to
another engine, because a fallback would change every downstream output without telling anyone.
"""

from __future__ import annotations

import os

import requests

DEFAULT_URL = "http://127.0.0.1:8081"

# The service ships with the package (package data, never imported) so every consumer starts the
# same checked-in code: `bash STACK_SCRIPT up`.
SERVICE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "service")
STACK_SCRIPT = os.path.join(SERVICE_DIR, "stack.sh")


class PaddleOCRUnavailable(RuntimeError):
    """The OCR service cannot be reached or is not healthy."""


def _unavailable(url: str, why: str) -> str:
    return (
        f"PaddleOCR service not usable at {url} ({why}).\n"
        f"  - start the OCR stack:  bash {STACK_SCRIPT} up\n"
        "    it needs ~13 GB of free GPU memory to start; stop other large GPU models first\n"
        "  - or use another backend, e.g. process_pdf(path, backend=\"vanilla\")"
    )


def check_service(url: str = DEFAULT_URL, timeout: float = 5.0) -> None:
    """Raise PaddleOCRUnavailable unless the service reports healthy."""
    base = url.rstrip("/")
    try:
        r = requests.get(f"{base}/health", timeout=timeout)
    except requests.RequestException as e:
        raise PaddleOCRUnavailable(_unavailable(base, type(e).__name__)) from e
    if r.status_code != 200:
        raise PaddleOCRUnavailable(_unavailable(base, f"health HTTP {r.status_code}: {r.text[:200]}"))


def post_document(data: bytes, ext: str, *, url: str = DEFAULT_URL, timeout: float = 3600.0) -> dict:
    """
    Document bytes -> the service's raw per-page layout blocks.

    ext is the file extension without the dot ("pdf", "png", ...); the service picks its decoder
    from it. timeout is per document: a 461-page scan took ~5 minutes.
    """
    base = url.rstrip("/")
    try:
        # Only the extension goes in the URL. Filenames can carry personal names, and a request
        # line is exactly what access logs record.
        r = requests.post(f"{base}/v1/ocr", params={"ext": ext}, data=data,
                          headers={"Content-Type": "application/octet-stream"},
                          timeout=(10, timeout))
    except requests.ConnectionError as e:
        raise PaddleOCRUnavailable(_unavailable(base, type(e).__name__)) from e
    if r.status_code != 200:
        try:
            detail = r.json().get("detail")
        except (ValueError, AttributeError):
            detail = r.text[:200]
        raise RuntimeError(f"PaddleOCR service returned HTTP {r.status_code}: {detail}")
    return r.json()


def fetch_payload(path: str, *, url: str = DEFAULT_URL, timeout: float = 3600.0) -> dict:
    """
    Document file -> the service's raw per-page layout blocks (what payload_to_document consumes).

    Kept separate from rendering so a caller can store the payload: OCR costs minutes per long
    document, rendering costs milliseconds, and a renderer fix should never force a re-OCR.
    """
    ext = os.path.splitext(str(path))[1].lower().lstrip(".")
    with open(path, "rb") as fh:
        data = fh.read()
    return post_document(data, ext, url=url, timeout=timeout)
