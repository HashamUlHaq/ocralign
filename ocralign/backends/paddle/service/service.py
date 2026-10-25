"""PaddleOCR-VL document service — the engine of ocralign's paddle backend, behind one HTTP call.

Runs INSIDE the `paddleocr-vl-client` image (see Dockerfile.client beside this file), not in the
caller's environment. That split is deliberate: PaddleOCR needs paddlepaddle-gpu, whose CUDA wheels
conflict with the torch most ML environments already ship, so the two never share an interpreter.
Callers talk to this over HTTP (ocralign/backends/paddle/client.py) and never import paddle.

Two services make up the OCR stack, started together by stack.sh beside this file:

    :8080  paddleocr_server    vLLM serving the PaddleOCR-VL-1.6-0.9B recognition model
    :8081  paddleocr_pipeline  THIS: layout detection (PP-DocLayoutV3) + block reading order,
                               sending each region to :8080 for recognition

The pipeline object is built ONCE at startup and reused for every request. Building it per
document costs ~10 s of model load; over the 143-chart corpus that was ~24 min of pure overhead.

API
    GET  /health            200 when the layout pipeline is loaded AND the VL server answers
    POST /v1/ocr?ext=pdf     body = raw document bytes -> per-page layout blocks as JSON

Only the file EXTENSION travels in the URL. Chart filenames carry patient names, and a request
line is exactly what access logs record, so the name never leaves the client. Access logging is
off regardless, request bodies go to a temp file that is removed in `finally`, and no log line
contains document text.

Recognition settings are the ones validated corpus-wide (8,579 pages, 0 failures, 3 empty pages):
layout detection on; orientation classify, unwarping, chart and seal recognition off. Do not turn
those on without re-validating — each adds a model and none was measured here.
"""
from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
from contextlib import asynccontextmanager

import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from starlette.concurrency import run_in_threadpool

VL_URL = os.environ.get("VL_URL", "http://127.0.0.1:8080/v1")
HOST = os.environ.get("OCR_SERVICE_HOST", "127.0.0.1")   # PHI: loopback only, never 0.0.0.0
PORT = int(os.environ.get("OCR_SERVICE_PORT", "8081"))
MAX_BYTES = int(os.environ.get("OCR_MAX_BYTES", str(512 * 1024 * 1024)))
ENGINE = "PaddleOCR-VL-1.6-0.9B"

ALLOWED_EXT = {"pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("paddle_ocr_service")

_state: dict = {"pipeline": None}
# PaddleOCRVL.predict is not documented as re-entrant, so documents are processed one at a time.
# That is not the throughput limit it looks like: WITHIN a document the pipeline already sends its
# regions to the vLLM server concurrently, which is where the time goes.
_lock = threading.Lock()


def _build_pipeline():
    from paddleocr import PaddleOCRVL
    return PaddleOCRVL(
        vl_rec_backend="vllm-server",
        vl_rec_server_url=VL_URL,
        # Keep EVERY block. The upstream default drops header/footer/number blocks from its
        # markdown; we don't use its markdown at all, but say it anyway so nothing is filtered.
        markdown_ignore_labels=[],
    )


@asynccontextmanager
async def _lifespan(app: FastAPI):
    t = time.time()
    _state["pipeline"] = _build_pipeline()
    log.info("layout pipeline ready in %.1fs (vl=%s)", time.time() - t, VL_URL)
    yield


app = FastAPI(title="paddle-ocr-service", lifespan=_lifespan)


def _vl_healthy() -> bool:
    base = VL_URL[:-3] if VL_URL.endswith("/v1") else VL_URL
    try:
        return requests.get(f"{base}/health", timeout=3).status_code == 200
    except requests.RequestException:
        return False


@app.get("/health")
def health():
    ok_pipeline = _state["pipeline"] is not None
    ok_vl = _vl_healthy()
    if not (ok_pipeline and ok_vl):
        raise HTTPException(status_code=503,
                            detail={"pipeline_loaded": ok_pipeline, "vl_server": ok_vl, "vl_url": VL_URL})
    return {"status": "ok", "engine": ENGINE, "vl_url": VL_URL}


def _page_record(res_json: dict, fallback_index: int) -> dict:
    """Reduce one PaddleOCRVL page result to what the pipeline renders."""
    d = res_json.get("res", res_json)
    idx = d.get("page_index")
    blocks = []
    for i, b in enumerate(d.get("parsing_res_list") or []):
        blocks.append({
            "label": b.get("block_label"),
            "content": b.get("block_content") or "",
            "bbox": b.get("block_bbox"),
            "order": i,                     # parsing_res_list is already in reading order
        })
    return {
        "page_index": fallback_index if idx is None else int(idx),
        "width": d.get("width"),
        "height": d.get("height"),
        "blocks": blocks,
        "_page_count": d.get("page_count"),
    }


def _ocr_blocking(body: bytes, ext: str) -> dict:
    """The actual OCR. Runs in a worker thread (see `ocr`), never on the event loop."""
    fd, path = tempfile.mkstemp(suffix=f".{ext}")
    t0 = time.time()
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(body)
        pages, declared = [], None
        with _lock:
            # predict_iter, not predict: a 400-page chart otherwise holds every page's rendered
            # image in memory at once. Keep only the reduced record and let the result go.
            for i, res in enumerate(_state["pipeline"].predict_iter(path, markdown_ignore_labels=[])):
                rec = _page_record(res.json, i)
                page_count_field = rec.pop("_page_count", None)
                declared = declared or page_count_field
                pages.append(rec)
        pages.sort(key=lambda p: p["page_index"])
        page_count = max(int(declared or 0), len(pages))
        dt = time.time() - t0
        # counts and timing only — never text, never a filename
        log.info("ocr ok: ext=%s pages=%d blocks=%d %.1fs", ext, page_count,
                 sum(len(p["blocks"]) for p in pages), dt)
        return {"engine": ENGINE, "page_count": page_count, "pages": pages,
                "seconds": round(dt, 2)}
    except Exception as e:
        # The exception TYPE only: a message could quote document content.
        log.error("ocr failed: ext=%s %s after %.1fs", ext, type(e).__name__, time.time() - t0)
        raise HTTPException(status_code=500, detail=f"ocr failed: {type(e).__name__}")
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


@app.post("/v1/ocr")
async def ocr(request: Request, ext: str = Query(..., description="file extension, e.g. pdf")):
    ext = ext.lower().lstrip(".")
    if ext not in ALLOWED_EXT:
        raise HTTPException(status_code=415, detail=f"unsupported extension {ext!r}")
    # Reject on the DECLARED size first: `await request.body()` buffers the whole upload, so a
    # check after it cannot prevent the memory blow-up the limit exists to stop.
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_BYTES:
        raise HTTPException(status_code=413, detail=f"document exceeds {MAX_BYTES} bytes")
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="empty request body")
    # A chunked upload declares no length, so the limit is still enforced after reading.
    if len(body) > MAX_BYTES:
        raise HTTPException(status_code=413, detail=f"document exceeds {MAX_BYTES} bytes")
    if _state["pipeline"] is None:
        raise HTTPException(status_code=503, detail="layout pipeline not loaded")
    # OCR of a large chart takes minutes. Running it inline in this async handler would block the
    # event loop, so /health would stop answering for the duration — and a health check that
    # hangs whenever the service is busy looks exactly like a service that is down.
    return await run_in_threadpool(_ocr_blocking, body, ext)


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, access_log=False, log_level="info")
