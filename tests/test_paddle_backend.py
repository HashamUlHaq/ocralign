"""Paddle backend: client, payload -> Document assembly, dispatch.

No service, GPU or real document is needed: the service's JSON is built by hand and the HTTP calls
are monkeypatched.
"""
import json
import os
import subprocess
import sys

import pytest
import requests
from PIL import Image

import ocralign
from ocralign import locate_substring
from ocralign.backends import paddle
from ocralign.backends.paddle import client
from ocralign.backends.paddle.render import render_page


def _block(label, content="", order=0, bbox=(0, 0, 10, 10)):
    return {"label": label, "content": content, "bbox": list(bbox) if bbox else bbox, "order": order}


def _page(i, *blocks, width=1000, height=2000):
    return {"page_index": i, "width": width, "height": height, "blocks": list(blocks)}


class _Resp:
    def __init__(self, status, payload=None, text=None):
        self.status_code = status
        self._payload = payload
        self.text = text if text is not None else json.dumps(payload)

    def json(self):
        if self._payload is None:
            raise ValueError("not JSON")
        return self._payload


def _refused(*a, **k):
    raise requests.ConnectionError("refused")


# ---- payload -> Document --------------------------------------------------------------------------
def test_every_page_is_emitted_in_order_even_when_the_service_skips_one():
    """Page N must always mean the N-th page of the source."""
    doc = paddle.payload_to_document({"page_count": 3, "pages": [
        _page(2, _block("text", "three")), _page(0, _block("text", "one"))]})
    assert [p.page_number for p in doc.pages] == [1, 2, 3]
    assert [p.text for p in doc.pages] == ["one", "", "three"]
    gap = doc.pages[1]
    assert (gap.blocks, gap.words, gap.width_px, gap.height_px) == ([], [], None, None)


def test_page_count_and_page_records_both_extend_the_document():
    assert len(paddle.payload_to_document({"page_count": 3, "pages": [_page(0)]}).pages) == 3
    assert len(paddle.payload_to_document({"page_count": 1, "pages": [_page(2)]}).pages) == 3


def test_zero_pages_is_an_error_not_an_empty_document():
    with pytest.raises(ValueError):
        paddle.payload_to_document({"page_count": 0, "pages": []})


def test_block_offsets_slice_exactly_their_segment_of_the_page_text():
    blocks = [
        _block("header", "Example Family Clinic\nTrusted Care", 0),
        _block("header_image", "", 1),
        _block("doc_title", "Progress Note", 2),
        _block("text", "", 3),
        _block("text", "Alert and oriented $ \\times 3 $", 4),
        _block("table", "<table><tr><td>Na</td><td>139</td></tr></table>", 5),
        _block("image", "", 6),
        _block("footer", "Page 1 of 1", 7),
    ]
    page = paddle.payload_to_document({"page_count": 1, "pages": [_page(0, *blocks)]}).pages[0]

    assert page.text == render_page(blocks)
    assert page.words == []
    assert [b.label for b in page.blocks] == [b["label"] for b in blocks]   # every region, in order
    emitted = [b for b in page.blocks if b.char_start is not None]
    for b in emitted:
        assert page.text[b.char_start:b.char_end] == b.text
    assert "\n\n".join(b.text for b in emitted) == page.text
    silent = [b for b in page.blocks if b.char_start is None]
    assert [b.label for b in silent] == ["header_image", "text"]
    assert all(b.text == "" and b.char_end is None and b.bbox is not None for b in silent)


def test_bboxes_are_normalized_top_left_and_clamped():
    page = paddle.payload_to_document({"page_count": 1, "pages": [_page(
        0,
        _block("text", "inside", 0, bbox=(100, 200, 500, 1000)),
        _block("text", "overhang", 1, bbox=(-5, 0, 1200, 2100)),
        _block("text", "no box", 2, bbox=None),
        width=1000, height=2000)]}).pages[0]
    assert [b.bbox for b in page.blocks] == [(0.1, 0.1, 0.5, 0.5), (0.0, 0.0, 1.0, 1.0), None]
    assert (page.width_px, page.height_px) == (1000, 2000)


def test_a_page_without_dimensions_still_renders_text_but_has_no_boxes():
    page = paddle.payload_to_document({"page_count": 1, "pages": [
        {"page_index": 0, "blocks": [_block("text", "hello")]}]}).pages[0]
    assert page.text == "hello"
    assert page.blocks[0].bbox is None and (page.width_px, page.height_px) == (None, None)


def test_locate_resolves_a_span_to_the_enclosing_block_box():
    page = paddle.payload_to_document({"page_count": 1, "pages": [_page(
        0,
        _block("header", "Example Family Clinic", 0, bbox=(0, 0, 1000, 100)),
        _block("text", "Diagnosis: type 2 diabetes", 1, bbox=(100, 400, 900, 500)),
        width=1000, height=2000)]}).pages[0]
    assert locate_substring(page, "type 2 diabetes") == [[(0.1, 0.2, 0.9, 0.25)]]
    assert locate_substring(page, "Clinic Diagnosis") == [[(0.0, 0.0, 1.0, 0.05), (0.1, 0.2, 0.9, 0.25)]]


def test_document_json_round_trip_keeps_blocks():
    doc = paddle.payload_to_document({"page_count": 2, "pages": [
        _page(0, _block("header", "Clinic", 0), _block("header_image", "", 1))]})
    assert ocralign.Document.from_dict(json.loads(json.dumps(doc.to_dict()))) == doc


# ---- client ---------------------------------------------------------------------------------------
def test_client_sends_only_the_extension_never_the_filename(monkeypatch, tmp_path):
    """Filenames can carry personal names; a request line is what access logs record."""
    chart = tmp_path / "Doe, Jane 01.01.1950 DOS 01.02.25-02.03.25.pdf"
    chart.write_bytes(b"%PDF-1.4 fake")
    calls = {}

    def fake_post(url, params, data, headers, timeout):
        calls.update(url=url, params=params, data=data, timeout=timeout)
        return _Resp(200, {"page_count": 1, "pages": [_page(0, _block("text", "hello"))]})

    monkeypatch.setattr(client.requests, "post", fake_post)
    doc = ocralign.process_pdf(str(chart), backend="paddle", url="http://127.0.0.1:8081/", timeout=5)
    assert [p.text for p in doc.pages] == ["hello"]
    assert calls["url"] == "http://127.0.0.1:8081/v1/ocr" and calls["params"] == {"ext": "pdf"}
    assert "Doe" not in calls["url"] + json.dumps(calls["params"])
    assert calls["data"] == b"%PDF-1.4 fake"
    assert calls["timeout"] == (10, 5)


def test_service_down_raises_with_the_remedy(monkeypatch, tmp_path):
    chart = tmp_path / "c.pdf"
    chart.write_bytes(b"x")
    monkeypatch.setattr(client.requests, "post", _refused)
    with pytest.raises(paddle.PaddleOCRUnavailable, match="stack.sh up"):
        paddle.fetch_payload(str(chart), url="http://127.0.0.1:8081")
    monkeypatch.setattr(client.requests, "get", _refused)
    with pytest.raises(paddle.PaddleOCRUnavailable, match="stack.sh up"):
        paddle.check_service("http://127.0.0.1:8081")


def test_health_check_passes_only_on_200(monkeypatch):
    monkeypatch.setattr(client.requests, "get", lambda url, timeout: _Resp(200, {"status": "ok"}))
    paddle.check_service("http://127.0.0.1:8081/")
    monkeypatch.setattr(client.requests, "get", lambda url, timeout: _Resp(503, {"detail": "loading"}))
    with pytest.raises(paddle.PaddleOCRUnavailable, match="health HTTP 503"):
        paddle.check_service("http://127.0.0.1:8081")


def test_service_error_is_not_mistaken_for_service_down(monkeypatch, tmp_path):
    chart = tmp_path / "c.pdf"
    chart.write_bytes(b"x")
    monkeypatch.setattr(client.requests, "post",
                        lambda *a, **k: _Resp(500, {"detail": "ocr failed: ValueError"}))
    with pytest.raises(RuntimeError, match="HTTP 500: ocr failed: ValueError") as e:
        paddle.fetch_payload(str(chart))
    assert not isinstance(e.value, paddle.PaddleOCRUnavailable)
    monkeypatch.setattr(client.requests, "post", lambda *a, **k: _Resp(502, text="Bad Gateway"))
    with pytest.raises(RuntimeError, match="HTTP 502: Bad Gateway"):
        paddle.fetch_payload(str(chart))


# ---- process_image --------------------------------------------------------------------------------
def _capture_post(monkeypatch):
    calls = {}

    def fake_post(url, params, data, headers, timeout):
        calls.update(params=params, data=data)
        return _Resp(200, {"page_count": 1, "pages": [_page(0, _block("text", "img text"))]})

    monkeypatch.setattr(client.requests, "post", fake_post)
    return calls


def test_process_image_sends_a_pil_image_as_png(monkeypatch):
    calls = _capture_post(monkeypatch)
    page = ocralign.process_image(Image.new("RGB", (20, 10), "white"), backend="paddle")
    assert (page.page_number, page.text) == (1, "img text")
    assert calls["params"] == {"ext": "png"} and calls["data"].startswith(b"\x89PNG")


def test_process_image_sends_a_file_with_its_own_extension(monkeypatch, tmp_path):
    calls = _capture_post(monkeypatch)
    path = tmp_path / "scan.JPG"
    Image.new("RGB", (20, 10), "white").save(path, format="JPEG")
    page = ocralign.process_image(path, backend="paddle")
    assert page.text == "img text" and calls["params"] == {"ext": "jpg"}


def test_process_image_refuses_other_inputs():
    with pytest.raises(TypeError, match="path or a PIL Image"):
        paddle.process_image(b"raw bytes")


# ---- dispatch & packaging -------------------------------------------------------------------------
def test_unknown_backend_error_lists_paddle():
    with pytest.raises(ValueError, match="vanilla, docling, paddle"):
        ocralign.process_pdf("x.pdf", backend="padle")


def test_service_assets_ship_with_the_package():
    assert {"Dockerfile.client", "backend_config.yaml", "service.py", "stack.sh"} <= \
        set(os.listdir(paddle.SERVICE_DIR))
    assert paddle.STACK_SCRIPT == os.path.join(paddle.SERVICE_DIR, "stack.sh")


def test_module_entry_point_runs_the_stack_script():
    """No subcommand -> the script's own usage error; nothing is started."""
    r = subprocess.run([sys.executable, "-m", "ocralign.backends.paddle"],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and "stack.sh up|down|status" in r.stderr
