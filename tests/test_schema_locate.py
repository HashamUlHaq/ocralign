"""Schema persistence and locate(), including the block fallback for region-level backends."""
from ocralign import Block, Document, Page, Word, locate, locate_in_document
from ocralign.core.schema import SCHEMA_VERSION


def _block_page(page_number=1):
    # "[page header] Clinic\n\nAlpha beta\n\ngamma" with one logo that adds no text
    return Page(
        page_number=page_number,
        text="[page header] Clinic\n\nAlpha beta\n\ngamma",
        blocks=[
            Block("header", "[page header] Clinic", (0.0, 0.0, 1.0, 0.1), 0, 20),
            Block("header_image", "", (0.8, 0.0, 1.0, 0.1)),
            Block("text", "Alpha beta", (0.1, 0.2, 0.9, 0.3), 22, 32),
            Block("text", "gamma", None, 34, 39),
        ],
    )


def test_block_page_offsets_are_consistent():
    page = _block_page()
    for b in page.blocks:
        if b.char_start is not None:
            assert page.text[b.char_start:b.char_end] == b.text


def test_locate_falls_back_to_block_boxes_when_a_page_has_no_words():
    page = _block_page()
    assert locate(page, 22, 27) == [(0.1, 0.2, 0.9, 0.3)]                        # inside one block
    assert locate(page, 14, 27) == [(0.0, 0.0, 1.0, 0.1), (0.1, 0.2, 0.9, 0.3)]  # across two
    assert locate(page, 20, 22) == []                                           # the separator only
    assert locate(page, 34, 39) == []                                           # block without a box
    assert locate(page, 27, 27) == []                                           # empty span


def test_locate_uses_words_whenever_a_page_has_them():
    page = Page(
        page_number=1,
        text="Alpha beta",
        words=[Word("Alpha", (0.1, 0.2, 0.2, 0.25), line_no=0, char_start=0, char_end=5),
               Word("beta", (0.25, 0.2, 0.3, 0.25), line_no=0, char_start=6, char_end=10)],
        blocks=[Block("text", "Alpha beta", (0.0, 0.0, 1.0, 1.0), 0, 10)],
    )
    assert locate(page, 0, 10) == [(0.1, 0.2, 0.3, 0.25)]


def test_locate_in_document_maps_offsets_onto_block_pages():
    doc = Document(pages=[Page(page_number=1, text="x"), _block_page(page_number=2)])
    full = doc.text(add_marker=True)
    start = full.index("Alpha")
    assert locate_in_document(doc, start, start + 5) == [(2, (0.1, 0.2, 0.9, 0.3))]


def test_blocks_round_trip_through_json(tmp_path):
    doc = Document(pages=[_block_page(page_number=1), Page(page_number=2)])
    assert Document.from_dict(doc.to_dict()) == doc
    path = tmp_path / "doc.json"
    doc.save_json(str(path))
    loaded = Document.load_json(str(path))
    assert loaded == doc and loaded.schema_version == SCHEMA_VERSION


def test_a_1_0_document_without_blocks_still_loads():
    data = {"schema_version": "1.0", "pages": [{
        "page_number": 1, "text": "hi", "width_px": 10, "height_px": 20,
        "words": [{"text": "hi", "bbox": [0, 0, 1, 1], "char_start": 0, "char_end": 2}]}]}
    doc = Document.from_dict(data)
    assert doc.schema_version == "1.0"
    assert doc.pages[0].blocks == [] and doc.pages[0].words[0].bbox == (0, 0, 1, 1)
    assert locate(doc.pages[0], 0, 2) == [(0, 0, 1, 1)]
