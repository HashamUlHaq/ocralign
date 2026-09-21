"""Paddle backend renderer: layout blocks -> page text. Pure functions, no service needed."""
import re

import pytest

from ocralign.backends.paddle.render import (
    html_table_to_markdown,
    normalize_latex,
    render_page,
    render_segments,
)


def _block(label, content="", order=0):
    return {"label": label, "content": content, "bbox": [0, 0, 1, 1], "order": order}


# ---- blocks ---------------------------------------------------------------------------------------
def test_header_and_footer_are_kept_and_tagged():
    """PaddleOCR's own markdown drops these; on clinical documents they are the facility and
    patient banner."""
    text = render_page([
        _block("header", "Example Family Clinic\nTrusted Care", 0),
        _block("text", "Assessment: stable", 1),
        _block("footer", "CONFIDENTIAL — Protected Health Information", 2),
    ])
    assert text.startswith("[page header] Example Family Clinic\nTrusted Care")
    assert "[page footer] CONFIDENTIAL" in text


def test_page_number_block_is_plain_text():
    assert render_page([_block("number", "Page 2 of 5")]) == "Page 2 of 5"


def test_headings_images_and_reading_order():
    text = render_page([
        _block("paragraph_title", "ASSESSMENT\nAND PLAN", 1),
        _block("doc_title", "Progress Note", 0),
        _block("image", "", 2),
        _block("header_image", "", 3),
        _block("seal", "", 4),
        _block("text", "", 5),
    ])
    assert text.split("\n\n") == ["# Progress Note", "## ASSESSMENT AND PLAN", "[image]", "[seal]"]


def test_block_text_cannot_forge_a_page_boundary():
    """Text split on page markers would split this page in two."""
    text = render_page([_block("text", "see note -- Page 9 -- attached")])
    assert not re.search(r"--\s*Page\s+\d+\s*--", text)
    assert text == "see note Page 9 attached"


def test_segments_mark_blocks_that_add_no_text():
    segments = render_segments([
        _block("text", "body", 2),
        _block("header_image", "", 0),
        _block("text", "   ", 1),
        _block("text", "\\n", 3),        # a literal escape and nothing else
    ])
    assert [(label, seg) for _, label, seg in segments] == [
        ("header_image", None), ("text", None), ("text", "body"), ("text", None)]


def test_missing_label_is_text_and_labels_are_case_insensitive():
    segments = render_segments([{"content": "plain", "order": 0},
                                {"label": "HEADER", "content": "Banner", "order": 1}])
    assert [(label, seg) for _, label, seg in segments] == [
        ("text", "plain"), ("header", "[page header] Banner")]


def test_latex_is_normalised_inside_text_blocks():
    assert render_page([_block("text", "Alert and oriented $ \\times 3 $, non-focal")]) == \
        "Alert and oriented ×3, non-focal"


# ---- tables ---------------------------------------------------------------------------------------
def test_table_becomes_a_pipe_table_with_no_html():
    html = ("<table><tr><td>Test</td><td>Result</td><td>Reference</td><td>Flag</td></tr>"
            "<tr><td style='text-align:center'>Glucose, fasting</td><td>176</td>"
            "<td>70-99</td><td>H</td></tr><tr><td>HbA1c</td><td>7.3</td><td>&lt;5.7</td><td>H</td></tr></table>")
    md = render_page([_block("table", html)])
    assert md.splitlines() == [
        "| Test | Result | Reference | Flag |",
        "|---|---|---|---|",
        "| Glucose, fasting | 176 | 70-99 | H |",
        "| HbA1c | 7.3 | <5.7 | H |",
    ]
    assert "<t" not in md and "style=" not in md


def test_rowspan_is_repeated_but_a_full_width_row_is_said_once():
    md = html_table_to_markdown(
        "<table><tr><td rowspan=2>CMP</td><td>Na</td><td>139</td></tr><tr><td>K</td><td>4.2</td></tr>"
        "<tr><td colspan=3>H = high | L = low</td></tr></table>")
    assert md.splitlines()[2] == "| CMP | K | 4.2 |"
    assert md.splitlines()[3] == r"| H = high \| L = low |  |  |"


def test_malformed_table_html_does_not_lose_rows():
    md = html_table_to_markdown("<table><tr><td>a</td><td>b</td><tr><td>c</td><td>d</td></table>")
    assert md.splitlines() == ["| a | b |", "|---|---|", "| c | d |"]


def test_pathological_span_is_capped():
    md = html_table_to_markdown("<table><tr><td colspan=1000000>x</td></tr></table>")
    assert md.splitlines()[1] == "|" + "---|" * 64


@pytest.mark.parametrize("rowspan", [5, 1000000])
def test_a_rowspan_past_the_last_row_invents_no_rows(rowspan):
    """Model-emitted HTML overstates rowspan; an unclamped span repeated the cell into rows
    that do not exist."""
    md = html_table_to_markdown(
        f"<table><tr><td rowspan={rowspan}>X</td><td>a</td></tr><tr><td>b</td></tr></table>")
    assert md.splitlines() == ["| X | a |", "|---|---|", "| X | b |"]


def test_a_rowspan_within_the_table_still_fills_its_rows():
    md = html_table_to_markdown(
        "<table><tr><td rowspan=2>CMP</td><td>Na</td></tr><tr><td>K</td></tr>"
        "<tr><td>WBC</td><td>5.1</td></tr></table>")
    assert md.splitlines()[2:] == ["| CMP | K |", "| WBC | 5.1 |"]


def test_a_page_marker_inside_a_table_cell_cannot_forge_a_boundary():
    """The text path neutralises it; the table path must too, or a cell that prints a page
    number splits a page for any consumer that chunks on the marker."""
    md = render_page([_block("table", "<table><tr><td>see -- Page 7 --</td><td>x</td></tr></table>")])
    assert md.splitlines()[0] == "| see Page 7 | x |"
    assert not re.search(r"--\s*Page\s+\d+\s*--", md)


def test_unparseable_table_falls_back_to_its_text():
    assert render_page([_block("table", "no table markup here")]) == "no table markup here"


def test_literal_backslash_n_from_the_model_is_a_break_but_latex_is_not():
    """Seen live on a synthetic flowsheet: the model wrote 'NET FLUID BALANCE:\\n+1508 mL'."""
    md = html_table_to_markdown("<table><tr><td>NET FLUID BALANCE:\\n+1508 mL</td><td>x</td></tr></table>")
    assert md.splitlines()[0] == "| NET FLUID BALANCE: +1508 mL | x |"
    assert render_page([_block("text", "Total:\\nNET 5")]) == "Total:\nNET 5"
    assert normalize_latex("$ a \\neq b $") == "a≠b"


@pytest.mark.parametrize("raw, expected", [
    # the COMMON real case, missed by a "no lowercase letter follows" rule
    ("Total:\\npatient taking", "Total:\npatient taking"),
    ("dose 5\\nmg daily", "dose 5\nmg daily"),
    ("item\\ttaking", "item taking"),
    # ...while real LaTeX commands that begin with n / t still survive to be converted
    ("$ a \\ne 5 $", "a≠5"),
    ("$ \\nu $", "ν"),
    ("$ 3 \\times 4 $", "3×4"),
    ("$ \\to $", "→"),
])
def test_literal_escapes_before_words_vs_latex_commands(raw, expected):
    assert render_page([_block("text", raw)]) == expected


# ---- LaTeX ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("raw, expected", [
    ("Alert and oriented $ \\times 3 $", "Alert and oriented ×3"),
    ("eGFR $ \\geq 60 $ mL/min", "eGFR ≥60 mL/min"),
    ("HbA1c 7.3 $ \\% $", "HbA1c 7.3%"),
    ("BSA 1.73 m $ ^{2} $", "BSA 1.73 m²"),
    ("SpO $ _{2} $ 98%", "SpO₂ 98%"),
    ("Temp 37.2 $ ^{\\circ} $C", "Temp 37.2 °C"),
    ("BP $ \\frac{120}{80} $ mmHg", "BP 120/80 mmHg"),
    # a command followed by a LETTER: the space is what ends the control word
    ("K $ \\neq b $", "K ≠b"),
    ("$ \\alpha \\to \\beta $ blocker", "α→β blocker"),
])
def test_simple_latex_becomes_unicode(raw, expected):
    assert normalize_latex(raw) == expected


@pytest.mark.parametrize("raw", [
    "copay $5 and $10 deductible",
    "$ x \\mapsto y $",
    # A bare "_" (a form blank) used to start a conversion, so the text between two dollar signs
    # was treated as math — "$" deleted and spaces stripped.
    "Paid $20 _____ balance $15",
    "Copay $____ Deductible $____",
    "Fee $ 10 per visit_code $ 5",
    "$120 mmHg$",
])
def test_currency_blanks_and_unknown_latex_are_left_alone(raw):
    """Converting half an expression is worse than not converting it — and prose is never math."""
    assert normalize_latex(raw) == raw


@pytest.mark.parametrize("raw", [
    # The "keep the original" guard cannot see inside a wrapper's argument, so these used to
    # emit a bare "\alpha" into text that is supposed to carry no markup.
    r"$\text{\alpha}$",
    r"$\underline{\text{\beta}}$",
])
def test_a_command_hidden_in_a_wrapper_keeps_the_original_span(raw):
    assert normalize_latex(raw) == raw


@pytest.mark.parametrize("raw, expected", [
    # ...while a wrapper holding plain text still converts, including literal ^ and braces,
    # which are not math inside \text{}.
    (r"$\text{5 mg}$", "5 mg"),
    (r"$\text{m^2}$", "m^2"),
])
def test_a_wrapper_holding_plain_text_still_converts(raw, expected):
    assert normalize_latex(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("Consent $\\underline{\\text{Yes}}$ signed", "Consent Yes signed"),
    ("Done $\\boxtimes$ declined $\\square$ n/a $\\checkmark$", "Done ☒ declined ☐ n/a ✓"),
    ("Registered$^{\\circledR}$ mark", "Registered® mark"),
    ("TNF-$\\alpha$ and $\\kappa$ chains", "TNF-α and κ chains"),
])
def test_form_markup_and_checkboxes_keep_their_meaning(raw, expected):
    """On a form the checkbox state IS the answer; \\underline{\\text{..}} wraps the filled value."""
    assert normalize_latex(raw) == expected
