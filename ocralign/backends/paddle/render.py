"""
Paddle backend renderer: one page's layout blocks -> the page text an LLM reads.

What the service returns is per-page LAYOUT BLOCKS (label, content, bbox, reading order), not
text. Turning those into page text is done here, and the rules are:

  * Never drop text. Every block with content is emitted, in the layout pipeline's reading order.
    PaddleOCR's own markdown drops `header`/`footer`/`number` blocks by default; on clinical
    documents those are the facility, address, signature line and patient banner, so they are kept.
  * Say what page furniture is. Headers and footers are tagged `[page header]` / `[page footer]`
    so a reader can tell a repeated banner from body content instead of guessing from position.
    The tags contain no digits, so a downstream "Page N" stripping rule can't touch them.
  * Keep structure in plain markdown: `#`/`##`/`###` headings, tables as pipe tables with
    row/col spans expanded so every row stands alone. Pipe tables carry the row structure of the
    HTML the model emits at a fraction of the tokens, and without the `<td style=...>` noise.
  * Normalise the VL model's LaTeX for symbols clinical text writes in Unicode: it reads
    "oriented x3" as `$ \\times 3 $`, which breaks any downstream regex looking for `×3`.
  * Block text can never contain a `-- Page N --` marker, so a quoted marker can't forge a page
    boundary in text that is later split on markers.

Pure functions over the service's JSON: no network, no paddle import, so stored payloads can be
re-rendered after a renderer fix without re-OCR'ing anything.
"""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from typing import Dict, List, Optional, Tuple

# Blocks of one page are joined with a blank line. processor.py computes each block's char offsets
# against this same separator, so page.text and the offsets can't disagree.
BLOCK_SEPARATOR = "\n\n"


# ---- block labels -------------------------------------------------------------------------------
_HEADINGS = {"doc_title": "# ", "paragraph_title": "## ", "figure_title": "### ",
             "chart_title": "### ", "table_title": "### "}
_TAGGED = {"header": "[page header] ", "footer": "[page footer] ", "aside_text": "[margin text] "}
# Visual regions with no recognised text. A placeholder tells the model something was there (a
# signature stamp, a wound photo, an ECG strip); logos in the banner carry nothing and are skipped.
_PLACEHOLDERS = {"image": "[image]", "chart": "[chart]", "seal": "[seal]"}
_SKIP = {"header_image", "footer_image"}

# A block's own text must never be able to forge a page boundary in text that is split on markers.
_MARKER_IN_CONTENT = re.compile(r"--\s*(Page\s+\d+)\s*--", re.IGNORECASE)

# Guard against pathological spans in model-emitted HTML (a colspan of 10^6 would allocate a grid).
_MAX_SPAN = 64


# ---- LaTeX --------------------------------------------------------------------------------------
_LATEX_COMMANDS = {
    "times": "×", "cdot": "·", "div": "÷", "pm": "±", "mp": "∓",
    "geq": "≥", "ge": "≥", "geqslant": "≥", "leq": "≤", "le": "≤", "leqslant": "≤",
    "neq": "≠", "ne": "≠", "approx": "≈", "sim": "~", "simeq": "≃", "equiv": "≡",
    "rightarrow": "→", "to": "→", "leftarrow": "←", "Rightarrow": "⇒", "uparrow": "↑",
    "downarrow": "↓", "degree": "°", "circ": "°", "infty": "∞", "partial": "∂",
    "mu": "µ", "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "Delta": "Δ",
    "kappa": "κ", "lambda": "λ", "sigma": "σ", "omega": "ω", "pi": "π", "nu": "ν",
    "epsilon": "ε", "eta": "η", "theta": "θ", "rho": "ρ", "tau": "τ", "phi": "φ", "chi": "χ",
    "psi": "ψ", "Sigma": "Σ", "Omega": "Ω",
    # form furniture: checkboxes and marks carry the ANSWER on a form ("Screening done ☒")
    "checkmark": "✓", "square": "☐", "Box": "☐", "boxtimes": "☒", "blacksquare": "■",
    "bullet": "•", "circledR": "®", "copyright": "©", "dagger": "†", "ddagger": "‡",
    "cdots": "…", "ldots": "…", "dots": "…", "prime": "′", "therefore": "∴",
    "quad": "\x01", "qquad": "\x01",
    "%": "%", "#": "#", "&": "&", "_": "_", "{": "{", "}": "}", "$": "$",
}
# Wrappers whose argument is the content. Unwrapped innermost-first, so the nesting the VL model
# emits on form fields ("\underline{\text{Yes}}") resolves instead of making the span unconvertible.
_UNWRAP = re.compile(r"\\(?:text|textrm|textbf|textit|mathrm|mathbf|mathit|mathsf|mathtt|"
                     r"operatorname|underline|overline|boxed|emph|bm)\s*\{([^{}]*)\}")
_SUPERSCRIPT = str.maketrans("0123456789+-=()", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾")
_SUBSCRIPT = str.maketrans("0123456789+-=()", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎")
# $$..$$ before $..$, so a display span isn't read as two empty inline ones. The optional leading
# space is captured so a result that must attach to the previous token ("m²", "SpO₂", "7.3%") can
# drop the space the VL model puts before every math span.
_MATH_SPAN = re.compile(r"( ?)(?:\$\$(.+?)\$\$|\$(.+?)\$|\\\((.+?)\\\)|\\\[(.+?)\\\])", re.DOTALL)
_ATTACHES_LEFT = set("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎%")


def _convert_math(expr: str) -> Optional[str]:
    """Plain-text form of a simple LaTeX expression, or None if it isn't simple enough.

    None means "leave the original span alone": converting half an expression is worse than not
    converting it, because the result looks like clean text but isn't.
    """
    # Only something that is unmistakably LaTeX starts a conversion. A bare "_" is NOT enough: forms
    # are full of blanks, and with it "Paid $20 _____ balance $15" was read as one math span — both
    # dollar signs deleted and the words squashed to "Paid 20_____balance15".
    if not re.search(r"\\[A-Za-z%#&_{}$]|\^|_\{|_\d", expr):
        return None                      # "$5 and $10": currency, not math — never rewrite it
    s = expr
    # \text{mg / dL} keeps its own spacing; everything else in math mode ignores whitespace.
    protected: List[str] = []

    def _protect(m):
        protected.append(m.group(1))
        return f"\x00{len(protected) - 1}\x00"

    while True:                          # innermost first: \underline{\text{Yes}} takes two passes
        unwrapped = _UNWRAP.sub(_protect, s)
        if unwrapped == s:
            break
        s = unwrapped
    # Prose between two "$" is currency or text, never math: a run of 3+ letters that isn't a
    # command name. Rewriting it would strip its spaces and delete the dollar signs.
    if re.search(r"(?<![\\A-Za-z])[A-Za-z]{3,}", s):
        return None
    s = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"\1/\2", s)
    s = re.sub(r"\\[,;:]|\\ ", "\x01", s)          # explicit math spaces survive as real spaces
    s = s.replace(r"\!", "")
    # Commands BEFORE stripping whitespace. A control word ends at the first non-letter, so in
    # "\neq b" the space is what ends `\neq`; strip it first and the name becomes `\neqb`.
    s = re.sub(r"\\([A-Za-z]+|[%#&_{}$])",
               lambda m: _LATEX_COMMANDS.get(m.group(1), m.group(0)), s)
    s = re.sub(r"\s+", "", s)                       # math-mode whitespace is insignificant
    # ^{\circ}, ^{\circledR}, ^{\dagger}: the command is already its symbol, so drop the ^{}.
    s = re.sub(r"\^\{?([°®©†‡*′✓])\}?", r"\1", s)
    s = re.sub(r"\^\{([0-9+\-=()]+)\}", lambda m: m.group(1).translate(_SUPERSCRIPT), s)
    s = re.sub(r"\^([0-9])", lambda m: m.group(1).translate(_SUPERSCRIPT), s)
    s = re.sub(r"_\{([0-9+\-=()]+)\}", lambda m: m.group(1).translate(_SUBSCRIPT), s)
    s = re.sub(r"_([0-9])", lambda m: m.group(1).translate(_SUBSCRIPT), s)
    if re.search(r"\\[A-Za-z]", s) or "^" in s or "{" in s or "}" in s:
        return None                      # something we don't understand survived: keep original
    while "\x00" in s:                   # placeholders can nest (a wrapper inside a wrapper)
        s = re.sub(r"\x00(\d+)\x00", lambda m: protected[int(m.group(1))], s)
    # The check above cannot see inside the placeholders, so a command hidden in a wrapper
    # argument ("\text{\alpha}") would be restored raw into text that is supposed to be clean.
    # Only commands are re-checked: "^" and braces are literal inside \text{} and must survive.
    if re.search(r"\\[A-Za-z]", s):
        return None
    return " ".join(s.replace("\x01", " ").split())


def normalize_latex(text: str) -> str:
    """Rewrite simple inline/display LaTeX to Unicode text; leave anything else untouched."""
    def repl(m):
        lead = m.group(1)
        expr = next(g for g in m.groups()[1:] if g is not None)
        out = _convert_math(expr)
        if out is None:
            return m.group(0)
        if out and out[0] in _ATTACHES_LEFT:
            lead = ""
        return lead + out
    return _MATH_SPAN.sub(repl, text)


# ---- tables -------------------------------------------------------------------------------------
class _TableParser(HTMLParser):
    """Collects rows of (text, colspan, rowspan) from the block's table markup.

    Every <tr> in the block is collected, so a block the model emitted as several <table>s
    becomes one row sequence. That is deliberate: the alternative is dropping all but the
    first table, and a lost lab row is worse than two tables read as one.
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows: List[List[Tuple[str, int, int]]] = []
        self._row: Optional[list] = None
        self._cell: Optional[List[str]] = None
        self._span = (1, 1)

    @staticmethod
    def _int(v, default=1):
        try:
            return max(1, min(_MAX_SPAN, int(str(v).strip())))
        except (TypeError, ValueError):
            return default

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag == "tr":
            # Model-emitted HTML isn't guaranteed well-formed: a <tr> with no </tr> must not
            # silently discard the row before it.
            if self._cell is not None and self._row is not None:
                self._row.append(("".join(self._cell), *self._span))
                self._cell = None
            if self._row:
                self.rows.append(self._row)
            self._row = []
        elif tag in ("td", "th"):
            a = dict(attrs)
            self._cell = []
            self._span = (self._int(a.get("colspan")), self._int(a.get("rowspan")))
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in ("td", "th") and self._cell is not None:
            if self._row is None:            # a cell outside any <tr>: give it a row
                self._row = []
            self._row.append(("".join(self._cell), *self._span))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)

    def close(self):
        super().close()
        if self._cell is not None and self._row is not None:
            self._row.append(("".join(self._cell), *self._span))
        if self._row:
            self.rows.append(self._row)


# The VL model sometimes writes a line break as the two characters backslash + n ("BALANCE:\n+1508
# mL", and far more often before a word: "\npatient", "\nmg"). Treat it as an escape UNLESS the
# letters after it spell a LaTeX command that starts with n / t, so \neq, \nu, \times, \to,
# \text survive. Keying on "no lowercase letter follows" instead missed nearly every real case.
_N_COMMANDS = ("e", "eq", "eg", "u", "ot", "otin", "abla", "i", "leq", "geq", "ewline", "atural",
               "exists", "earrow", "warrow", "mid", "parallel", "subseteq", "supseteq")
_T_COMMANDS = ("imes", "o", "ext", "extrm", "extbf", "extit", "heta", "au", "ilde", "op",
               "riangle", "frac", "herefore", "extstyle")
_LITERAL_ESCAPE = re.compile(
    r"\\n(?!(?:%s)(?![A-Za-z]))|\\t(?!(?:%s)(?![A-Za-z]))"
    % ("|".join(sorted(_N_COMMANDS, key=len, reverse=True)),
       "|".join(sorted(_T_COMMANDS, key=len, reverse=True))))


def _cell_text(s: str) -> str:
    s = normalize_latex(" ".join(_LITERAL_ESCAPE.sub(" ", s).split()))
    # Same neutralisation as _clean: a cell that prints "-- Page 7 --" must not be able to forge
    # a page boundary for a consumer that splits the document text on that pattern.
    s = _MARKER_IN_CONTENT.sub(r"\1", s)
    return s.replace("|", r"\|")


def html_table_to_markdown(table_html: str) -> Optional[str]:
    """HTML table -> markdown pipe table, spans expanded. None if there is no usable table."""
    p = _TableParser()
    try:
        p.feed(table_html)
        p.close()
    except Exception:
        return None
    if not p.rows:
        return None

    # (row, col) -> (value, source cell id). The id lets a full-width span be told apart from
    # neighbouring cells that merely happen to hold the same text.
    grid: Dict[Tuple[int, int], Tuple[str, int]] = {}
    source = 0
    for r, row in enumerate(p.rows):
        c = 0
        for text, colspan, rowspan in row:
            while (r, c) in grid:            # skip cells already claimed by a rowspan above
                c += 1
            value = _cell_text(text)
            # A rowspan may not reach past the last row that actually exists. Model-emitted HTML
            # overstates it, and an unclamped span INVENTS rows: rowspan=5 on a 2-row table used
            # to emit three further rows repeating this cell's text.
            rowspan = max(1, min(rowspan, len(p.rows) - r))
            # A spanned value is repeated into every cell it covers, so each row still reads
            # correctly on its own ("| CMP | K | 4.2 |") after chunking or retrieval.
            for dr in range(rowspan):
                for dc in range(colspan):
                    grid[(r + dr, c + dc)] = (value, source)
            source += 1
            c += colspan
    if not grid:
        return None
    n_rows = max(r for r, _ in grid) + 1
    n_cols = max(c for _, c in grid) + 1
    rows = []
    for r in range(n_rows):
        cells = [grid.get((r, c), ("", -1)) for c in range(n_cols)]
        if n_cols > 1 and cells[0][1] >= 0 and all(s == cells[0][1] for _, s in cells):
            # One cell spans the whole row (a legend, a section band). Repeating it in every
            # column adds tokens and no information: say it once.
            rows.append([cells[0][0]] + [""] * (n_cols - 1))
        else:
            rows.append([v for v, _ in cells])
    rows = [row for row in rows if any(cell.strip() for cell in row)]
    if not rows:
        return None
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * n_cols]
    lines += ["| " + " | ".join(row) + " |" for row in rows[1:]]
    return "\n".join(lines)


def _strip_tags(s: str) -> str:
    return re.sub(r"<[^>]+>", " ", s)


# ---- rendering ----------------------------------------------------------------------------------
def _clean(text: str) -> str:
    text = html.unescape(text)
    text = _LITERAL_ESCAPE.sub(lambda m: "\n" if m.group(0) == "\\n" else " ", text)
    text = normalize_latex(text)
    text = _MARKER_IN_CONTENT.sub(r"\1", text)
    text = "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _one_line(text: str) -> str:
    return " ".join(_clean(text).split())


def _render_block(label: str, content: str) -> Optional[str]:
    """One block's text, or None when the block adds none (a logo, an empty region)."""
    if label in _SKIP:
        return None
    if label in _PLACEHOLDERS:
        return _clean(content) if content else _PLACEHOLDERS[label]
    if not content:
        return None
    if label == "table":
        md = html_table_to_markdown(content)
        return md if md is not None else _clean(_strip_tags(content))
    if label in _HEADINGS:
        return _HEADINGS[label] + _one_line(content)
    if label in _TAGGED:
        # Tag once and keep the block's own line breaks (name / tagline / address).
        return _TAGGED[label] + _clean(content)
    return _clean(content)


def render_segments(blocks: Optional[List[dict]]) -> List[Tuple[dict, str, Optional[str]]]:
    """
    A page's blocks in reading order as (block, label, segment).

    segment is the block's text in the page, or None when the block contributes nothing. The page
    text is the non-None segments joined with BLOCK_SEPARATOR — render_page() and the processor's
    char offsets are both built from this one list.
    """
    out: List[Tuple[dict, str, Optional[str]]] = []
    for b in sorted(blocks or [], key=lambda b: b.get("order", 0)):
        label = (b.get("label") or "text").lower()
        segment = _render_block(label, (b.get("content") or "").strip())
        out.append((b, label, segment if segment is not None and segment.strip() else None))
    return out


def render_page(blocks: Optional[List[dict]]) -> str:
    """One page's layout blocks -> the page text."""
    return BLOCK_SEPARATOR.join(s for _, _, s in render_segments(blocks) if s is not None)
