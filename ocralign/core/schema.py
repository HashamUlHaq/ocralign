"""
Canonical, engine-agnostic document schema.

Every OCR engine adapter converts its native output into these structures.
All downstream logic (layout rendering, locate(), overlays) operates only
on this schema and never on engine-specific formats.

Conventions:
- Bounding boxes are (x0, y0, x1, y1) as fractions of page width/height
  in the range 0-1, origin at the top-left of the page.
- char_start/char_end are offsets into the page's formatted `text` and are
  filled in by the layout renderer (None until a page has been rendered).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

SCHEMA_VERSION = "1.0"

BBox = Tuple[float, float, float, float]  # (x0, y0, x1, y1), normalized 0-1


@dataclass
class Word:
    text: str
    bbox: BBox
    confidence: Optional[float] = None  # engine confidence 0-100, None if N/A
    line_no: Optional[int] = None       # reading-order line index, set by layout
    char_start: Optional[int] = None    # offset into Page.text, set by layout
    char_end: Optional[int] = None

    @property
    def x0(self) -> float: return self.bbox[0]

    @property
    def y0(self) -> float: return self.bbox[1]

    @property
    def x1(self) -> float: return self.bbox[2]

    @property
    def y1(self) -> float: return self.bbox[3]

    @property
    def width(self) -> float: return max(1e-9, self.x1 - self.x0)

    @property
    def height(self) -> float: return max(1e-9, self.y1 - self.y0)

    @property
    def y_center(self) -> float: return (self.y0 + self.y1) / 2.0


@dataclass
class Page:
    page_number: int                    # 1-based
    words: List[Word] = field(default_factory=list)
    text: str = ""                      # formatted layout text, set by renderer
    width_px: Optional[int] = None      # source render size, informational
    height_px: Optional[int] = None


@dataclass
class Document:
    pages: List[Page] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def text(self, add_marker: bool = True) -> str:
        """Concatenated formatted text of all pages."""
        if add_marker:
            parts = [f"-- Page {p.page_number} --\n{p.text}\n\n" for p in self.pages]
        else:
            parts = [p.text for p in self.pages]
        return "".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        pages = []
        for p in data.get("pages", []):
            words = [
                Word(
                    text=w["text"],
                    bbox=tuple(w["bbox"]),
                    confidence=w.get("confidence"),
                    line_no=w.get("line_no"),
                    char_start=w.get("char_start"),
                    char_end=w.get("char_end"),
                )
                for w in p.get("words", [])
            ]
            pages.append(
                Page(
                    page_number=p["page_number"],
                    words=words,
                    text=p.get("text", ""),
                    width_px=p.get("width_px"),
                    height_px=p.get("height_px"),
                )
            )
        return cls(pages=pages, schema_version=data.get("schema_version", SCHEMA_VERSION))

    def save_json(self, path: str, indent: Optional[int] = None) -> None:
        import json

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=indent)

    @classmethod
    def load_json(cls, path: str) -> "Document":
        import json

        with open(path) as f:
            return cls.from_dict(json.load(f))
