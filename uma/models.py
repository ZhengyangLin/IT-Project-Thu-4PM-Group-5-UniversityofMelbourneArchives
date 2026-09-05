from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

BBox = tuple[int, int, int, int]  # Bounding box coordinates


# Calculate IoU between two boxes
def iou(a: BBox, b: BBox) -> float:
    ox0, oy0 = max(a[0], b[0]), max(a[1], b[1])
    ox1, oy1 = min(a[2], b[2]), min(a[3], b[3])
    if ox1 <= ox0 or oy1 <= oy0:
        return 0.0
    inter = (ox1 - ox0) * (oy1 - oy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1)


# Calculate overlap ratio based on the smaller box
def io_min(a: BBox, b: BBox) -> float:
    ox0, oy0 = max(a[0], b[0]), max(a[1], b[1])
    ox1, oy1 = min(a[2], b[2]), min(a[3], b[3])
    if ox1 <= ox0 or oy1 <= oy0:
        return 0.0
    inter = (ox1 - ox0) * (oy1 - oy0)
    sa = (a[2] - a[0]) * (a[3] - a[1])
    sb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(min(sa, sb), 1)


# Merge multiple boxes
def union(boxes: list[BBox]) -> BBox:
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


# Calculate horizontal overlap
def h_overlap(a: BBox, b: BBox) -> float:
    ov = min(a[2], b[2]) - max(a[0], b[0])
    if ov <= 0:
        return 0.0
    return ov / max(min(a[2] - a[0], b[2] - b[0]), 1)


# OCR candidate result
@dataclass
class Candidate:
    text: str
    engine: str
    conf: float
    source_bbox: BBox | None = None


# OCR token
@dataclass
class Token:
    text: str
    bbox: BBox
    conf: float
    candidates: list[Candidate] = field(default_factory=list)
    agreement: float = 1.0
    tid: int = -1

    @property
    def x0(self) -> int: return self.bbox[0]
    @property
    def y0(self) -> int: return self.bbox[1]
    @property
    def x1(self) -> int: return self.bbox[2]
    @property
    def y1(self) -> int: return self.bbox[3]
    @property
    def w(self) -> int: return self.bbox[2] - self.bbox[0]
    @property
    def h(self) -> int: return self.bbox[3] - self.bbox[1]
    @property
    def cx(self) -> float: return (self.bbox[0] + self.bbox[2]) / 2
    @property
    def cy(self) -> float: return (self.bbox[1] + self.bbox[3]) / 2

    # Return alternative OCR results
    def alternatives(self) -> list[Candidate]:
        return [c for c in self.candidates if c.text.strip() != self.text.strip()]


# OCR text line
@dataclass
class Line:
    tokens: list[Token]
    row_no: int = -1

    @property
    def text(self) -> str:
        return " ".join(t.text for t in self.tokens)

    @property
    def bbox(self) -> BBox:
        return union([t.bbox for t in self.tokens])

    @property
    def x0(self) -> int: return self.bbox[0]
    @property
    def y0(self) -> int: return self.bbox[1]
    @property
    def x1(self) -> int: return self.bbox[2]
    @property
    def y1(self) -> int: return self.bbox[3]
    @property
    def h(self) -> int: return self.bbox[3] - self.bbox[1]
    @property
    def cy(self) -> float: return (self.bbox[1] + self.bbox[3]) / 2

    # Calculate distance between lines
    def gap_to(self, other: "Line", ky: float = 0.6) -> float:
        a, b = self.bbox, other.bbox
        dx = max(0, a[0] - b[2], b[0] - a[2])
        dy = max(0, a[1] - b[3], b[1] - a[3])
        return (dx ** 2 + (ky * dy) ** 2) ** 0.5


# Text block
@dataclass
class Block:
    bbox: BBox
    lines: list[Line] = field(default_factory=list)
    source: str = "xy_cut"          # xy_cut | caption_resplit
    is_title_block: bool = False

    @property
    def area(self) -> int:
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])


# Group of related lines
@dataclass
class Cluster:
    lines: list[Line]
    block_idx: int = -1
    score: float = 0.0

    @property
    def bbox(self) -> BBox:
        return union([l.bbox for l in self.lines])

    @property
    def text(self) -> str:
        return " ".join(l.text for l in self.lines)

    # Calculate text compactness
    def compactness(self) -> float:
        bb = self.bbox
        area = max((bb[2] - bb[0]) * (bb[3] - bb[1]), 1)
        ink = sum((l.x1 - l.x0) * l.h for l in self.lines)
        return min(ink / area, 1.0)


# Extracted metadata field
@dataclass
class FieldValue:
    name: str
    tokens: list[int] = field(default_factory=list)
    verbatim: Optional[str] = None
    corrected: Optional[str] = None
    correction_basis: Optional[str] = None
    source: str = ""
    reason: Optional[str] = None
    confidence: float = 0.0
    signals: dict = field(default_factory=dict)
    status: str = "needs_review"
    notes: list[str] = field(default_factory=list)


# Complete result for one drawing sheet
@dataclass
class SheetRecord:
    node_id: str
    path: str
    series: str = ""
    fields: dict[str, FieldValue] = field(default_factory=dict)
    title_bbox: Optional[BBox] = None
    localization_method: str = ""
    localization_conf: float = 0.0
    grid: str = ""
    index_view: str = ""
    token_map: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)