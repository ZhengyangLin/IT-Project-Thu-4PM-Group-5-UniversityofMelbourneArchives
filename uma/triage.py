from __future__ import annotations

import hashlib
from pathlib import Path
from statistics import median

import cv2
import numpy as np


# Image quality thresholds and supported image formats
RELIABLE = 20
HOPELESS = 8
IMG_EXT = {".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Common title block keywords
PROBE_ANCHORS = ["SCALE", "DATE", "DRAWN", "DRAWING", "CHECKED",
                 "ARCHITECT", "TITLE", "SHEET", "JOB"]


# Estimate the character height in the selected image region
def char_height(gray: np.ndarray, roi: tuple[float, float, float, float] =
                (0.60, 0.60, 1.0, 1.0)) -> tuple[float, int]:
    h, w = gray.shape
    sub = gray[int(h * roi[1]):int(h * roi[3]), int(w * roi[0]):int(w * roi[2])]
    if sub.size == 0:
        return 0.0, 0

    _, binary = cv2.threshold(sub, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary)

    hs = []
    for st in stats[1:]:
        x, y, w, h, area = st
        if not (3 <= h <= 120 and 2 <= w <= 120):
            continue
        if area < 0.15 * w * h:
            continue
        if w / max(h, 1) > 8 or h / max(w, 1) > 8:
            continue
        hs.append(h)

    if len(hs) < 10:
        return 0.0, len(hs)

    return float(np.percentile(hs, 90)), len(hs)


# Classify image quality based on character height
def verdict(ch: float) -> str:
    if ch >= RELIABLE:
        return "reliable"
    if ch >= HOPELESS:
        return "marginal"
    return "unrecoverable"


# Calculate the proportion of likely OCR noise
def noise_ratio(texts: list[str]) -> float:
    if not texts:
        return 1.0
    junk = sum(1 for t in texts
               if len(t.strip()) <= 3 and not any(c.isdigit() for c in t))
    return junk / len(texts)


# Assess the readability and basic properties of an image
def assess(gray: np.ndarray) -> dict:
    ch, n = char_height(gray)
    return {"char_height_p90_px": round(ch, 1),
            "components": n,
            "verdict": verdict(ch),
            "note": {"reliable": "Key fields should be legible",
                     "marginal": "Large text is legible, but small text "
                                 "(consultant lists and revision tables) probably is not",
                     "unrecoverable": "The image appears to be severely downsampled; "
                                      "request the source-quality original"}.get(verdict(ch), ""),
            "width": int(gray.shape[1]),
            "height": int(gray.shape[0])}


# Count common title block keywords in OCR text
def probe_anchors(texts: list[str]) -> int:
    up = " ".join(texts).upper()
    return sum(1 for k in PROBE_ANCHORS if k in up)


# Generate a SHA-256 hash for a file
def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


# Infer the drawing series from the file path
def infer_series(path: Path) -> str:
    parts = [p.upper() for p in path.parts]
    if any("BSM" in p for p in parts):
        return "BSM"
    if any("CABANA" in p for p in parts):
        return "YFA-Cabana"
    if any("STATE" in p for p in parts):
        return "YFA-StateGov"
    if any("YFA" in p for p in parts):
        return "YFA"
    return "unknown"


# Scan image files and build a manifest with file and quality information
def build_manifest(root: str | Path, sample: int | None = None) -> list[dict]:
    root = Path(root)
    rows = []
    files = [p for p in sorted(root.rglob("*")) if p.suffix.lower() in IMG_EXT]
    for p in files:
        img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_GRAYSCALE)
        row = {"node_id": p.parent.name, "path": str(p),
               "sha256": sha256(p)[:16], "series": infer_series(p),
               "filesize_kb": round(p.stat().st_size / 1024, 1)}
        row.update(assess(img) if img is not None else {"verdict": "unreadable_file"})
        rows.append(row)
    return rows


# Summarize image quality statistics for each drawing series
def summarize(rows: list[dict]) -> dict:
    by_series: dict[str, list[dict]] = {}
    for r in rows:
        by_series.setdefault(r.get("series", "unknown"), []).append(r)

    out = {"total": len(rows), "series": {}}
    for s, items in by_series.items():
        vs = [i.get("verdict") for i in items]
        chs = [i.get("char_height_p90_px", 0) for i in items]
        out["series"][s] = {
            "count": len(items),
            "char_height_p90_median": round(median(chs), 1) if chs else 0,
            "reliable": vs.count("reliable"),
            "marginal": vs.count("marginal"),
            "unrecoverable": vs.count("unrecoverable"),
            "unrecoverable_pct": round(vs.count("unrecoverable") / len(items) * 100, 1),
        }
    return out