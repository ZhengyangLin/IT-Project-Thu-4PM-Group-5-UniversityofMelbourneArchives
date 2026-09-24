from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np
from rapidfuzz import fuzz

from .models import BBox, Token

log = logging.getLogger(__name__)

# Keyword groups used to identify likely title-block text.
LOCATOR_STRONG = ["YUNCKEN FREEMAN", "BATES SMART", "ARCHITECTS",
                  "DRAWING NO", "DRAWING NUMBER", "DWG NO", "DRG NO"]
LOCATOR_WEAK = ["SCALE", "DRAWN", "CHECKED", "DATE", "TRACED",
                "APPROVED", "JOB NO", "SHEET NO", "TITLE", "REVISION"]


# Check whether OCR text matches any locator keyword using exact or fuzzy matching.
def _hit(text: str, keywords: list[str], threshold: int) -> bool:

    for kw in keywords:
        if len(text) < max(len(kw) * 0.5, 4):
            continue
        if kw in text:
            return True
        if fuzz.ratio(kw, text) >= threshold:
            return True
        if len(text) >= len(kw) and fuzz.partial_ratio(kw, text) >= threshold:
            return True
    return False


# Store the detected title-block region together with its method and confidence.
@dataclass
class LocateResult:
    bbox: BBox
    method: str                 # anchor | frame | density | fallback
    confidence: float
    detail: dict = field(default_factory=dict)




# Extract long horizontal and vertical line structures from the drawing.
def line_masks(plain: np.ndarray, ink_percentile: float = 15.0,
               min_ratio: float = 0.10) -> tuple[np.ndarray, np.ndarray]:

    t = np.percentile(plain, ink_percentile)
    binary = (plain < t).astype(np.uint8)
    h, w = binary.shape
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(int(w * min_ratio), 25), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(int(h * min_ratio), 25)))
    horiz = cv2.morphologyEx(binary, cv2.MORPH_OPEN, hk)
    vert = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vk)
    return horiz, vert


# Convert line masks into representative horizontal and vertical coordinates.
def line_positions(horiz: np.ndarray, vert: np.ndarray,
                   min_cover: float = 0.25) -> tuple[list[int], list[int]]:
    h, w = horiz.shape
    hp = horiz.sum(1) / max(w, 1)
    vp = vert.sum(0) / max(h, 1)
    ys = _peaks(hp, min_cover)
    xs = _peaks(vp, min_cover)
    return ys, xs


# Merge nearby profile peaks so that thick lines are represented by one coordinate.
def _peaks(profile: np.ndarray, thresh: float, merge_gap: int = 6) -> list[int]:
    idx = np.where(profile >= thresh)[0]
    if len(idx) == 0:
        return []
    out, run = [], [idx[0]]
    for i in idx[1:]:
        if i - run[-1] <= merge_gap:
            run.append(i)
        else:
            out.append(int(np.mean(run)))
            run = [i]
    out.append(int(np.mean(run)))
    return out


# Adjust a candidate bounding box to nearby detected frame lines when possible.
def snap_to_frame(box: BBox, ys: list[int], xs: list[int],
                  page: tuple[int, int], tol: float = 0.06) -> tuple[BBox, bool]:
    W, H = page
    x0, y0, x1, y1 = box
    tw, th = W * tol, H * tol
    snapped = False

    cand = [x for x in xs if abs(x - x0) <= tw and x <= x0 + tw]
    if cand:
        x0 = min(cand, key=lambda v: abs(v - x0)); snapped = True
    cand = [x for x in xs if abs(x - x1) <= tw and x >= x1 - tw]
    if cand:
        x1 = min(cand, key=lambda v: abs(v - x1)); snapped = True
    cand = [y for y in ys if abs(y - y0) <= th and y <= y0 + th]
    if cand:
        y0 = min(cand, key=lambda v: abs(v - y0)); snapped = True
    cand = [y for y in ys if abs(y - y1) <= th and y >= y1 - th]
    if cand:
        y1 = min(cand, key=lambda v: abs(v - y1)); snapped = True

    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    if x1 - x0 < 20 or y1 - y0 < 20:
        return box, False
    return (int(x0), int(y0), int(x1), int(y1)), snapped




# L1: locate the title block from OCR keyword anchors, favouring the lower-right area.
def locate_by_anchor(tokens: list[Token], page: tuple[int, int],
                     ys: list[int], xs: list[int],
                     fuzz_threshold: int = 85,
                     min_hits: int = 2,
                     expand_ratio: float = 0.03) -> LocateResult | None:

    W, H = page
    strong, weak = [], []
    for t in tokens:
        up = t.text.upper().strip(" .:,;")

        if len(up) < 4 or not any(c.isalpha() for c in up):
            continue
        if _hit(up, LOCATOR_STRONG, fuzz_threshold):
            strong.append(t)
        elif _hit(up, LOCATOR_WEAK, fuzz_threshold):
            weak.append(t)

    hits = strong + weak
    if not hits or (len(strong) == 0 and len(weak) < min_hits):
        return None


    corner = np.array([W, H], dtype=float)
    hits.sort(key=lambda t: np.hypot(t.cx - corner[0], t.cy - corner[1]))
    keep = [hits[0]]
    for t in hits[1:]:
        near = min(max(abs(t.cx - k.cx) / W, abs(t.cy - k.cy) / H) for k in keep)
        if near <= 0.22:
            keep.append(t)

    x0 = min(t.x0 for t in keep); y0 = min(t.y0 for t in keep)
    x1 = max(t.x1 for t in keep); y1 = max(t.y1 for t in keep)
    ex, ey = W * expand_ratio, H * expand_ratio
    box = (int(max(x0 - ex, 0)), int(max(y0 - ey, 0)),
           int(min(x1 + ex, W)), int(min(y1 + ey, H)))
    box, snapped = snap_to_frame(box, ys, xs, page)

    n_strong, n_weak = len(strong), len(weak)
    conf = min(0.55 + 0.12 * n_strong + 0.05 * n_weak + (0.08 if snapped else 0), 0.97)
    return LocateResult(box, "anchor", round(conf, 3),
                        {"strong": n_strong, "weak": n_weak,
                         "used": len(keep), "snapped": snapped,
                         "words": [t.text for t in keep[:8]]})




# L2: locate the title block from table/frame structure and OCR density inside cells.
def locate_by_frame(plain: np.ndarray, tokens: list[Token],
                    horiz: np.ndarray, vert: np.ndarray,
                    area_range: tuple[float, float] = (0.01, 0.30)
                    ) -> LocateResult | None:

    H, W = plain.shape
    lines = cv2.dilate(cv2.bitwise_or(horiz, vert), np.ones((3, 3), np.uint8))
    cells = (1 - lines).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cells, connectivity=4)

    best, best_score, detail = None, -1.0, {}
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        ratio = area / (W * H)
        if not (area_range[0] <= ratio <= area_range[1]):
            continue
        if w < 40 or h < 40:
            continue
        inside = [t for t in tokens if x <= t.cx <= x + w and y <= t.cy <= y + h]
        if len(inside) < 3:
            continue
        pos = ((x + w / 2) / W) * 0.5 + ((y + h / 2) / H) * 0.5
        ink = sum(t.w * t.h for t in inside) / max(area, 1)
        dens = min(ink * 4, 1.0)
        shape = 1.0 if 0.02 <= ratio <= 0.25 else 0.4
        score = 0.45 * pos + 0.30 * dens + 0.25 * shape
        if score > best_score:
            best, best_score = (int(x), int(y), int(x + w), int(y + h)), score
            detail = {"cell_ratio": round(ratio, 4), "tokens": len(inside),
                      "pos": round(pos, 3), "density": round(dens, 3)}
    if best is None:
        return None
    best, merged_n = _merge_neighbour_cells(best, stats, n, tokens, (W, H))
    detail["merged_cells"] = merged_n
    return LocateResult(best, "frame", round(min(best_score, 0.92), 3), detail)


# Expand a selected frame cell by merging nearby text-containing cells.
def _merge_neighbour_cells(box: BBox, stats, n: int, tokens: list[Token],
                           page: tuple[int, int], gap_ratio: float = 0.02
                           ) -> tuple[BBox, int]:
    W, H = page
    x0, y0, x1, y1 = box
    gx, gy = W * gap_ratio, H * gap_ratio
    merged = 0
    for _ in range(6):
        grew = False
        for i in range(1, n):
            cx, cy, cw, ch, area = stats[i]
            if area > W * H * 0.30 or cw < 20 or ch < 20:
                continue
            bx0, by0, bx1, by1 = cx, cy, cx + cw, cy + ch
            if bx0 >= x0 and by0 >= y0 and bx1 <= x1 and by1 <= y1:
                continue
            inside = [t for t in tokens if bx0 <= t.cx <= bx1 and by0 <= t.cy <= by1]
            if len(inside) < 2:
                continue
            h_ov = min(bx1, x1) - max(bx0, x0)
            v_ov = min(by1, y1) - max(by0, y0)
            dx = max(0, bx0 - x1, x0 - bx1)
            dy = max(0, by0 - y1, y0 - by1)
            near = (v_ov > 0 and dx <= gx) or (h_ov > 0 and dy <= gy)
            if not near:
                continue
            x0, y0 = min(x0, bx0), min(y0, by0)
            x1, y1 = max(x1, bx1), max(y1, by1)
            merged += 1
            grew = True
        if not grew:
            break
        if (x1 - x0) * (y1 - y0) > W * H * 0.35:
            return box, 0
    return (int(x0), int(y0), int(x1), int(y1)), merged


# L3: locate a dense OCR region using a coarse spatial heat map.
def locate_by_density(tokens: list[Token], page: tuple[int, int],
                      grid: int = 48, keep_ratio: float = 0.45
                      ) -> LocateResult | None:
    W, H = page
    if len(tokens) < 10:
        return None
    heat = np.zeros((grid, grid), np.float32)
    for t in tokens:
        gx = min(int(t.cx / W * grid), grid - 1)
        gy = min(int(t.cy / H * grid), grid - 1)
        heat[gy, gx] += t.w * t.h
    heat = cv2.GaussianBlur(heat, (5, 5), 0)
    if heat.max() <= 0:
        return None
    quad = heat.copy()
    quad[:int(grid * 0.35), :] = 0
    quad[:, :int(grid * 0.35)] = 0
    if quad.max() <= 0:
        quad = heat
    gy, gx = np.unravel_index(int(np.argmax(quad)), quad.shape)
    thresh = quad[gy, gx] * keep_ratio
    y0 = y1 = gy
    x0 = x1 = gx
    changed = True
    while changed:
        changed = False
        for dy0, dy1, dx0, dx1 in [(-1, 0, 0, 0), (0, 1, 0, 0), (0, 0, -1, 0), (0, 0, 0, 1)]:
            ny0, ny1 = max(y0 + dy0, 0), min(y1 + dy1, grid - 1)
            nx0, nx1 = max(x0 + dx0, 0), min(x1 + dx1, grid - 1)
            if (ny0, ny1, nx0, nx1) == (y0, y1, x0, x1):
                continue
            if heat[ny0:ny1 + 1, nx0:nx1 + 1].mean() >= thresh:
                y0, y1, x0, x1 = ny0, ny1, nx0, nx1
                changed = True

    box = (int(x0 / grid * W), int(y0 / grid * H),
           int((x1 + 1) / grid * W), int((y1 + 1) / grid * H))
    inside = [t for t in tokens if box[0] <= t.cx <= box[2] and box[1] <= t.cy <= box[3]]
    if len(inside) < 5:
        return None
    conf = 0.45 + 0.15 * min(len(inside) / 20, 1.0)
    return LocateResult(box, "density", round(conf, 3),
                        {"tokens": len(inside), "peak": [int(gx), int(gy)]})


# Run the localization cascade: anchor first, then frame, then density, then fallback.
def locate(plain: np.ndarray, tokens: list[Token],
           ink_percentile: float = 15.0,
           fuzz_threshold: int = 85,
           line_min_ratio: float = 0.10) -> LocateResult:
    H, W = plain.shape
    page = (W, H)
    horiz, vert = line_masks(plain, ink_percentile, line_min_ratio)
    ys, xs = line_positions(horiz, vert)
    log.info("Detected frame lines: %d horizontal, %d vertical", len(ys), len(xs))

    r = locate_by_anchor(tokens, page, ys, xs, fuzz_threshold)
    if r is not None:
        log.info("L1 anchor localization matched: %d strong, %d weak; %s",
                 r.detail["strong"], r.detail["weak"], r.detail["words"][:5])
        return r
    log.info("L1 anchor localization did not match; trying L2 frame localization")

    r = locate_by_frame(plain, tokens, horiz, vert)
    if r is not None:
        log.info("L2 frame localization matched: %s", r.detail)
        return r
    log.info("L2 frame localization did not match; trying L3 density localization")

    r = locate_by_density(tokens, page)
    if r is not None:
        log.info("L3 density localization matched: %s", r.detail)
        return r

    # Final fallback: use the lower-right quadrant when all evidence-based methods fail.
    box = (int(W * 0.60), int(H * 0.55), W, H)
    inside = [t for t in tokens if box[0] <= t.cx <= box[2] and box[1] <= t.cy <= box[3]]
    log.warning("All three localization stages failed; falling back to the "
                "lower-right quadrant (%d text blocks). The image is marked "
                "as degraded and queued for manual review", len(inside))
    return LocateResult(box, "fallback", 0.15,
                        {"degraded": True, "tokens": len(inside),
                         "note": "Lower-right-quadrant prior; not confirmed by evidence"})