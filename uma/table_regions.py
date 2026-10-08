from __future__ import annotations

from dataclasses import dataclass, field
import math
import re

import cv2
import numpy as np

from uma.models import BBox, Token
from uma.patterns import is_drawing_number

Line = tuple[float, float, float, float]
LABELS = {
    "drawn": re.compile(r"^DRAWN\b"),
    "checked": re.compile(r"^(?:CHECK(?:ED)?|CHKD)\b"),
    "scale": re.compile(r"^SCALES?\b"),
    "date": re.compile(r"^(?:DATE|D4TE|DA7E|DAT3|0ATE)\b"),
    "number": re.compile(r"^(?:DRG|DWG|DRAWING)\W*(?:NO\b|NUMBER\b)"),
    "approved": re.compile(r"^APPROVED\b"),
}
OPERATORS = {"drawn", "checked", "scale", "number", "approved"}


@dataclass
class TableProposal:
    bbox: BBox | None
    reliable: bool
    method: str = "local_frame"
    detail: dict = field(default_factory=dict)


def label_groups(tokens: list[Token]) -> set[str]:
    result = set()
    for t in tokens:
        text = t.text.upper().strip(" .:,;")
        result.update(k for k, pattern in LABELS.items() if pattern.search(text))
    return result


def _inside(tokens: list[Token], box) -> list[Token]:
    return [t for t in tokens if box[0] <= t.cx <= box[2]
            and box[1] <= t.cy <= box[3]]


def _union(boxes) -> tuple:
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def metadata_core(tokens, page):
    """Use a compact group of working labels; drawing references cannot seed it."""
    w, h = page
    operators = {'drawn', 'checked', 'scale', 'approved'}
    seeds = [t for t in tokens if t.cy >= h * .70 and label_groups([t]) & operators]
    choices = []
    for seed in seeds:
        band = [t for t in seeds if abs(t.cx - seed.cx) <= w * .09
                and abs(t.cy - seed.cy) <= h * .07]
        gs = label_groups(band)
        if len(gs) >= 2:
            choices.append((len(gs), sum(t.cy for t in band)/len(band), band))
    return _union([t.bbox for t in max(choices, key=lambda c:c[:2])[2]]) if choices else None


def core_column(core, vs, page):
    if core is None:
        return None
    w, h = page
    choices = []
    for a in vs:
        left = min(a[0], a[2])
        if not core[0] - w*.08 <= left <= core[0] + w*.012:
            continue
        for b in vs:
            right = max(b[0], b[2])
            top, bottom = max(a[1], b[1]), min(a[3], b[3])
            if (core[2] - w*.012 <= right <= core[2] + w*.08
                    and w*.05 <= right-left <= w*.22
                    and bottom-top >= h*.16
                    and top <= core[1] + h*.01 and bottom >= core[3] - h*.01):
                choices.append((right-left, -(bottom-top), (left,top,right,bottom)))
    if not choices:
        return None
    narrowest = min(c[0] for c in choices)
    compatible = [c for c in choices if c[0] <= narrowest * 1.12]
    return min(compatible, key=lambda c:(c[1],c[0]))[2]


def column_members(tokens, column):
    x0,y0,x1,y1 = column
    return [t for t in tokens if y0 <= t.cy <= y1
            and max(0,min(t.x1,x1)-max(t.x0,x0)) >= .80*max(t.w,1)]


def header_text(text):
    compact = re.sub(r'[^A-Z]', '', text.upper())
    return (bool(re.match(r'^TITLE\b', text.upper().strip()))
            or 'ARCHITECTS' in compact or 'PTYLTD' in compact)


def grow_text_up(box, tokens, page):
    w,h = page
    inside = _inside(tokens, box)
    if any(header_text(t.text) and t.cy <= box[1]+.4*(box[3]-box[1]) for t in inside):
        return tuple(box)
    heights = [t.h for t in inside if t.h > 0 and t.h < h*.04]
    char_h = float(np.median(heights)) if heights else h*.008
    limit = box[1] - min(h*.10, max(h*.045, 8*char_h))
    eligible = [t for t in tokens if limit <= t.y0 < box[1]
                and box[0]-w*.012 <= t.cx <= box[2]+w*.012
                and max(0,min(t.x1,box[2])-max(t.x0,box[0])) >= .8*max(t.w,1)
                and len(re.sub('[^A-Za-z]','',t.text)) >= 3]
    frontier = box[1]
    selected = []
    for _ in range(12):
        band = [t for t in eligible if t.y1 >= frontier - 2.5*char_h and t.y0 < frontier]
        if not band:
            break
        selected.extend(band)
        new_top = min(t.y0 for t in band)
        headings = [t for t in band if header_text(t.text)]
        if headings:
            stop = min(t.y0 for t in headings) - char_h
            selected.extend(t for t in eligible if t.y0 >= stop)
            break
        if new_top >= frontier:
            break
        frontier = new_top
    if not any(header_text(t.text) for t in selected):
        return tuple(box)
    return _union([box] + [t.bbox for t in selected])


def content_coverage(box, members):
    if not members:
        return 1.0
    return sum(box[0] <= t.cx <= box[2] and box[1] <= t.cy <= box[3]
               for t in members) / len(members)


def merge_segments(lines: list[Line], *, horizontal: bool,
                   max_gap: float, offset_tol: float = 8,
                   angle_tol: float = 2) -> list[Line]:
    normalized = []
    for x0, y0, x1, y1 in lines:
        u0, v0, u1, v1 = (x0, y0, x1, y1) if horizontal else (y0, x0, y1, x1)
        if u1 < u0:
            u0, v0, u1, v1 = u1, v1, u0, v0
        if u1 - u0 < 1:
            continue
        slope = (v1 - v0) / (u1 - u0)
        normalized.append((u0, v0, u1, v1, slope))
    normalized.sort(key=lambda a: a[2] - a[0], reverse=True)
    groups = []
    for line in normalized:
        u0, v0, u1, v1, slope = line
        for group in groups:
            ref = group[0]
            if abs(math.degrees(math.atan(slope) - math.atan(ref[4]))) > angle_tol:
                continue
            mid = (u0 + u1) / 2
            predicted = ref[1] + ref[4] * (mid - ref[0])
            if abs((v0 + v1) / 2 - predicted) > offset_tol:
                continue
            gap = min(max(0, u0 - other[2], other[0] - u1) for other in group)
            if gap <= max_gap:
                group.append(line)
                break
        else:
            groups.append([line])
    merged = []
    for group in groups:
        first = min(group, key=lambda a: a[0])
        last = max(group, key=lambda a: a[2])
        line = (first[0], first[1], last[2], last[3])
        merged.append(line if horizontal else (line[1], line[0], line[3], line[2]))
    return merged


def detect_segments(gray: np.ndarray) -> tuple[list[Line], list[Line], dict]:
    h, w = gray.shape
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 41, 12)
    binary[:int(h * .45)] = 0
    raw = cv2.HoughLinesP(binary, 1, np.pi / 360, 50,
                         minLineLength=max(40, int(w * .025)), maxLineGap=12)
    hs, vs = [], []
    if raw is not None:
        for values in raw[:, 0]:
            x0, y0, x1, y1 = map(float, values)
            angle = math.degrees(math.atan2(abs(y1 - y0), abs(x1 - x0)))
            if angle <= 3:
                hs.append((x0, y0, x1, y1))
            elif angle >= 87:
                vs.append((x0, y0, x1, y1))
    straight = []
    for horizontal in (True, False):
        size = max(35, round(w * .025 if horizontal else h * .055))
        kernel = (1, size) if horizontal else (size, 1)
        mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones(kernel, np.uint8))
        closing = (1, 11) if horizontal else (11, 1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones(closing, np.uint8))
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        lines = []
        for x, y, cw, ch, _ in stats[1:]:
            if horizontal and cw >= w * .045 and ch <= 18:
                lines.append((float(x), y + ch / 2, float(x + cw), y + ch / 2))
            elif not horizontal and ch >= h * .10 and cw <= 18:
                lines.append((x + cw / 2, float(y), x + cw / 2, float(y + ch)))
        straight.append(lines)
    key = lambda a: math.hypot(a[2] - a[0], a[3] - a[1])
    hs = sorted(hs, key=key, reverse=True)[:240]
    vs = sorted(vs, key=key, reverse=True)[:240]
    merged_h = merge_segments(hs, horizontal=True, max_gap=w * .05,
                             offset_tol=max(4, h * .006))
    merged_v = merge_segments(vs, horizontal=False, max_gap=h * .035,
                             offset_tol=max(4, w * .003))
    return merged_h + straight[0], merged_v + straight[1], {"horizontal_raw": len(hs), "vertical_raw": len(vs),
                               "horizontal_merged": len(merged_h),
                               "vertical_merged": len(merged_v)}


def text_candidates(tokens: list[Token], page: tuple[int, int]) -> list[tuple]:
    w, h = page
    labels = [t for t in tokens if t.cy >= h * .72 and label_groups([t])]
    candidates = []
    for seed in labels:
        band = [t for t in labels if abs(t.cy - seed.cy) <= h * .11
                and abs(t.cx - seed.cx) <= w * .40]
        gs = label_groups(band)
        if len(gs) < 3 or len(gs & OPERATORS) < 2:
            continue
        box = _union([t.bbox for t in band])
        nearby = [t for t in tokens if box[0] - w * .10 <= t.cx <= box[2] + w * .16
                  and box[1] - h * .018 <= t.cy <= box[3] + h * .025]
        if nearby:
            box = _union([box] + [t.bbox for t in nearby])
        if box[2] - box[0] >= w * .22 and box[3] - box[1] <= h * .22:
            candidates.append((*box, "text_layout"))
    return list(dict.fromkeys(candidates))


def _frame_candidates(hs, vs, w, h):
    candidates = []
    for a in vs:
        for b in vs:
            left, right = min(a[0], a[2]), max(b[0], b[2])
            top, bottom = max(a[1], b[1]), min(a[3], b[3])
            if (w * .055 <= right - left <= w * .30
                    and h * .13 <= bottom - top <= h * .65
                    and left >= w * .55 and bottom >= h * .80):
                candidates.append((left, top, right, bottom, "local_frame"))
    for a in hs:
        for b in hs:
            left, right = max(a[0], b[0]), min(a[2], b[2])
            top, bottom = min(a[1], a[3]), max(b[1], b[3])
            if (h * .035 <= bottom - top <= h * .24
                    and w * .23 <= right - left <= w * .72
                    and right >= w * .85 and top >= h * .72):
                candidates.append((left, top, right, bottom, "local_frame"))
    return candidates


def _support(box, hs, vs, w, h):
    """Local support: a rule must overlap the candidate, not just share x/y."""
    x0, y0, x1, y1 = box
    supports = {}
    for name, edge, lines, horizontal, tol in (
            ("top", y0, hs, True, h * .025), ("bottom", y1, hs, True, h * .025),
            ("left", x0, vs, False, w * .02), ("right", x1, vs, False, w * .02)):
        matches = []
        for a in lines:
            p0, p1 = (a[0], a[2]) if horizontal else (a[1], a[3])
            lo, hi = (x0, x1) if horizontal else (y0, y1)
            overlap = max(0, min(p1, hi) - max(p0, lo)) / max(hi - lo, 1)
            position = (a[1] + a[3]) / 2 if horizontal else (a[0] + a[2]) / 2
            if overlap >= .60 and abs(position - edge) <= tol:
                matches.append((abs(position - edge), position))
        if matches:
            supports[name] = min(matches)[1]
    return supports


def complete_horizontal_frame(box, hs, vs, page):
    w, h = page
    x0, y0, x1, y1 = box
    if x1 - x0 < 1.5 * (y1 - y0):
        return box
    completed = list(box)
    for a in hs:
        length = a[2] - a[0]
        overlap = max(0, min(a[2], x1) - max(a[0], x0))
        near = min(abs((a[1] + a[3]) / 2 - edge) for edge in (y0, y1))
        if (w * .22 <= length <= w * .72 and overlap >= .65 * (x1 - x0)
                and near <= h * .025 and length <= 1.5 * (x1 - x0)):
            connected = []
            for side, endpoint in ((0, (a[0], a[1])), (2, (a[2], a[3]))):
                ex, ey = endpoint
                for v in vs:
                    overlap_v = max(0, min(v[3], y1) - max(v[1], y0))
                    if overlap_v < .60 * (y1 - y0) or v[3] <= v[1]:
                        continue
                    vx = v[0] + (v[2] - v[0]) * (ey - v[1]) / (v[3] - v[1])
                    if (v[1] - h * .012 <= ey <= v[3] + h * .012 and abs(vx - ex) <= w * .012):
                        at_top = v[0] + (v[2] - v[0]) * (y0 - v[1]) / (v[3] - v[1])
                        at_bottom = v[0] + (v[2] - v[0]) * (y1 - v[1]) / (v[3] - v[1])
                        target = min(ex, at_top, at_bottom) if side == 0 else max(ex, at_top, at_bottom)
                        if side == 0 and x0 - w * .08 <= target <= x0:
                            completed[0] = min(completed[0], target)
                        elif side == 2 and x1 <= target <= x1 + w * .08:
                            completed[2] = max(completed[2], target)
                        connected.append(side)
                        break
            if {0, 2} <= set(connected):
                completed[1] = min(completed[1], a[1], a[3])
                completed[3] = max(completed[3], a[1], a[3])
    return tuple(completed)


def boundary_issues(box, tokens, hs, page):
    w, h = page
    x0, y0, x1, y1 = box
    issues = []
    header = re.compile(r"^(?:ITEM|DESCRIPTION|NO\.?\s*REQ\.?|MATL\.?|MATERIAL|REVISIONS?)$")
    for t in tokens:
        if (label_groups([t]) or header.fullmatch(t.text.upper().strip())):
            near = (x0 - w * .035 <= t.cx <= x1 + w * .035
                    and y0 - h * .025 <= t.cy <= y1 + h * .025)
            contained = x0 <= t.x0 and t.x1 <= x1 and y0 <= t.y0 and t.y1 <= y1
            if near and not contained:
                issues.append('Adjacent metadata or header is outside crop: ' + t.text)
    if x1 - x0 > 1.5 * (y1 - y0):
        for side in ('left', 'right'):
            levels = []
            for a in hs:
                overlap = max(0, min(a[2], x1) - max(a[0], x0))
                yy = (a[1] + a[3]) / 2
                crosses = (x0 - w * .10 <= a[0] < x0 - w * .008 if side == 'left'
                           else x1 + w * .008 < a[2] <= x1 + w * .10)
                if (crosses and overlap >= .60 * (x1 - x0)
                        and y0 - h * .008 <= yy <= y1 + h * .008
                        and a[2] - a[0] <= w * .72):
                    levels.append(yy)
            if levels and max(levels) - min(levels) > .30 * (y1 - y0):
                issues.append('Multiple table rules continue beyond ' + side + ' edge')
    return sorted(set(issues))


def protect_content(box, tokens, page):
    w, h = page
    members = _inside(tokens, box)
    for t in tokens:
        if (box[0] <= t.cx <= box[2] and t.y1 >= box[3]
                and box[3] - h * .035 <= t.y0 <= box[3] + h * .035
                and is_drawing_number(t.text)):
            members.append(t)
        if (box[0] <= t.cx <= box[2] and box[1] - h * .035 <= t.y0 < box[1]
                and re.fullmatch(r"REVISION[S]?|ITEM|DESCRIPTION|NO\.?\s*REQ\.?",
                                 t.text.upper().strip())):
            members.append(t)
        side_label = t.h > 2 * max(t.w, 1) and re.search(
            r"TITLE|ITLE|DRG\W*NO", t.text.upper())
        if (side_label and box[1] <= t.cy <= box[3]
                and box[0] - w * .025 <= t.cx <= box[2] + w * .025):
            members.append(t)
    pad = max(2, h * .002)
    union = _union([box] + [t.bbox for t in members])
    return (max(0, union[0] - pad), max(0, union[1] - pad),
            min(w, union[2] + pad), min(h, union[3] + pad))


def find_table(plain: np.ndarray, tokens: list[Token]) -> TableProposal:
    H, W = plain.shape
    if min(H, W) < 48 or not tokens:
        return TableProposal(None, False, detail={"reason": "Insufficient image or OCR evidence"})
    scale = min(2400 / W, 1)
    small = cv2.resize(plain, (round(W * scale), round(H * scale)), interpolation=cv2.INTER_AREA)
    h, w = small.shape
    sx, sy = w / W, h / H
    ts = [Token(t.text, (t.x0 * sx, t.y0 * sy, t.x1 * sx, t.y1 * sy), t.conf)
          for t in tokens]
    hs, vs, diag = detect_segments(small)
    geometric = _frame_candidates(hs, vs, w, h)
    text = text_candidates(ts, (w, h))
    ranked = []
    ownership_rejected = 0
    for *box, source in geometric + text:
        core = metadata_core(_inside(ts, box), (w,h))
        column = core_column(core, vs, (w,h))
        reference = []
        if column:
            if (box[2]-box[0] > 1.6*(column[2]-column[0]) + w*.02
                    or box[0] < column[0]-w*.04):
                ownership_rejected += 1
                continue
            reference = column_members(ts, column)
        box = complete_horizontal_frame(box, hs, vs, (w, h))
        if not column and box[2]-box[0] > 1.5*(box[3]-box[1]):
            box = grow_text_up(box, ts, (w,h))
        members = _inside(ts, box)
        gs = label_groups(column_members(members, column) if column else members)
        if len(gs) < 3 or len(gs & OPERATORS) < 2:
            continue
        support = _support(box, hs, vs, w, h)
        reliable = len(support) >= 2 and (
            {"top", "bottom"} <= support.keys() or {"left", "right"} <= support.keys())
        area = (box[2] - box[0]) * (box[3] - box[1]) / (w * h)
        coverage = content_coverage(box, reference)
        score = len(gs) * 2 - 4 * area
        ranked.append({"bbox": box, "source": source, "groups": sorted(gs),
                       "support": support, "reliable": reliable, "score": score,
                       "coverage": coverage, "column": column, "reference": reference})
    ranked.sort(key=lambda c: (c["reliable"], c['coverage'], c["score"]), reverse=True)
    diag.update(frame_candidates=len(geometric), text_candidates=len(text),
                evaluated_candidates=len(ranked), ownership_rejected_candidates=ownership_rejected)
    safe = []
    for candidate in ranked:
        if not candidate['reliable']:
            continue
        protected = protect_content(candidate['bbox'], ts, (w, h))
        problems = boundary_issues(protected, ts, hs, (w, h))
        omitted = [t.text for t in candidate['reference'] if header_text(t.text)
                   and (t.cy < protected[1] or t.cy > protected[3])]
        if omitted:
            problems.append('Column heading omitted: ' + '; '.join(omitted))
        expanded = grow_text_up(protected, ts, (w,h)) if not candidate['column'] else protected
        if expanded[1] < protected[1] - h*.005:
            problems.append('Aligned title or company text remains above crop')
        before = candidate['bbox']
        if ((protected[2] - protected[0]) * (protected[3] - protected[1])
                > 1.5 * (before[2] - before[0]) * (before[3] - before[1])):
            problems.append('Content protection requires excessive expansion')
        candidate['boundary_issues'] = problems
        candidate['protected'] = protected
        if not problems:
            safe.append(candidate)
    diag['boundary_rejected_candidates'] = sum(bool(c.get('boundary_issues')) for c in ranked)
    if not safe:
        diag["reason"] = "No candidate has both grouped metadata and sufficient local frame support"
        if ranked and ranked[0].get('boundary_issues'):
            diag['reason'] = 'Candidate boundary is incomplete or ambiguous'
            diag['boundary_issues'] = ranked[0]['boundary_issues']
            diag['rejected_candidate_bbox'] = [round(v / (sx if i % 2 == 0 else sy))
                                               for i,v in enumerate(ranked[0]['protected'])]
        return TableProposal(None, False, detail=diag)
    chosen = safe[0]
    box = chosen['protected']
    bbox = (math.floor(box[0] / sx), math.floor(box[1] / sy),
            min(W, math.ceil(box[2] / sx)), min(H, math.ceil(box[3] / sy)))
    diag.update(groups=chosen["groups"], supported_edges=sorted(chosen["support"]),
                column_content_coverage=chosen['coverage'],
                metadata_column=([round(v/(sx if i%2==0 else sy)) for i,v in enumerate(chosen['column'])]
                                 if chosen['column'] else None),
                source=chosen["source"], before_protection=[round(v / (sx if i % 2 == 0 else sy))
                                                          for i, v in enumerate(chosen["bbox"])])
    return TableProposal(bbox, True, chosen["source"], diag)
