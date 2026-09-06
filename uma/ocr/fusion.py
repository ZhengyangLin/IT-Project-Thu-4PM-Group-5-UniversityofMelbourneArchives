from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

from ..models import Candidate, Token, io_min, iou
from .engines import Engine, RawToken

log = logging.getLogger(__name__)


# Normalize OCR text before comparison.
def _norm(s: str) -> str:
    return " ".join(s.upper().split())


# Group overlapping OCR results into shared regions.
def align(raws: list[RawToken], iou_thresh: float) -> list[list[RawToken]]:
    groups: list[list[RawToken]] = []
    for r in sorted(raws, key=lambda t: (t.bbox[1], t.bbox[0])):
        placed = dropped = False
        for g in groups:
            if not any(iou(r.bbox, x.bbox) >= iou_thresh for x in g):
                continue
            if any(x.engine == r.engine for x in g):
                dropped = True
                break
            g.append(r)
            placed = True
            break
        if not (placed or dropped):
            groups.append([r])
    return groups


# Fuse OCR candidates into unified tokens.
def fuse(raws: list[RawToken], iou_thresh: float, min_conf: float,
         n_engines: int) -> list[Token]:
    tokens: list[Token] = []
    for g in align([r for r in raws if r.conf >= min_conf], iou_thresh):
        cands = [Candidate(r.text, r.engine, r.conf) for r in g]
        best = max(cands, key=lambda c: c.conf)
        texts = [_norm(c.text) for c in cands]
        n_same = sum(1 for t in texts if t == _norm(best.text))
        agreement = n_same / max(len(cands), 1)
        if len(cands) < n_engines:
            agreement *= len(cands) / n_engines
        bbox = (min(r.bbox[0] for r in g), min(r.bbox[1] for r in g),
                max(r.bbox[2] for r in g), max(r.bbox[3] for r in g))
        tokens.append(Token(text=best.text.strip(), bbox=bbox, conf=best.conf,
                            candidates=cands, agreement=round(agreement, 3)))
    tokens.sort(key=lambda t: (t.cy, t.x0))
    return tokens


# Run all configured OCR engines and collect their results.
def read_all(gray: np.ndarray, engines: list[Engine], upscale: float,
             iou_thresh: float, min_conf: float, **kw) -> tuple[list[Token], dict]:
    raws, used = [], []
    per_engine = defaultdict(int)
    import time
    for eng in engines:
        t0 = time.time()
        log.info("→ %s text recognition started (%dx%d, upscale %.1fx)",
                 eng.name, gray.shape[1], gray.shape[0], upscale)
        try:
            out = eng.read(gray, upscale=upscale, **kw)
        except Exception as e:
            log.error("OCR engine %s crashed and will be skipped: %s", eng.name, e)
            continue
        log.info("← %s finished in %.1fs: %d blocks", eng.name,
                 time.time() - t0, len(out))
        raws.extend(out)
        per_engine[eng.name] = len(out)
        used.append(eng.name)

    tokens = fuse(raws, iou_thresh, min_conf, n_engines=max(len(used), 1))
    diag = {
        "engines_used": used,
        "engines_configured": [e.name for e in engines],
        "tokens_per_engine": dict(per_engine),
        "token_count": len(tokens),
        "disagreements": sum(1 for t in tokens if t.agreement < 1.0),
    }
    if len(used) < len(engines):
        log.warning("%d OCR engines were configured, but only %d returned "
                    "results (%s). Agreement is calculated across %d engines.",
                    len(engines), len(used), used or "none", max(len(used), 1))
    return tokens, diag


def _same_text(a: str, b: str) -> bool:
    from rapidfuzz import fuzz
    x, y = _norm(a), _norm(b)
    if not x or not y:
        return False
    if x in y or y in x:
        return True
    return fuzz.ratio(x, y) >= 70


def _contains(outer, inner, ratio: float = 0.7) -> bool:
    ox0 = max(outer[0], inner[0]); oy0 = max(outer[1], inner[1])
    ox1 = min(outer[2], inner[2]); oy1 = min(outer[3], inner[3])
    if ox1 <= ox0 or oy1 <= oy0:
        return False
    ov = (ox1 - ox0) * (oy1 - oy0)
    a = max((inner[2] - inner[0]) * (inner[3] - inner[1]), 1)
    return ov / a >= ratio


# Merge fine and coarse OCR passes while preserving conflicts.
def merge_passes(fine: list[Token], coarse: list[Token],
                 iomin_thresh: float = 0.55) -> tuple[list[Token], dict]:
    used = set()
    conflicts = 0
    for ft in fine:
        cands = sorted(
            ((io_min(ft.bbox, ct.bbox), i, ct) for i, ct in enumerate(coarse)
             if i not in used),
            key=lambda x: -x[0])
        for score, i, ct in cands:
            if score < iomin_thresh:
                break
            if not _same_text(ft.text, ct.text) and score < 0.85:
                continue
            used.add(i)
            if _norm(ct.text) != _norm(ft.text):
                ft.candidates.append(
                    Candidate(ct.text, f"{ct.candidates[0].engine if ct.candidates else 'ocr'}@coarse",
                              ct.conf, source_bbox=ct.bbox))
                n = len(ft.candidates)
                same = sum(1 for c in ft.candidates
                           if _norm(c.text) == _norm(ft.text))
                ft.agreement = round(same / max(n, 1), 3)
                conflicts += 1
            break

    for i, ct in enumerate(coarse):
        if i in used:
            continue
        if any(_contains(ct.bbox, ft.bbox) for ft in fine):
            used.add(i)

    recovered = []
    for i, ct in enumerate(coarse):
        if i in used:
            continue
        ct.agreement = min(ct.agreement, 0.5)
        ct.candidates = [Candidate(c.text, f"{c.engine}@coarse", c.conf)
                         for c in ct.candidates] or \
                        [Candidate(ct.text, "ocr@coarse", ct.conf)]
        recovered.append(ct)

    out = sorted(fine + recovered, key=lambda t: (t.cy, t.x0))
    diag = {"fine_only": len(fine) - conflicts, "conflicts": conflicts,
            "recovered_from_coarse": len(recovered), "merged_total": len(out)}
    if conflicts or recovered:
        log.info("Coarse/fine OCR merge: %d text conflicts (both candidates "
                 "retained); restored %d coarse-only blocks",
                 conflicts, len(recovered))
    return out, diag


_UNIT_WORDS = ("INCH", "INCHES", "FOOT", "FEET", "FT", "IN")


def _looks_incomplete(row: list[Token]) -> bool:
    text = " ".join(t.text.upper() for t in row)
    if not any(w in text for w in _UNIT_WORDS):
        return False
    import re as _re
    for w in ("INCH", "FOOT", "FEET"):
        for m in _re.finditer(rf"\b{w}\b", text):
            before = text[max(0, m.start() - 12):m.start()]
            if not _re.search(r"\d\s*$|\d\s*/\s*\d\s*$", before):
                return True
    return False


# Re-read suspicious gaps to recover missed narrow characters.
def probe_gaps(gray, tokens: list[Token], engines: list,
               upscale: float = 1.0,
               gap_factor: float = 1.5, min_gap_px: int = 20,
               max_probes: int = 24) -> tuple[list[Token], dict]:
    tess = next((e for e in engines if e.name == "tesseract"), None)
    if tess is None or len(tokens) < 4:
        return [], {"skipped": "Tesseract unavailable or too few tokens"}

    rows: list[list[Token]] = []
    for t in sorted(tokens, key=lambda t: (t.cy, t.x0)):
        for r in rows:
            if abs(t.cy - r[-1].cy) <= max(r[-1].h, 1) * 0.6:
                r.append(t)
                break
        else:
            rows.append([t])

    gaps = []
    for r in rows:
        r.sort(key=lambda t: t.x0)
        for a, b in zip(r, r[1:]):
            g = b.x0 - a.x1
            if g > 0:
                gaps.append((g, a, b))
    if not gaps:
        return [], {"skipped": "No gaps found"}

    widths = sorted(g for g, _, _ in gaps)
    normal = widths[len(widths) // 2]
    thresh = max(normal * gap_factor, min_gap_px)
    suspects = [(g, a, b) for g, a, b in gaps if g > thresh]

    for r in rows:
        if not _looks_incomplete(r):
            continue
        for a, b in zip(r, r[1:]):
            g = b.x0 - a.x1
            if g >= min_gap_px and not any(x[1] is a and x[2] is b for x in suspects):
                suspects.append((g, a, b))

    suspects.sort(key=lambda x: -x[0])
    suspects = suspects[:max_probes]

    H, W = gray.shape
    found: list[Token] = []
    for g, a, b in suspects:
        pad = max(int(a.h * 0.25), 4)
        x0, x1 = max(a.x1 - pad, 0), min(b.x0 + pad, W)
        y0 = max(min(a.y0, b.y0) - pad, 0)
        y1 = min(max(a.y1, b.y1) + pad, H)
        if x1 - x0 < 6 or y1 - y0 < 6:
            continue
        crop = gray[y0:y1, x0:x1]
        f = max(1.0, min(60.0 / max(a.h, 1), 8.0))
        try:
            out = tess.read(crop, upscale=f, psms=(7, 10),
                            whitelist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ/.-")
        except Exception as e:
            log.debug("Gap probing failed: %s", e)
            continue
        for rt in out:
            txt = rt.text.strip()
            if not txt or len(txt) > 4:
                continue
            bb = (x0 + rt.bbox[0], y0 + rt.bbox[1],
                  x0 + rt.bbox[2], y0 + rt.bbox[3])
            found.append(Token(text=txt, bbox=bb, conf=rt.conf * 0.85,
                               candidates=[Candidate(txt, "tesseract@gap", rt.conf)],
                               agreement=0.5))

    diag = {"normal_gap_px": int(normal), "threshold_px": int(thresh),
            "suspect_gaps": len(suspects), "recovered": len(found),
            "texts": [t.text for t in found[:10]]}
    if found:
        log.info("Gap probing: %d anomalous gaps; recovered %d narrow "
                 "characters %s",
                 len(suspects), len(found), [t.text for t in found[:8]])
    return found, diag