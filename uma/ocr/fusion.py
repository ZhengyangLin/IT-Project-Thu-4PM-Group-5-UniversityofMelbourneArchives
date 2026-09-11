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
