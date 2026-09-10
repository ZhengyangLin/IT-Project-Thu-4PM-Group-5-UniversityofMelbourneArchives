from __future__ import annotations

import logging
import re


from .config import ConfidenceCfg
from .models import FieldValue
from .patterns import (DATE_RE, DRAUGHTSPERSON_CONFLICT_RE, DRAWING_NO_RE,
                       SCALE_RE, YEAR_RE, is_drawing_number)

log = logging.getLogger(__name__)



def format_check(fv: FieldValue, year_range=(1900, 2000)) -> float | None:
    # Check whether the field value matches its expected format.
    v = fv.corrected or fv.verbatim or ""

    if fv.name == "scale":
        return 1.0 if SCALE_RE.search(v) else 0.0

    if fv.name == "date":
        m = YEAR_RE.search(v)
        if not m:
            return 0.0
        return 1.0 if year_range[0] <= int(m.group(1)) <= year_range[1] else 0.0

    if fv.name == "drawing_number":
        return 1.0 if is_drawing_number(v) else 0.0

    if fv.name == "draughtsperson":
        # Reject values that look like another field instead of a person's name.
        text = " ".join(v.split()).strip()
        if not text:
            return 0.0
        if DATE_RE.search(text):
            return 0.0
        if DRAWING_NO_RE.search(text):
            return 0.0
        if SCALE_RE.search(text):
            return 0.0
        if DRAUGHTSPERSON_CONFLICT_RE.search(text):
            return 0.0
        if not re.search(r"[A-Za-z]", text):
            return 0.0
        return 1.0

    return None



def fuse(fv: FieldValue, cfg: ConfidenceCfg) -> FieldValue:
    # Combine different signals into one final confidence score.
    s = fv.signals

    # Failed OCR evidence must be reviewed manually.
    if s.get("evidence_ok") is False:
        fv.confidence = 0.0
        fv.status = "needs_review"
        fv.notes.append("Verbatim evidence check failed; confidence was forcibly downgraded")
        return fv

    # Handle fields where no OCR value was extracted.
    if fv.verbatim is None and fv.reason:
        fv.confidence = 0.0
        fv.status = "rejected" if fv.reason in ("illegible", "low_resolution") \
            else "needs_review"
        return fv

    # Store available confidence components.
    parts: list[tuple[float, float]] = []

    if "agreement" in s:
        agr = float(s["agreement"])
        if agr < 1.0 and s.get("adjudicated"):
            agr = max(agr, cfg.agreement_after_adjudication)
            s["agreement_adjusted"] = agr
        parts.append((cfg.w_agreement, agr))

    if "ocr_conf" in s:
        parts.append((cfg.w_ocr_conf, float(s["ocr_conf"])))

    # Add format validation to the confidence calculation.
    fmt = format_check(fv)
    if fmt is not None:
        parts.append((cfg.w_format, fmt))
        s["format"] = fmt

    # Add LLM confidence if available.
    lc = s.get("llm_confidence")
    if lc:
        parts.append((cfg.w_llm_conf, cfg.llm_conf_map.get(str(lc).lower(), 0.5)))

    if not parts:
        fv.confidence = 0.0
        fv.status = "needs_review"
        return fv

    # Calculate the weighted final confidence.
    total_w = sum(w for w, _ in parts)
    fv.confidence = round(sum(w * v for w, v in parts) / total_w, 3)

    # Corrected values have stricter confidence limits.
    if s.get("corrected"):
        fv.confidence = min(fv.confidence, cfg.correction_conf_ceiling)
        if s.get("format") is not None and s["format"] < 1.0:
            fv.confidence = min(fv.confidence, cfg.t_low)
            fv.notes.append(
                f"Corrected value {fv.corrected!r} does not match the expected "
                f"format for {fv.name}; confidence was forcibly downgraded")

    # Repaired OCR tokens should not be auto-accepted.
    if s.get("tokens_repaired"):
        fv.confidence = min(fv.confidence, cfg.t_high - 0.01)

    # Set the final review status.
    if fv.confidence >= cfg.t_high:
        fv.status = "auto_accept"
    elif fv.confidence >= cfg.t_low:
        fv.status = "needs_review"
    else:
        fv.status = "rejected"

    return fv