from __future__ import annotations

from typing import Any


# Shared field names used to extract OCR results.
OCR_RESULT_FIELDS = (
    "drawing_number",
    "project_name",
    "drawing_title",
    "architect",
    "draughtsperson",
    "date",
    "scale",
    "drawing_type",
)
OCR_REASON_CODES = frozenset({
    "illegible",
    "not_present",
    "ambiguous",
    "low_resolution",
    "requires_visual",
})


# Flatten note collections into unique, non-empty strings.
def note_values(*values: Any) -> list[str]:
    notes: list[str] = []

    def append(value: Any) -> None:
        if isinstance(value, (list, tuple, set)):
            for item in value:
                append(item)
            return
        if value is None:
            return
        text = str(value).strip()
        if text and text not in notes:
            notes.append(text)

    for value in values:
        append(value)
    return notes


def split_reason_and_notes(
    raw_reason: Any, raw_notes: Any,
) -> tuple[str | None, list[str]]:
    notes = note_values(raw_notes)
    # For collections, use the first known reason code and keep the rest as notes.
    if isinstance(raw_reason, (list, tuple, set)):
        reason: str | None = None
        for text in note_values(raw_reason):
            if reason is None and text in OCR_REASON_CODES:
                reason = text
            elif text not in notes:
                notes.append(text)
        return reason, notes
    if raw_reason is None:
        return None, notes
    # Preserve non-empty scalar reasons, including unknown codes.
    return str(raw_reason).strip() or None, notes


# Combine valid evidence regions into a single enclosing box.
def evidence_regions_bbox(signals: Any) -> dict[str, int] | None:
    if not isinstance(signals, dict):
        return None

    boxes: list[list[int]] = []
    regions = signals.get("evidence_regions")
    if isinstance(regions, list):
        for region in regions:
            bbox = region.get("bbox") if isinstance(region, dict) else None
            if (
                isinstance(bbox, (list, tuple))
                and len(bbox) == 4
                and all(isinstance(value, (int, float)) for value in bbox)
                and bbox[0] < bbox[2]
                and bbox[1] < bbox[3]
            ):
                boxes.append([int(value) for value in bbox])
    if boxes:
        return {
            "x0": min(box[0] for box in boxes),
            "y0": min(box[1] for box in boxes),
            "x1": max(box[2] for box in boxes),
            "y1": max(box[3] for box in boxes),
        }

    # Fall back to the overall evidence box when no valid regions are available.
    total = signals.get("evidence_bbox")
    if not isinstance(total, dict):
        return None
    values = [total.get(key) for key in ("x0", "y0", "x1", "y1")]
    if (
        not all(isinstance(value, (int, float)) for value in values)
        or values[0] >= values[2]
        or values[1] >= values[3]
    ):
        return None
    return dict(zip(("x0", "y0", "x1", "y1"), map(int, values)))


def tokens_bbox(
    tokens: list[int], token_map: dict[Any, Any],
) -> dict[str, int] | None:
    boxes: list[list[int]] = []
    for token_id in tokens:
        # Accept token IDs stored as either strings or integers.
        token = token_map.get(str(token_id), token_map.get(token_id))
        bbox = token.get("bbox") if isinstance(token, dict) else None
        if (
            isinstance(bbox, (list, tuple))
            and len(bbox) == 4
            and all(isinstance(value, (int, float)) for value in bbox)
        ):
            boxes.append([int(value) for value in bbox])
    if not boxes:
        return None
    return {
        "x0": min(box[0] for box in boxes),
        "y0": min(box[1] for box in boxes),
        "x1": max(box[2] for box in boxes),
        "y1": max(box[3] for box in boxes),
    }


# Collect OCR field values and metadata into a flat result mapping.
def extract_ocr_result(record: Any | None) -> dict[str, Any]:
    fields = getattr(record, "fields", {}) if record is not None else {}
    token_map = (
        getattr(record, "token_map", {}) if record is not None else {}
    )
    result: dict[str, Any] = {}
    for field_name in OCR_RESULT_FIELDS:
        field_value = fields.get(field_name)
        tokens = list(getattr(field_value, "tokens", []) or [])
        verbatim = getattr(field_value, "verbatim", None)
        corrected = getattr(field_value, "corrected", None)
        confidence = getattr(field_value, "confidence", None)
        # Prefer corrected text, falling back to verbatim OCR text.
        result[field_name] = corrected or verbatim
        result[f"{field_name}_verbatim"] = verbatim
        result[f"{field_name}_corrected"] = corrected
        result[f"{field_name}_correction_basis"] = getattr(
            field_value, "correction_basis", None
        )
        result[f"{field_name}_conf"] = (
            round(float(confidence), 3)
            if confidence is not None else None
        )
        result[f"{field_name}_source"] = (
            getattr(field_value, "source", None) or None
        )
        result[f"{field_name}_tokens"] = tokens
        signals = getattr(field_value, "signals", {})
        # Prefer evidence bounds, then fall back to token bounds.
        result[f"{field_name}_tokens_bbox"] = (
            evidence_regions_bbox(signals)
            or tokens_bbox(tokens, token_map)
        )
        reason, notes = split_reason_and_notes(
            getattr(field_value, "reason", None),
            getattr(field_value, "notes", None),
        )
        result[f"{field_name}_reason"] = reason
        result[f"{field_name}_notes"] = notes
        result[f"{field_name}_review_status"] = getattr(
            field_value, "status", None
        )
    return result
