from uma.config import ConfidenceCfg, RuleCfg
from uma.models import FieldValue, Token
from uma.rules import extract, infer_drawing_type
from uma.verify import format_check, fuse


def test_rule_extraction_produces_locked_and_pending_fields():
    tokens = [
        Token("DATE", (10, 10, 50, 30), 0.98, tid=0),
        Token("4.11.1963", (60, 10, 135, 30), 0.97, tid=1),
        Token("S24", (10, 55, 45, 75), 0.98, tid=2),
        Token("MULLION", (10, 100, 85, 120), 0.98, tid=3),
        Token("DETAILS", (95, 100, 160, 120), 0.98, tid=4),
    ]

    locked, pending, rejected = extract(tokens, RuleCfg())

    # Current extraction preserves the date as written on the drawing.
    assert locked["date"].corrected == "4.11.1963"
    assert locked["date"].verbatim == "4.11.1963"
    assert locked["drawing_number"].verbatim == "S24"
    assert infer_drawing_type("MULLION DETAILS") == "detail"
    assert "architect" in pending
    assert isinstance(rejected, dict)


def test_format_check_and_confidence_fusion_set_review_status():
    field = FieldValue(
        name="drawing_number",
        verbatim="S24",
        confidence=0.0,
        signals={"agreement": 1.0, "ocr_conf": 0.98},
    )

    assert format_check(field) == 1.0

    fused = fuse(field, ConfidenceCfg())

    assert fused.confidence >= 0.75
    assert fused.status == "auto_accept"


def test_failed_evidence_forces_manual_review():
    field = FieldValue(
        name="project_name",
        verbatim="STATE GOVERNMENT OFFICES",
        confidence=0.9,
        signals={"evidence_ok": False},
    )

    fused = fuse(field, ConfidenceCfg())

    assert fused.confidence == 0.0
    assert fused.status == "needs_review"
