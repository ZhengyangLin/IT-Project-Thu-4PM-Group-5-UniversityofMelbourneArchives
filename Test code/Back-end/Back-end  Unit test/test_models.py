import pytest

from uma.models import (
    iou,
    io_min,
    union,
    h_overlap,
    Candidate,
    Token,
    Line,
    Block,
    Cluster,
    FieldValue,
    SheetRecord,
)


# ============================================================
# iou
# ============================================================

def test_iou_identical_boxes():
    a = (0, 0, 10, 10)
    b = (0, 0, 10, 10)

    assert iou(a, b) == pytest.approx(1.0)


def test_iou_partial_overlap():
    a = (0, 0, 10, 10)
    b = (5, 5, 15, 15)

    # intersection = 5 * 5 = 25
    # union = 100 + 100 - 25 = 175
    assert iou(a, b) == pytest.approx(25 / 175)


def test_iou_no_overlap():
    a = (0, 0, 10, 10)
    b = (20, 20, 30, 30)

    assert iou(a, b) == 0.0


def test_iou_touching_edge():
    a = (0, 0, 10, 10)
    b = (10, 0, 20, 10)

    assert iou(a, b) == 0.0


def test_iou_touching_corner():
    a = (0, 0, 10, 10)
    b = (10, 10, 20, 20)

    assert iou(a, b) == 0.0


def test_iou_one_inside_another():
    outer = (0, 0, 20, 20)
    inner = (5, 5, 15, 15)

    # intersection = 100
    # union = 400
    assert iou(outer, inner) == pytest.approx(0.25)


# ============================================================
# io_min
# ============================================================

def test_io_min_identical():
    a = (0, 0, 10, 10)

    assert io_min(a, a) == pytest.approx(1.0)


def test_io_min_inner_box():
    outer = (0, 0, 20, 20)
    inner = (5, 5, 15, 15)

    # smaller box is completely contained
    assert io_min(outer, inner) == pytest.approx(1.0)


def test_io_min_partial_overlap():
    a = (0, 0, 10, 10)
    b = (5, 0, 15, 10)

    # intersection = 50
    # smaller area = 100
    assert io_min(a, b) == pytest.approx(0.5)


def test_io_min_no_overlap():
    a = (0, 0, 10, 10)
    b = (20, 20, 30, 30)

    assert io_min(a, b) == 0.0


def test_io_min_touching():
    a = (0, 0, 10, 10)
    b = (10, 0, 20, 10)

    assert io_min(a, b) == 0.0


# ============================================================
# union
# ============================================================

def test_union_single_box():
    boxes = [
        (10, 20, 30, 40)
    ]

    assert union(boxes) == (10, 20, 30, 40)


def test_union_multiple_boxes():
    boxes = [
        (10, 20, 30, 40),
        (5, 25, 50, 60),
        (15, 10, 25, 70),
    ]

    assert union(boxes) == (5, 10, 50, 70)


def test_union_two_non_overlapping_boxes():
    boxes = [
        (0, 0, 10, 10),
        (100, 100, 120, 130),
    ]

    assert union(boxes) == (0, 0, 120, 130)


def test_union_nested_boxes():
    boxes = [
        (0, 0, 100, 100),
        (20, 20, 40, 40),
    ]

    assert union(boxes) == (0, 0, 100, 100)


# ============================================================
# h_overlap
# ============================================================

def test_h_overlap_identical():
    a = (0, 0, 10, 10)

    assert h_overlap(a, a) == pytest.approx(1.0)


def test_h_overlap_full_containment():
    outer = (0, 0, 20, 20)
    inner = (5, 100, 15, 110)

    # Function only considers horizontal coordinates.
    assert h_overlap(outer, inner) == pytest.approx(1.0)


def test_h_overlap_partial():
    a = (0, 0, 10, 10)
    b = (5, 20, 15, 30)

    assert h_overlap(a, b) == pytest.approx(0.5)


def test_h_overlap_none():
    a = (0, 0, 10, 10)
    b = (20, 0, 30, 10)

    assert h_overlap(a, b) == 0.0


def test_h_overlap_touching():
    a = (0, 0, 10, 10)
    b = (10, 0, 20, 10)

    assert h_overlap(a, b) == 0.0


# ============================================================
# Candidate
# ============================================================

def test_candidate_creation():
    candidate = Candidate(
        text="SITE PLAN",
        engine="paddle",
        conf=0.95,
        source_bbox=(10, 20, 100, 40),
    )

    assert candidate.text == "SITE PLAN"
    assert candidate.engine == "paddle"
    assert candidate.conf == pytest.approx(0.95)
    assert candidate.source_bbox == (10, 20, 100, 40)


def test_candidate_default_source_bbox():
    candidate = Candidate(
        text="TEST",
        engine="ocr",
        conf=0.8,
    )

    assert candidate.source_bbox is None


# ============================================================
# Token properties
# ============================================================

def test_token_coordinates():
    token = Token(
        text="DRAWING",
        bbox=(10, 20, 110, 70),
        conf=0.9,
    )

    assert token.x0 == 10
    assert token.y0 == 20
    assert token.x1 == 110
    assert token.y1 == 70


def test_token_width_height():
    token = Token(
        text="DRAWING",
        bbox=(10, 20, 110, 70),
        conf=0.9,
    )

    assert token.w == 100
    assert token.h == 50


def test_token_center():
    token = Token(
        text="DRAWING",
        bbox=(10, 20, 110, 70),
        conf=0.9,
    )

    assert token.cx == pytest.approx(60.0)
    assert token.cy == pytest.approx(45.0)


def test_token_defaults():
    token = Token(
        text="TEXT",
        bbox=(0, 0, 10, 10),
        conf=0.5,
    )

    assert token.candidates == []
    assert token.agreement == 1.0
    assert token.tid == -1


def test_token_custom_values():
    token = Token(
        text="TEXT",
        bbox=(0, 0, 10, 10),
        conf=0.8,
        agreement=0.7,
        tid=12,
    )

    assert token.agreement == pytest.approx(0.7)
    assert token.tid == 12


# ============================================================
# Token alternatives
# ============================================================

def test_token_alternatives():
    candidates = [
        Candidate(
            text="603",
            engine="paddle",
            conf=0.95,
        ),
        Candidate(
            text="603/A1",
            engine="coarse",
            conf=0.80,
        ),
        Candidate(
            text="603",
            engine="other",
            conf=0.75,
        ),
    ]

    token = Token(
        text="603",
        bbox=(0, 0, 100, 20),
        conf=0.95,
        candidates=candidates,
    )

    alternatives = token.alternatives()

    assert len(alternatives) == 1
    assert alternatives[0].text == "603/A1"


def test_token_alternatives_ignore_whitespace():
    candidates = [
        Candidate(
            text=" SITE PLAN ",
            engine="engine1",
            conf=0.9,
        ),
        Candidate(
            text="SITE PLAN",
            engine="engine2",
            conf=0.8,
        ),
        Candidate(
            text="SITE PL4N",
            engine="engine3",
            conf=0.7,
        ),
    ]

    token = Token(
        text="SITE PLAN",
        bbox=(0, 0, 100, 20),
        conf=0.9,
        candidates=candidates,
    )

    alternatives = token.alternatives()

    assert len(alternatives) == 1
    assert alternatives[0].text == "SITE PL4N"


def test_token_no_alternatives():
    token = Token(
        text="SITE",
        bbox=(0, 0, 10, 10),
        conf=0.9,
        candidates=[],
    )

    assert token.alternatives() == []


# ============================================================
# Line
# ============================================================

def test_line_text():
    t1 = Token(
        text="SITE",
        bbox=(10, 20, 40, 40),
        conf=0.9,
    )

    t2 = Token(
        text="PLAN",
        bbox=(50, 20, 100, 40),
        conf=0.9,
    )

    line = Line(
        tokens=[t1, t2],
        row_no=1,
    )

    assert line.text == "SITE PLAN"


def test_line_bbox():
    t1 = Token(
        text="SITE",
        bbox=(10, 20, 40, 40),
        conf=0.9,
    )

    t2 = Token(
        text="PLAN",
        bbox=(50, 15, 100, 45),
        conf=0.9,
    )

    line = Line(tokens=[t1, t2])

    assert line.bbox == (10, 15, 100, 45)


def test_line_properties():
    token = Token(
        text="TEST",
        bbox=(10, 20, 110, 70),
        conf=0.9,
    )

    line = Line(tokens=[token])

    assert line.x0 == 10
    assert line.y0 == 20
    assert line.x1 == 110
    assert line.y1 == 70
    assert line.h == 50
    assert line.cy == pytest.approx(45.0)


# ============================================================
# Line gap_to
# ============================================================

def test_line_gap_same_position():
    line1 = Line(
        tokens=[
            Token(
                "A",
                (0, 0, 10, 10),
                1.0,
            )
        ]
    )

    line2 = Line(
        tokens=[
            Token(
                "B",
                (0, 0, 10, 10),
                1.0,
            )
        ]
    )

    assert line1.gap_to(line2) == pytest.approx(0.0)


def test_line_gap_horizontal():
    line1 = Line(
        tokens=[
            Token(
                "A",
                (0, 0, 10, 10),
                1.0,
            )
        ]
    )

    line2 = Line(
        tokens=[
            Token(
                "B",
                (20, 0, 30, 10),
                1.0,
            )
        ]
    )

    # Horizontal gap = 10
    assert line1.gap_to(line2) == pytest.approx(10.0)


def test_line_gap_vertical():
    line1 = Line(
        tokens=[
            Token(
                "A",
                (0, 0, 10, 10),
                1.0,
            )
        ]
    )

    line2 = Line(
        tokens=[
            Token(
                "B",
                (0, 20, 10, 30),
                1.0,
            )
        ]
    )

    # Vertical gap = 10
    # ky = 0.6
    # result = 6
    assert line1.gap_to(line2) == pytest.approx(6.0)


def test_line_gap_diagonal():
    line1 = Line(
        tokens=[
            Token(
                "A",
                (0, 0, 10, 10),
                1.0,
            )
        ]
    )

    line2 = Line(
        tokens=[
            Token(
                "B",
                (20, 20, 30, 30),
                1.0,
            )
        ]
    )

    # dx = 10
    # dy = 10
    # sqrt(10^2 + (0.6*10)^2)
    expected = (100 + 36) ** 0.5

    assert line1.gap_to(line2) == pytest.approx(expected)


def test_line_gap_custom_ky():
    line1 = Line(
        tokens=[
            Token(
                "A",
                (0, 0, 10, 10),
                1.0,
            )
        ]
    )

    line2 = Line(
        tokens=[
            Token(
                "B",
                (0, 20, 10, 30),
                1.0,
            )
        ]
    )

    assert line1.gap_to(
        line2,
        ky=1.0,
    ) == pytest.approx(10.0)


# ============================================================
# Block
# ============================================================

def test_block_area():
    block = Block(
        bbox=(10, 20, 110, 70),
    )

    assert block.area == 5000


def test_block_defaults():
    block = Block(
        bbox=(0, 0, 100, 100),
    )

    assert block.lines == []
    assert block.source == "xy_cut"
    assert block.is_title_block is False


def test_block_custom_values():
    line = Line(
        tokens=[
            Token(
                "TEST",
                (0, 0, 10, 10),
                1.0,
            )
        ]
    )

    block = Block(
        bbox=(0, 0, 100, 100),
        lines=[line],
        source="caption_resplit",
        is_title_block=True,
    )

    assert block.lines == [line]
    assert block.source == "caption_resplit"
    assert block.is_title_block is True


# ============================================================
# Cluster
# ============================================================

def test_cluster_bbox():
    line1 = Line(
        tokens=[
            Token(
                "SITE",
                (10, 10, 50, 30),
                0.9,
            )
        ]
    )

    line2 = Line(
        tokens=[
            Token(
                "PLAN",
                (20, 40, 100, 60),
                0.9,
            )
        ]
    )

    cluster = Cluster(
        lines=[line1, line2],
    )

    assert cluster.bbox == (10, 10, 100, 60)


def test_cluster_text():
    line1 = Line(
        tokens=[
            Token(
                "SITE",
                (0, 0, 10, 10),
                1.0,
            )
        ]
    )

    line2 = Line(
        tokens=[
            Token(
                "PLAN",
                (0, 20, 10, 30),
                1.0,
            )
        ]
    )

    cluster = Cluster(
        lines=[line1, line2],
    )

    assert cluster.text == "SITE PLAN"


def test_cluster_defaults():
    line = Line(
        tokens=[
            Token(
                "TEST",
                (0, 0, 10, 10),
                1.0,
            )
        ]
    )

    cluster = Cluster(
        lines=[line],
    )

    assert cluster.block_idx == -1
    assert cluster.score == 0.0


# ============================================================
# Cluster compactness
# ============================================================

def test_cluster_compactness_single_full_line():
    line = Line(
        tokens=[
            Token(
                "FULL",
                (0, 0, 100, 20),
                1.0,
            )
        ]
    )

    cluster = Cluster(
        lines=[line]
    )

    # bbox area = 100 * 20
    # ink area = 100 * 20
    assert cluster.compactness() == pytest.approx(1.0)


def test_cluster_compactness_multiple_lines():
    line1 = Line(
        tokens=[
            Token(
                "A",
                (0, 0, 50, 10),
                1.0,
            )
        ]
    )

    line2 = Line(
        tokens=[
            Token(
                "B",
                (0, 20, 50, 30),
                1.0,
            )
        ]
    )

    cluster = Cluster(
        lines=[line1, line2]
    )

    # bbox = (0,0,50,30)
    # area = 1500
    #
    # line1 ink = 50 * 10 = 500
    # line2 ink = 50 * 10 = 500
    #
    # total ink = 1000
    expected = 1000 / 1500

    assert cluster.compactness() == pytest.approx(expected)


# ============================================================
# FieldValue
# ============================================================

def test_field_value_defaults():
    fv = FieldValue(
        name="drawing_title"
    )

    assert fv.name == "drawing_title"
    assert fv.tokens == []
    assert fv.verbatim is None
    assert fv.corrected is None
    assert fv.correction_basis is None
    assert fv.source == ""
    assert fv.reason is None
    assert fv.confidence == 0.0
    assert fv.signals == {}
    assert fv.status == "needs_review"
    assert fv.notes == []


def test_field_value_full():
    fv = FieldValue(
        name="drawing_number",
        tokens=[1, 2],
        verbatim="603/A1",
        corrected="603/A1",
        correction_basis="alternative OCR",
        source="llm",
        reason=None,
        confidence=0.95,
        signals={"evidence_ok": True},
        status="accepted",
        notes=["verified"],
    )

    assert fv.name == "drawing_number"
    assert fv.tokens == [1, 2]
    assert fv.verbatim == "603/A1"
    assert fv.corrected == "603/A1"
    assert fv.correction_basis == "alternative OCR"
    assert fv.source == "llm"
    assert fv.confidence == pytest.approx(0.95)
    assert fv.signals["evidence_ok"] is True
    assert fv.status == "accepted"
    assert fv.notes == ["verified"]


# ============================================================
# SheetRecord
# ============================================================

def test_sheet_record_defaults():
    record = SheetRecord(
        node_id="node-001",
        path="/drawing/test.png",
    )

    assert record.node_id == "node-001"
    assert record.path == "/drawing/test.png"

    assert record.series == ""
    assert record.fields == {}
    assert record.title_bbox is None
    assert record.localization_method == ""
    assert record.localization_conf == 0.0
    assert record.grid == ""
    assert record.index_view == ""
    assert record.token_map == {}
    assert record.diagnostics == {}


def test_sheet_record_full():
    fv = FieldValue(
        name="drawing_title",
        tokens=[1],
        verbatim="SITE PLAN",
    )

    record = SheetRecord(
        node_id="node-123",
        path="/test/site_plan.png",
        series="architecture",
        fields={
            "drawing_title": fv
        },
        title_bbox=(100, 100, 500, 400),
        localization_method="anchor",
        localization_conf=0.95,
        grid="GRID",
        index_view="[1] SITE PLAN",
        token_map={
            1: "SITE PLAN"
        },
        diagnostics={
            "test": True
        },
    )

    assert record.node_id == "node-123"
    assert record.path == "/test/site_plan.png"
    assert record.series == "architecture"

    assert record.fields["drawing_title"] is fv

    assert record.title_bbox == (
        100,
        100,
        500,
        400,
    )

    assert record.localization_method == "anchor"
    assert record.localization_conf == pytest.approx(0.95)

    assert record.grid == "GRID"
    assert record.index_view == "[1] SITE PLAN"

    assert record.token_map == {
        1: "SITE PLAN"
    }

    assert record.diagnostics == {
        "test": True
    }


# ============================================================
# Mutable default independence
# ============================================================

def test_token_candidate_lists_are_independent():
    token1 = Token(
        "A",
        (0, 0, 10, 10),
        1.0,
    )

    token2 = Token(
        "B",
        (0, 0, 10, 10),
        1.0,
    )

    token1.candidates.append(
        Candidate(
            "ALT",
            "test",
            0.5,
        )
    )

    assert len(token1.candidates) == 1
    assert token2.candidates == []


def test_field_value_mutable_defaults_are_independent():
    fv1 = FieldValue(
        name="a"
    )

    fv2 = FieldValue(
        name="b"
    )

    fv1.tokens.append(1)
    fv1.signals["test"] = True
    fv1.notes.append("note")

    assert fv2.tokens == []
    assert fv2.signals == {}
    assert fv2.notes == []


def test_sheet_record_mutable_defaults_are_independent():
    r1 = SheetRecord(
        node_id="1",
        path="a"
    )

    r2 = SheetRecord(
        node_id="2",
        path="b"
    )

    r1.fields["test"] = FieldValue(
        name="test"
    )

    r1.token_map[1] = "TEXT"
    r1.diagnostics["ok"] = True

    assert r2.fields == {}
    assert r2.token_map == {}
    assert r2.diagnostics == {}