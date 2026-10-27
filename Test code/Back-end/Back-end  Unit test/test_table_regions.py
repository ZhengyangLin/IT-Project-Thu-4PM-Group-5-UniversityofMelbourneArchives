import numpy as np
import pytest
from unittest.mock import patch

from uma.models import Token
from uma.table_regions import (
    TableProposal,
    label_groups,
    _inside,
    _union,
    metadata_core,
    core_column,
    column_members,
    header_text,
    grow_text_up,
    content_coverage,
    merge_segments,
    detect_segments,
    text_candidates,
    _frame_candidates,
    _support,
    complete_horizontal_frame,
    boundary_issues,
    protect_content,
    find_table,
)


# ============================================================
# Helpers
# ============================================================

def tok(
    text,
    bbox=(0, 0, 100, 20),
    conf=0.9,
    tid=-1,
):
    t = Token(
        text=text,
        bbox=bbox,
        conf=conf,
    )
    t.tid = tid
    return t


PAGE = (1000, 1000)


# ============================================================
# TableProposal
# ============================================================

def test_table_proposal_defaults():
    p = TableProposal(
        bbox=(1, 2, 3, 4),
        reliable=True,
    )

    assert p.bbox == (1, 2, 3, 4)
    assert p.reliable is True
    assert p.method == "local_frame"
    assert p.detail == {}


# ============================================================
# label_groups
# ============================================================

@pytest.mark.parametrize(
    "text,expected",
    [
        ("DRAWN", "drawn"),
        ("DRAWN BY", "drawn"),
        ("CHECKED", "checked"),
        ("CHECK", "checked"),
        ("CHKD", "checked"),
        ("SCALE", "scale"),
        ("SCALES", "scale"),
        ("DATE", "date"),
        ("D4TE", "date"),
        ("DA7E", "date"),
        ("DAT3", "date"),
        ("0ATE", "date"),
        ("DRAWING NUMBER", "number"),
        ("DWG NO", "number"),
        ("DRG NO", "number"),
        ("APPROVED", "approved"),
    ],
)
def test_label_groups(
    text,
    expected,
):
    result = label_groups(
        [tok(text)]
    )

    assert expected in result


def test_label_groups_multiple():
    tokens = [
        tok("DRAWN"),
        tok("DATE"),
        tok("SCALE"),
    ]

    result = label_groups(tokens)

    assert result == {
        "drawn",
        "date",
        "scale",
    }


def test_label_groups_case_insensitive():
    result = label_groups(
        [tok("drawn")]
    )

    assert "drawn" in result


def test_label_groups_strips_punctuation():
    result = label_groups(
        [tok(" DATE: ")]
    )

    assert "date" in result


def test_label_groups_unknown():
    assert label_groups(
        [tok("HELLO")]
    ) == set()


def test_label_groups_empty():
    assert label_groups([]) == set()


# ============================================================
# _inside
# ============================================================

def test_inside():
    a = tok(
        "A",
        bbox=(10, 10, 30, 30),
    )

    b = tok(
        "B",
        bbox=(200, 200, 220, 220),
    )

    result = _inside(
        [a, b],
        (0, 0, 100, 100),
    )

    assert result == [a]


def test_inside_boundary():
    a = tok(
        "A",
        bbox=(0, 0, 20, 20),
    )

    result = _inside(
        [a],
        (10, 10, 100, 100),
    )

    assert a in result


def test_inside_empty():
    assert _inside(
        [],
        (0, 0, 100, 100),
    ) == []


# ============================================================
# _union
# ============================================================

def test_union():
    boxes = [
        (10, 20, 30, 40),
        (5, 25, 50, 60),
        (20, 10, 40, 30),
    ]

    assert _union(boxes) == (
        5,
        10,
        50,
        60,
    )


def test_union_single():
    assert _union([
        (1, 2, 3, 4)
    ]) == (1, 2, 3, 4)


# ============================================================
# metadata_core
# ============================================================

def test_metadata_core_success():
    tokens = [
        tok(
            "DRAWN",
            (700, 750, 760, 770),
        ),
        tok(
            "SCALE",
            (700, 780, 760, 800),
        ),
        tok(
            "CHECKED",
            (700, 810, 780, 830),
        ),
    ]

    result = metadata_core(
        tokens,
        PAGE,
    )

    assert result is not None


def test_metadata_core_not_enough_groups():
    tokens = [
        tok(
            "DRAWN",
            (700, 750, 760, 770),
        )
    ]

    assert metadata_core(
        tokens,
        PAGE,
    ) is None


def test_metadata_core_ignores_top_tokens():
    tokens = [
        tok(
            "DRAWN",
            (700, 100, 760, 120),
        ),
        tok(
            "SCALE",
            (700, 130, 760, 150),
        ),
    ]

    assert metadata_core(
        tokens,
        PAGE,
    ) is None


def test_metadata_core_number_cannot_seed():
    tokens = [
        tok(
            "DRAWING NUMBER",
            (700, 750, 800, 770),
        ),
        tok(
            "DWG NO",
            (700, 780, 800, 800),
        ),
    ]

    assert metadata_core(
        tokens,
        PAGE,
    ) is None


# ============================================================
# core_column
# ============================================================

def test_core_column_none():
    assert core_column(
        None,
        [],
        PAGE,
    ) is None


def test_core_column_no_lines():
    core = (
        700,
        750,
        800,
        850,
    )

    assert core_column(
        core,
        [],
        PAGE,
    ) is None


def test_core_column_success():
    core = (
        700,
        750,
        800,
        850,
    )

    vs = [
        (690, 700, 690, 900),
        (810, 700, 810, 900),
    ]

    result = core_column(
        core,
        vs,
        PAGE,
    )

    assert result is not None

    x0, y0, x1, y1 = result

    assert x0 < x1
    assert y0 < y1


# ============================================================
# column_members
# ============================================================

def test_column_members():
    column = (
        100,
        100,
        300,
        400,
    )

    inside = tok(
        "DRAWN",
        (120, 150, 200, 170),
    )

    outside = tok(
        "DATE",
        (400, 150, 450, 170),
    )

    result = column_members(
        [inside, outside],
        column,
    )

    assert inside in result
    assert outside not in result


def test_column_members_empty():
    assert column_members(
        [],
        (0, 0, 100, 100),
    ) == []


# ============================================================
# header_text
# ============================================================

@pytest.mark.parametrize(
    "text",
    [
        "TITLE",
        "TITLE OF DRAWING",
        "ARCHITECTS",
        "SMITH ARCHITECTS",
        "PTY LTD",
        "PTYLTD",
    ],
)
def test_header_text_true(text):
    assert header_text(text)


@pytest.mark.parametrize(
    "text",
    [
        "DRAWN",
        "DATE",
        "SCALE",
        "HELLO",
    ],
)
def test_header_text_false(text):
    assert not header_text(text)


# ============================================================
# grow_text_up
# ============================================================

def test_grow_text_up_existing_header():
    box = (
        100,
        200,
        400,
        400,
    )

    title = tok(
        "TITLE",
        (150, 210, 250, 230),
    )

    result = grow_text_up(
        box,
        [title],
        PAGE,
    )

    assert result == box


def test_grow_text_up_no_header():
    box = (
        100,
        300,
        400,
        500,
    )

    normal = tok(
        "HELLO",
        (150, 270, 250, 290),
    )

    result = grow_text_up(
        box,
        [normal],
        PAGE,
    )

    assert result == box


def test_grow_text_up_adds_title():
    box = (
        100,
        300,
        400,
        500,
    )

    title = tok(
        "TITLE",
        (150, 260, 250, 285),
    )

    result = grow_text_up(
        box,
        [title],
        PAGE,
    )

    assert result[1] <= box[1]


# ============================================================
# content_coverage
# ============================================================

def test_content_coverage_no_members():
    assert content_coverage(
        (0, 0, 100, 100),
        [],
    ) == 1.0


def test_content_coverage_all_inside():
    members = [
        tok(
            "A",
            (10, 10, 20, 20),
        ),
        tok(
            "B",
            (30, 30, 40, 40),
        ),
    ]

    result = content_coverage(
        (0, 0, 100, 100),
        members,
    )

    assert result == 1.0


def test_content_coverage_partial():
    inside = tok(
        "A",
        (10, 10, 20, 20),
    )

    outside = tok(
        "B",
        (200, 200, 220, 220),
    )

    result = content_coverage(
        (0, 0, 100, 100),
        [inside, outside],
    )

    assert result == pytest.approx(
        0.5
    )


# ============================================================
# merge_segments
# ============================================================

def test_merge_segments_horizontal():
    lines = [
        (0, 10, 50, 10),
        (52, 10, 100, 10),
    ]

    result = merge_segments(
        lines,
        horizontal=True,
        max_gap=5,
    )

    assert len(result) == 1

    assert result[0][0] == 0
    assert result[0][2] == 100


def test_merge_segments_vertical():
    lines = [
        (10, 0, 10, 50),
        (10, 52, 10, 100),
    ]

    result = merge_segments(
        lines,
        horizontal=False,
        max_gap=5,
    )

    assert len(result) == 1


def test_merge_segments_large_gap():
    lines = [
        (0, 10, 50, 10),
        (100, 10, 150, 10),
    ]

    result = merge_segments(
        lines,
        horizontal=True,
        max_gap=5,
    )

    assert len(result) == 2


def test_merge_segments_different_offset():
    lines = [
        (0, 10, 50, 10),
        (52, 50, 100, 50),
    ]

    result = merge_segments(
        lines,
        horizontal=True,
        max_gap=5,
        offset_tol=5,
    )

    assert len(result) == 2


def test_merge_segments_short_line_removed():
    result = merge_segments(
        [
            (0, 0, 0.5, 0),
        ],
        horizontal=True,
        max_gap=5,
    )

    assert result == []


def test_merge_segments_reversed():
    result = merge_segments(
        [
            (100, 10, 0, 10),
        ],
        horizontal=True,
        max_gap=5,
    )

    assert len(result) == 1


def test_merge_segments_empty():
    assert merge_segments(
        [],
        horizontal=True,
        max_gap=5,
    ) == []


# ============================================================
# detect_segments
# ============================================================

def test_detect_segments_blank_image():
    gray = np.full(
        (500, 500),
        255,
        dtype=np.uint8,
    )

    hs, vs, detail = detect_segments(
        gray
    )

    assert isinstance(hs, list)
    assert isinstance(vs, list)

    assert "horizontal_raw" in detail
    assert "vertical_raw" in detail
    assert "horizontal_merged" in detail
    assert "vertical_merged" in detail


def test_detect_segments_mock_hough():
    gray = np.full(
        (500, 500),
        255,
        dtype=np.uint8,
    )

    raw = np.array(
        [
            [[10, 400, 300, 400]],
            [[400, 250, 400, 490]],
            [[10, 400, 100, 450]],
        ],
        dtype=np.int32,
    )

    with patch(
        "uma.table_regions.cv2.HoughLinesP",
        return_value=raw,
    ):
        hs, vs, detail = detect_segments(
            gray
        )

    assert detail[
        "horizontal_raw"
    ] >= 1

    assert detail[
        "vertical_raw"
    ] >= 1


# ============================================================
# text_candidates
# ============================================================

def test_text_candidates_success():
    tokens = [
        tok(
            "DRAWN",
            (600, 750, 680, 780),
        ),
        tok(
            "CHECKED",
            (700, 780, 800, 810),
        ),
        tok(
            "SCALE",
            (820, 810, 900, 840),
        ),
    ]

    result = text_candidates(
        tokens,
        PAGE,
    )

    assert isinstance(result, list)


def test_text_candidates_insufficient_groups():
    tokens = [
        tok(
            "DRAWN",
            (700, 750, 760, 780),
        )
    ]

    assert text_candidates(
        tokens,
        PAGE,
    ) == []


def test_text_candidates_top_ignored():
    tokens = [
        tok(
            "DRAWN",
            (600, 100, 680, 130),
        ),
        tok(
            "CHECKED",
            (700, 120, 800, 150),
        ),
        tok(
            "SCALE",
            (820, 140, 900, 170),
        ),
    ]

    assert text_candidates(
        tokens,
        PAGE,
    ) == []


# ============================================================
# _frame_candidates
# ============================================================

def test_frame_candidates_vertical():
    w = h = 1000

    vs = [
        (700, 700, 700, 950),
        (850, 700, 850, 950),
    ]

    result = _frame_candidates(
        [],
        vs,
        w,
        h,
    )

    assert len(result) >= 1
    assert result[0][-1] == "local_frame"


def test_frame_candidates_horizontal():
    w = h = 1000

    hs = [
        (400, 750, 950, 750),
        (400, 900, 950, 900),
    ]

    result = _frame_candidates(
        hs,
        [],
        w,
        h,
    )

    assert len(result) >= 1


def test_frame_candidates_none():
    assert _frame_candidates(
        [],
        [],
        1000,
        1000,
    ) == []


# ============================================================
# _support
# ============================================================

def test_support_all_edges():
    box = (
        600,
        700,
        900,
        900,
    )

    hs = [
        (600, 700, 900, 700),
        (600, 900, 900, 900),
    ]

    vs = [
        (600, 700, 600, 900),
        (900, 700, 900, 900),
    ]

    result = _support(
        box,
        hs,
        vs,
        1000,
        1000,
    )

    assert "top" in result
    assert "bottom" in result
    assert "left" in result
    assert "right" in result


def test_support_no_lines():
    result = _support(
        (600, 700, 900, 900),
        [],
        [],
        1000,
        1000,
    )

    assert result == {}


def test_support_partial():
    box = (
        600,
        700,
        900,
        900,
    )

    hs = [
        (600, 700, 900, 700),
    ]

    result = _support(
        box,
        hs,
        [],
        1000,
        1000,
    )

    assert "top" in result
    assert "bottom" not in result


# ============================================================
# complete_horizontal_frame
# ============================================================

def test_complete_horizontal_frame_vertical_box():
    box = (
        700,
        600,
        800,
        900,
    )

    result = complete_horizontal_frame(
        box,
        [],
        [],
        PAGE,
    )

    assert result == box


def test_complete_horizontal_frame_no_lines():
    box = (
        300,
        700,
        900,
        900,
    )

    result = complete_horizontal_frame(
        box,
        [],
        [],
        PAGE,
    )

    assert result == box


def test_complete_horizontal_frame_with_lines():
    box = (
        300,
        700,
        800,
        900,
    )

    hs = [
        (280, 700, 820, 700),
    ]

    vs = [
        (280, 680, 280, 920),
        (820, 680, 820, 920),
    ]

    result = complete_horizontal_frame(
        box,
        hs,
        vs,
        PAGE,
    )

    assert len(result) == 4


# ============================================================
# boundary_issues
# ============================================================

def test_boundary_issues_none():
    box = (
        600,
        700,
        900,
        900,
    )

    token = tok(
        "DRAWN",
        (650, 750, 720, 780),
    )

    result = boundary_issues(
        box,
        [token],
        [],
        PAGE,
    )

    assert result == []


def test_boundary_issues_adjacent_metadata():
    box = (
        600,
        700,
        900,
        900,
    )

    token = tok(
        "DRAWN",
        (570, 750, 610, 780),
    )

    result = boundary_issues(
        box,
        [token],
        [],
        PAGE,
    )

    assert len(result) >= 1

    assert any(
        "outside crop" in x
        for x in result
    )


def test_boundary_issues_header_outside():
    box = (
        600,
        700,
        900,
        900,
    )

    token = tok(
        "DESCRIPTION",
        (570, 750, 610, 780),
    )

    result = boundary_issues(
        box,
        [token],
        [],
        PAGE,
    )

    assert len(result) >= 1


# ============================================================
# protect_content
# ============================================================

def test_protect_content_adds_padding():
    box = (
        600,
        700,
        900,
        900,
    )

    result = protect_content(
        box,
        [],
        PAGE,
    )

    assert result[0] < box[0]
    assert result[1] < box[1]
    assert result[2] > box[2]
    assert result[3] > box[3]


def test_protect_content_clamps_page():
    box = (
        0,
        0,
        1000,
        1000,
    )

    result = protect_content(
        box,
        [],
        PAGE,
    )

    assert result == (
        0,
        0,
        1000,
        1000,
    )


def test_protect_content_drawing_number_below():
    box = (
        600,
        700,
        900,
        850,
    )

    drawing_no = tok(
        "A101",
        (700, 850, 800, 875),
    )

    result = protect_content(
        box,
        [drawing_no],
        PAGE,
    )

    assert result[3] >= 875


def test_protect_content_revision_above():
    box = (
        600,
        700,
        900,
        900,
    )

    revision = tok(
        "REVISION",
        (650, 675, 750, 695),
    )

    result = protect_content(
        box,
        [revision],
        PAGE,
    )

    assert result[1] <= 675


def test_protect_content_side_title():
    box = (
        600,
        700,
        900,
        900,
    )

    side = tok(
        "TITLE",
        (585, 740, 595, 850),
    )

    result = protect_content(
        box,
        [side],
        PAGE,
    )

    assert result[0] <= 585


# ============================================================
# find_table - early exits
# ============================================================

def test_find_table_no_tokens():
    image = np.full(
        (1000, 1000),
        255,
        dtype=np.uint8,
    )

    result = find_table(
        image,
        [],
    )

    assert isinstance(
        result,
        TableProposal,
    )

    assert result.bbox is None
    assert result.reliable is False

    assert (
        result.detail["reason"]
        == "Insufficient image or OCR evidence"
    )


def test_find_table_small_image():
    image = np.full(
        (40, 40),
        255,
        dtype=np.uint8,
    )

    result = find_table(
        image,
        [tok("DRAWN")],
    )

    assert result.bbox is None
    assert result.reliable is False


# ============================================================
# find_table - no candidates
# ============================================================

def test_find_table_no_candidates():
    image = np.full(
        (1000, 1000),
        255,
        dtype=np.uint8,
    )

    tokens = [
        tok(
            "HELLO",
            (700, 800, 750, 820),
        )
    ]

    with patch(
        "uma.table_regions.detect_segments",
        return_value=(
            [],
            [],
            {
                "horizontal_raw": 0,
                "vertical_raw": 0,
                "horizontal_merged": 0,
                "vertical_merged": 0,
            },
        ),
    ):
        result = find_table(
            image,
            tokens,
        )

    assert result.bbox is None
    assert result.reliable is False

    assert (
        result.detail[
            "evaluated_candidates"
        ]
        == 0
    )


# ============================================================
# find_table - unreliable candidate
# ============================================================

def test_find_table_candidate_without_support():
    image = np.full(
        (1000, 1000),
        255,
        dtype=np.uint8,
    )

    tokens = [
        tok(
            "DRAWN",
            (650, 760, 700, 780),
        ),
        tok(
            "CHECKED",
            (720, 790, 790, 810),
        ),
        tok(
            "SCALE",
            (810, 820, 870, 840),
        ),
    ]

    candidate = (
        600,
        730,
        900,
        900,
        "text_layout",
    )

    with patch(
        "uma.table_regions.detect_segments",
        return_value=(
            [],
            [],
            {
                "horizontal_raw": 0,
                "vertical_raw": 0,
                "horizontal_merged": 0,
                "vertical_merged": 0,
            },
        ),
    ), patch(
        "uma.table_regions._frame_candidates",
        return_value=[],
    ), patch(
        "uma.table_regions.text_candidates",
        return_value=[candidate],
    ):

        result = find_table(
            image,
            tokens,
        )

    assert result.bbox is None
    assert result.reliable is False


# ============================================================
# find_table - successful candidate
# ============================================================

def test_find_table_success():
    image = np.full(
        (1000, 1000),
        255,
        dtype=np.uint8,
    )

    tokens = [
        tok(
            "DRAWN",
            (650, 760, 700, 780),
        ),
        tok(
            "CHECKED",
            (720, 790, 790, 810),
        ),
        tok(
            "SCALE",
            (810, 820, 870, 840),
        ),
    ]

    candidate = (
        600,
        730,
        900,
        900,
        "local_frame",
    )

    support = {
        "top": 730,
        "bottom": 900,
    }

    with patch(
        "uma.table_regions.detect_segments",
        return_value=(
            [],
            [],
            {
                "horizontal_raw": 0,
                "vertical_raw": 0,
                "horizontal_merged": 0,
                "vertical_merged": 0,
            },
        ),
    ), patch(
        "uma.table_regions._frame_candidates",
        return_value=[candidate],
    ), patch(
        "uma.table_regions.text_candidates",
        return_value=[],
    ), patch(
        "uma.table_regions.metadata_core",
        return_value=None,
    ), patch(
        "uma.table_regions._support",
        return_value=support,
    ), patch(
        "uma.table_regions.boundary_issues",
        return_value=[],
    ), patch(
        "uma.table_regions.grow_text_up",
        side_effect=lambda box, *_: tuple(box),
    ):

        result = find_table(
            image,
            tokens,
        )

    assert result.reliable is True
    assert result.bbox is not None

    assert result.method == "local_frame"

    assert "groups" in result.detail
    assert "supported_edges" in result.detail


# ============================================================
# find_table - boundary rejection
# ============================================================

def test_find_table_boundary_rejected():
    image = np.full(
        (1000, 1000),
        255,
        dtype=np.uint8,
    )

    tokens = [
        tok(
            "DRAWN",
            (650, 760, 700, 780),
        ),
        tok(
            "CHECKED",
            (720, 790, 790, 810),
        ),
        tok(
            "SCALE",
            (810, 820, 870, 840),
        ),
    ]

    candidate = (
        600,
        730,
        900,
        900,
        "local_frame",
    )

    with patch(
        "uma.table_regions.detect_segments",
        return_value=(
            [],
            [],
            {
                "horizontal_raw": 0,
                "vertical_raw": 0,
                "horizontal_merged": 0,
                "vertical_merged": 0,
            },
        ),
    ), patch(
        "uma.table_regions._frame_candidates",
        return_value=[candidate],
    ), patch(
        "uma.table_regions.text_candidates",
        return_value=[],
    ), patch(
        "uma.table_regions.metadata_core",
        return_value=None,
    ), patch(
        "uma.table_regions._support",
        return_value={
            "top": 730,
            "bottom": 900,
        },
    ), patch(
        "uma.table_regions.boundary_issues",
        return_value=[
            "Boundary problem"
        ],
    ), patch(
        "uma.table_regions.grow_text_up",
        side_effect=lambda box, *_: tuple(box),
    ):

        result = find_table(
            image,
            tokens,
        )

    assert result.reliable is False
    assert result.bbox is None

    assert (
        result.detail["reason"]
        == "Candidate boundary is incomplete or ambiguous"
    )

    assert (
        "Boundary problem"
        in result.detail[
            "boundary_issues"
        ]
    )