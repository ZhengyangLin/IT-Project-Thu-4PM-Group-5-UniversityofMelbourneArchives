from types import SimpleNamespace
from unittest.mock import patch

from uma.models import Token, Line
from uma.views import (
    assign_ids,
    build_grid,
    build_index,
    build_views,
)


# ============================================================
# Helpers
# ============================================================

def make_token(
    text,
    bbox=(0, 0, 40, 20),
    conf=0.9,
    candidates=None,
):
    return Token(
        text=text,
        bbox=bbox,
        conf=conf,
        candidates=[] if candidates is None else candidates,
    )


def make_line(tokens, row_no=0):
    return Line(
        tokens=tokens,
        row_no=row_no,
    )


def make_cfg(
    grid_max_blank_rows=3,
    grid_max_cols=200,
    show_row_no=True,
):
    return SimpleNamespace(
        grid_max_blank_rows=grid_max_blank_rows,
        grid_max_cols=grid_max_cols,
        show_row_no=show_row_no,
    )


def make_candidate(
    text,
    engine="test",
    conf=0.8,
):
    return SimpleNamespace(
        text=text,
        engine=engine,
        conf=conf,
    )


# ============================================================
# assign_ids
# ============================================================

def test_assign_ids_empty():
    assert assign_ids([]) == []


def test_assign_ids_single_line():
    t1 = make_token(
        "HELLO",
        (0, 0, 40, 20),
    )

    t2 = make_token(
        "WORLD",
        (50, 0, 100, 20),
    )

    line = make_line(
        [t1, t2]
    )

    result = assign_ids(
        [line]
    )

    assert result == [t1, t2]

    assert line.row_no == 1

    assert t1.tid == 0
    assert t2.tid == 1


def test_assign_ids_sorts_tokens_left_to_right():
    right = make_token(
        "RIGHT",
        (100, 0, 150, 20),
    )

    left = make_token(
        "LEFT",
        (10, 0, 50, 20),
    )

    line = make_line(
        [right, left]
    )

    result = assign_ids(
        [line]
    )

    assert result == [
        left,
        right,
    ]

    assert left.tid == 0
    assert right.tid == 1


def test_assign_ids_sorts_lines_top_to_bottom():
    bottom_token = make_token(
        "BOTTOM",
        (0, 100, 60, 120),
    )

    top_token = make_token(
        "TOP",
        (0, 10, 40, 30),
    )

    bottom = make_line(
        [bottom_token]
    )

    top = make_line(
        [top_token]
    )

    result = assign_ids(
        [bottom, top]
    )

    assert result == [
        top_token,
        bottom_token,
    ]

    assert top.row_no == 1
    assert bottom.row_no == 2

    assert top_token.tid == 0
    assert bottom_token.tid == 1


def test_assign_ids_same_y_sorts_lines_by_x():
    right_token = make_token(
        "RIGHT",
        (100, 10, 150, 30),
    )

    left_token = make_token(
        "LEFT",
        (10, 10, 50, 30),
    )

    right_line = make_line(
        [right_token]
    )

    left_line = make_line(
        [left_token]
    )

    result = assign_ids(
        [
            right_line,
            left_line,
        ]
    )

    assert result == [
        left_token,
        right_token,
    ]

    assert left_line.row_no == 1
    assert right_line.row_no == 2


def test_assign_ids_multiple_lines_and_tokens():
    t1 = make_token(
        "A",
        (100, 100, 120, 120),
    )

    t2 = make_token(
        "B",
        (10, 100, 30, 120),
    )

    t3 = make_token(
        "C",
        (20, 10, 40, 30),
    )

    line_bottom = make_line(
        [t1, t2]
    )

    line_top = make_line(
        [t3]
    )

    result = assign_ids(
        [
            line_bottom,
            line_top,
        ]
    )

    assert result == [
        t3,
        t2,
        t1,
    ]

    assert [
        t.tid for t in result
    ] == [0, 1, 2]

    assert line_top.row_no == 1
    assert line_bottom.row_no == 2


# ============================================================
# build_grid - empty/basic
# ============================================================

def test_build_grid_empty():
    cfg = make_cfg()

    assert build_grid(
        [],
        cfg,
    ) == ""


def test_build_grid_single_token():
    token = make_token(
        "HELLO",
        (0, 0, 50, 20),
    )

    line = make_line(
        [token]
    )

    result = build_grid(
        [line],
        make_cfg(),
    )

    assert result == "HELLO"


def test_build_grid_two_adjacent_tokens():
    t1 = make_token(
        "HELLO",
        (0, 0, 50, 20),
    )

    t2 = make_token(
        "WORLD",
        (50, 0, 100, 20),
    )

    line = make_line(
        [t1, t2]
    )

    result = build_grid(
        [line],
        make_cfg(),
    )

    assert "HELLO" in result
    assert "WORLD" in result

    assert result.index(
        "HELLO"
    ) < result.index(
        "WORLD"
    )


def test_build_grid_sorts_tokens():
    right = make_token(
        "RIGHT",
        (100, 0, 150, 20),
    )

    left = make_token(
        "LEFT",
        (0, 0, 40, 20),
    )

    line = make_line(
        [right, left]
    )

    result = build_grid(
        [line],
        make_cfg(),
    )

    assert result.index(
        "LEFT"
    ) < result.index(
        "RIGHT"
    )


def test_build_grid_sorts_lines():
    bottom = make_line([
        make_token(
            "BOTTOM",
            (0, 100, 60, 120),
        )
    ])

    top = make_line([
        make_token(
            "TOP",
            (0, 0, 40, 20),
        )
    ])

    result = build_grid(
        [bottom, top],
        make_cfg(),
    )

    assert result.index(
        "TOP"
    ) < result.index(
        "BOTTOM"
    )


# ============================================================
# build_grid - character width
# ============================================================

def test_build_grid_empty_text_uses_minimum_length_one():
    token = make_token(
        "",
        (0, 0, 20, 20),
    )

    line = make_line(
        [token]
    )

    result = build_grid(
        [line],
        make_cfg(),
    )

    assert result == ""


def test_build_grid_ignores_extreme_width_ratio():
    extreme = make_token(
        "X",
        (0, 0, 2000, 20),
    )

    normal = make_token(
        "NORMAL",
        (2100, 0, 2160, 20),
    )

    line = make_line(
        [
            extreme,
            normal,
        ]
    )

    result = build_grid(
        [line],
        make_cfg(
            grid_max_cols=500,
        ),
    )

    assert "X" in result
    assert "NORMAL" in result


def test_build_grid_ignores_zero_height_for_width_estimate():
    token = make_token(
        "ZERO",
        (0, 0, 40, 0),
    )

    line = make_line(
        [token]
    )

    result = build_grid(
        [line],
        make_cfg(),
    )

    assert result == "ZERO"


def test_build_grid_no_valid_widths_uses_default():
    t1 = make_token(
        "A",
        (0, 0, 1000, 0),
    )

    t2 = make_token(
        "B",
        (80, 0, 1080, 0),
    )

    line = make_line(
        [t1, t2]
    )

    result = build_grid(
        [line],
        make_cfg(),
    )

    assert "A" in result
    assert "B" in result


def test_build_grid_unit_never_below_one():
    token1 = make_token(
        "AAAAAAAAAA",
        (0, 0, 1, 10),
    )

    token2 = make_token(
        "B",
        (10, 0, 11, 10),
    )

    line = make_line(
        [token1, token2]
    )

    result = build_grid(
        [line],
        make_cfg(),
    )

    assert "AAAAAAAAAA" in result
    assert "B" in result


# ============================================================
# build_grid - horizontal positioning
# ============================================================

def test_build_grid_preserves_horizontal_gap():
    left = make_token(
        "A",
        (0, 0, 10, 10),
    )

    right = make_token(
        "B",
        (100, 0, 110, 10),
    )

    line = make_line(
        [left, right]
    )

    result = build_grid(
        [line],
        make_cfg(),
    )

    assert result.startswith("A")
    assert "B" in result

    # There should be spaces between them.
    assert "A B" in result or "A  " in result


def test_build_grid_adds_separator_when_tokens_overlap():
    t1 = make_token(
        "HELLO",
        (0, 0, 100, 20),
    )

    t2 = make_token(
        "WORLD",
        (10, 0, 110, 20),
    )

    line = make_line(
        [t1, t2]
    )

    result = build_grid(
        [line],
        make_cfg(),
    )

    # If the calculated column is already
    # occupied, views.py explicitly adds
    # a separating space.
    assert "HELLO WORLD" in result


# ============================================================
# build_grid - vertical blank rows
# ============================================================

def test_build_grid_no_blank_row_for_close_lines():
    line1 = make_line([
        make_token(
            "FIRST",
            (0, 0, 50, 20),
        )
    ])

    line2 = make_line([
        make_token(
            "SECOND",
            (0, 20, 60, 40),
        )
    ])

    result = build_grid(
        [line1, line2],
        make_cfg(),
    )

    assert result.splitlines() == [
        "FIRST",
        "SECOND",
    ]


def test_build_grid_inserts_blank_rows():
    line1 = make_line([
        make_token(
            "FIRST",
            (0, 0, 50, 20),
        )
    ])

    line2 = make_line([
        make_token(
            "SECOND",
            (0, 100, 60, 120),
        )
    ])

    result = build_grid(
        [line1, line2],
        make_cfg(
            grid_max_blank_rows=10,
        ),
    )

    rows = result.split("\n")

    assert rows[0] == "FIRST"
    assert rows[-1] == "SECOND"

    assert "" in rows[1:-1]


def test_build_grid_blank_rows_are_capped():
    line1 = make_line([
        make_token(
            "FIRST",
            (0, 0, 50, 20),
        )
    ])

    line2 = make_line([
        make_token(
            "SECOND",
            (0, 1000, 60, 1020),
        )
    ])

    result = build_grid(
        [line1, line2],
        make_cfg(
            grid_max_blank_rows=2,
        ),
    )

    rows = result.split("\n")

    assert rows == [
        "FIRST",
        "",
        "",
        "SECOND",
    ]


def test_build_grid_negative_vertical_gap_adds_no_blank():
    line1 = make_line([
        make_token(
            "FIRST",
            (0, 0, 50, 100),
        )
    ])

    line2 = make_line([
        make_token(
            "SECOND",
            (0, 50, 60, 70),
        )
    ])

    result = build_grid(
        [line1, line2],
        make_cfg(),
    )

    assert result.splitlines() == [
        "FIRST",
        "SECOND",
    ]


# ============================================================
# build_grid - max cols
# ============================================================

def test_build_grid_max_cols():
    token = make_token(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        (0, 0, 260, 20),
    )

    line = make_line(
        [token]
    )

    result = build_grid(
        [line],
        make_cfg(
            grid_max_cols=10,
        ),
    )

    assert result == "ABCDEFGHIJ"
    assert len(result) == 10


def test_build_grid_rstrip():
    left = make_token(
        "A",
        (0, 0, 10, 10),
    )

    line = make_line(
        [left]
    )

    result = build_grid(
        [line],
        make_cfg(),
    )

    assert not result.endswith(" ")


# ============================================================
# build_index - basic
# ============================================================

def test_build_index_empty():
    result = build_index(
        [],
        make_cfg(),
    )

    assert result == ""


def test_build_index_basic_with_row_number():
    token = make_token(
        "HELLO"
    )

    token.tid = 5

    line = make_line(
        [token],
        row_no=2,
    )

    result = build_index(
        [line],
        make_cfg(
            show_row_no=True,
        ),
    )

    assert "[5]" in result
    assert "row 2" in result
    assert "HELLO" in result


def test_build_index_without_row_number():
    token = make_token(
        "HELLO"
    )

    token.tid = 5

    line = make_line(
        [token],
        row_no=2,
    )

    result = build_index(
        [line],
        make_cfg(
            show_row_no=False,
        ),
    )

    assert "[5]" in result
    assert "HELLO" in result

    assert "row 2" not in result


def test_build_index_sorts_lines():
    top_token = make_token(
        "TOP",
        (0, 0, 40, 20),
    )

    bottom_token = make_token(
        "BOTTOM",
        (0, 100, 60, 120),
    )

    top_token.tid = 0
    bottom_token.tid = 1

    top = make_line(
        [top_token],
        row_no=1,
    )

    bottom = make_line(
        [bottom_token],
        row_no=2,
    )

    result = build_index(
        [bottom, top],
        make_cfg(),
    )

    assert result.index(
        "TOP"
    ) < result.index(
        "BOTTOM"
    )


def test_build_index_sorts_tokens():
    left = make_token(
        "LEFT",
        (0, 0, 40, 20),
    )

    right = make_token(
        "RIGHT",
        (100, 0, 150, 20),
    )

    left.tid = 0
    right.tid = 1

    line = make_line(
        [right, left],
        row_no=1,
    )

    result = build_index(
        [line],
        make_cfg(),
    )

    assert result.index(
        "LEFT"
    ) < result.index(
        "RIGHT"
    )


# ============================================================
# build_index - alternatives
# ============================================================

def test_build_index_with_alternative():
    candidate = make_candidate(
        text="HELLO!",
        engine="easyocr",
    )

    token = make_token(
        "HELLO",
        candidates=[
            candidate,
        ],
    )

    token.tid = 3

    line = make_line(
        [token],
        row_no=1,
    )

    # Make the alternative explicit so this
    # test does not depend on Candidate
    # filtering rules in Token.alternatives().
    with patch.object(
        token,
        "alternatives",
        return_value=[candidate],
    ):
        result = build_index(
            [line],
            make_cfg(),
        )

    assert "← alternatives" in result
    assert "easyocr:HELLO!" in result


def test_build_index_multiple_alternatives():
    c1 = make_candidate(
        "ABC",
        "engine1",
    )

    c2 = make_candidate(
        "ABD",
        "engine2",
    )

    token = make_token(
        "AB",
    )

    token.tid = 4

    line = make_line(
        [token],
        row_no=1,
    )

    with patch.object(
        token,
        "alternatives",
        return_value=[
            c1,
            c2,
        ],
    ):
        result = build_index(
            [line],
            make_cfg(),
        )

    assert (
        "engine1:ABC | engine2:ABD"
        in result
    )


def test_build_index_without_alternatives():
    token = make_token(
        "HELLO"
    )

    token.tid = 1

    line = make_line(
        [token],
        row_no=1,
    )

    with patch.object(
        token,
        "alternatives",
        return_value=[],
    ):
        result = build_index(
            [line],
            make_cfg(),
        )

    assert "alternatives" not in result


# ============================================================
# build_index - hints
# ============================================================

def test_build_index_with_hint():
    token = make_token(
        "AB"
    )

    token.tid = 7

    line = make_line(
        [token],
        row_no=3,
    )

    hints = {
        7: {
            "preferred": "ABC",
            "reasons": [
                "more complete",
                "better OCR",
            ],
        }
    }

    result = build_index(
        [line],
        make_cfg(),
        hints,
    )

    assert "⚠ prefer 'ABC'" in result

    assert (
        "more complete; better OCR"
        in result
    )


def test_build_index_hint_without_row_number():
    token = make_token(
        "AB"
    )

    token.tid = 2

    line = make_line(
        [token],
        row_no=5,
    )

    hints = {
        2: {
            "preferred": "ABC",
            "reasons": [
                "candidate is better"
            ],
        }
    }

    result = build_index(
        [line],
        make_cfg(
            show_row_no=False,
        ),
        hints,
    )

    assert "[2]" in result
    assert "row 5" not in result
    assert "⚠ prefer 'ABC'" in result


def test_build_index_hint_for_different_token_ignored():
    token = make_token(
        "HELLO"
    )

    token.tid = 1

    line = make_line(
        [token],
        row_no=1,
    )

    hints = {
        999: {
            "preferred": "OTHER",
            "reasons": [
                "not relevant"
            ],
        }
    }

    result = build_index(
        [line],
        make_cfg(),
        hints,
    )

    assert "⚠ prefer" not in result


def test_build_index_none_hints():
    token = make_token(
        "HELLO"
    )

    token.tid = 0

    line = make_line(
        [token],
        row_no=1,
    )

    result = build_index(
        [line],
        make_cfg(),
        None,
    )

    assert "HELLO" in result


# ============================================================
# build_views - empty
# ============================================================

def test_build_views_empty():
    cfg = make_cfg()

    grid, index, flat = build_views(
        [],
        cfg,
    )

    assert grid == ""
    assert index == ""
    assert flat == []


# ============================================================
# build_views - supplied hints
# ============================================================

def test_build_views_with_supplied_hints():
    token = make_token(
        "AB"
    )

    line = make_line(
        [token]
    )

    hints = {
        0: {
            "preferred": "ABC",
            "reasons": [
                "manual hint"
            ],
        }
    }

    with patch(
        "uma.views.scan_candidates"
    ) as mock_scan:
        grid, index, flat = build_views(
            [line],
            make_cfg(),
            hints,
        )

    # scan_candidates is only automatically
    # used when hints is None.
    mock_scan.assert_not_called()

    assert grid == "AB"
    assert flat == [token]

    assert token.tid == 0
    assert line.row_no == 1

    assert "⚠ prefer 'ABC'" in index
    assert "manual hint" in index


def test_build_views_empty_dict_does_not_auto_scan():
    token = make_token(
        "HELLO"
    )

    line = make_line(
        [token]
    )

    with patch(
        "uma.views.scan_candidates"
    ) as mock_scan:
        build_views(
            [line],
            make_cfg(),
            {},
        )

    # Important: {} is different from None.
    assert mock_scan.call_count == 0


# ============================================================
# build_views - automatic scan_candidates
# ============================================================

def test_build_views_auto_scan_no_better_candidate():
    token = make_token(
        "HELLO"
    )

    line = make_line(
        [token]
    )

    with patch(
        "uma.views.scan_candidates",
        return_value=(None, ""),
    ) as mock_scan:

        grid, index, flat = build_views(
            [line],
            make_cfg(),
            hints=None,
        )

    mock_scan.assert_called_once_with(
        token
    )

    assert flat == [token]
    assert grid == "HELLO"

    assert "⚠ prefer" not in index


def test_build_views_auto_scan_with_better_candidate():
    token = make_token(
        "AB"
    )

    line = make_line(
        [token]
    )

    better = make_candidate(
        text="ABC",
        engine="easyocr",
    )

    with patch(
        "uma.views.scan_candidates",
        return_value=(
            better,
            "C",
        ),
    ) as mock_scan:

        grid, index, flat = build_views(
            [line],
            make_cfg(),
            hints=None,
        )

    mock_scan.assert_called_once_with(
        token
    )

    assert grid == "AB"
    assert flat == [token]

    assert "⚠ prefer 'ABC'" in index
    assert "alternative easyocr" in index
    assert "adds 'C'" in index

    assert (
        "detector missed narrow characters"
        in index
    )


def test_build_views_auto_scan_multiple_tokens():
    t1 = make_token(
        "AB",
        (0, 0, 20, 20),
    )

    t2 = make_token(
        "HELLO",
        (50, 0, 100, 20),
    )

    line = make_line(
        [t1, t2]
    )

    better = make_candidate(
        "ABC",
        "engine2",
    )

    def fake_scan(token):
        if token is t1:
            return better, "C"

        return None, ""

    with patch(
        "uma.views.scan_candidates",
        side_effect=fake_scan,
    ) as mock_scan:

        grid, index, flat = build_views(
            [line],
            make_cfg(),
            None,
        )

    assert mock_scan.call_count == 2

    assert flat == [
        t1,
        t2,
    ]

    assert t1.tid == 0
    assert t2.tid == 1

    assert "ABC" in index
    assert "HELLO" in index


# ============================================================
# build_views - integration of local view functions
# ============================================================

def test_build_views_assigns_rows_and_ids():
    bottom_token = make_token(
        "BOTTOM",
        (0, 100, 60, 120),
    )

    top_right = make_token(
        "RIGHT",
        (100, 0, 150, 20),
    )

    top_left = make_token(
        "LEFT",
        (0, 0, 40, 20),
    )

    bottom = make_line([
        bottom_token
    ])

    top = make_line([
        top_right,
        top_left,
    ])

    with patch(
        "uma.views.scan_candidates",
        return_value=(None, ""),
    ):
        grid, index, flat = build_views(
            [
                bottom,
                top,
            ],
            make_cfg(),
        )

    assert flat == [
        top_left,
        top_right,
        bottom_token,
    ]

    assert top.row_no == 1
    assert bottom.row_no == 2

    assert top_left.tid == 0
    assert top_right.tid == 1
    assert bottom_token.tid == 2

    assert "LEFT" in grid
    assert "RIGHT" in grid
    assert "BOTTOM" in grid

    assert "[0]" in index
    assert "[1]" in index
    assert "[2]" in index


# ============================================================
# Direct delegation test
# ============================================================

def test_build_views_calls_grid_and_index():
    token = make_token(
        "TEST"
    )

    line = make_line(
        [token]
    )

    cfg = make_cfg()

    hints = {}

    with (
        patch(
            "uma.views.build_grid",
            return_value="GRID",
        ) as mock_grid,

        patch(
            "uma.views.build_index",
            return_value="INDEX",
        ) as mock_index,
    ):
        grid, index, flat = build_views(
            [line],
            cfg,
            hints,
        )

    assert grid == "GRID"
    assert index == "INDEX"

    assert flat == [token]

    mock_grid.assert_called_once_with(
        [line],
        cfg,
    )

    mock_index.assert_called_once_with(
        [line],
        cfg,
        hints,
    )