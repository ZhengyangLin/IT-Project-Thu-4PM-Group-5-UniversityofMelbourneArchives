from types import SimpleNamespace
from unittest.mock import (
    Mock,
    patch,
)

import numpy as np
import pytest

from uma.models import (
    Candidate,
    Token,
)
from uma.ocr.engines import RawToken
from uma.ocr.fusion import (
    _norm,
    align,
    fuse,
    read_all,
    _same_text,
    _contains,
    merge_passes,
)


# ============================================================
# Helpers
# ============================================================

def raw(
    text="ABC",
    bbox=(0, 0, 100, 20),
    conf=0.9,
    engine="a",
):
    return RawToken(
        text=text,
        bbox=bbox,
        conf=conf,
        engine=engine,
    )


def token(
    text="ABC",
    bbox=(0, 0, 100, 20),
    conf=0.9,
    engine="fine",
    agreement=1.0,
    with_candidate=True,
):
    candidates = []

    if with_candidate:
        candidates = [
            Candidate(
                text,
                engine,
                conf,
            )
        ]

    return Token(
        text=text,
        bbox=bbox,
        conf=conf,
        candidates=candidates,
        agreement=agreement,
    )


# ============================================================
# _norm
# ============================================================

def test_norm_uppercase():
    assert _norm(
        "hello"
    ) == "HELLO"


def test_norm_whitespace():
    assert _norm(
        "  hello    world  "
    ) == "HELLO WORLD"


def test_norm_empty():
    assert _norm("") == ""


# ============================================================
# align
# ============================================================

def test_align_empty():
    assert align(
        [],
        0.5,
    ) == []


def test_align_single():
    r = raw()

    assert align(
        [r],
        0.5,
    ) == [[r]]


def test_align_non_overlapping():
    r1 = raw(
        "A",
        (0, 0, 20, 20),
        engine="one",
    )

    r2 = raw(
        "B",
        (100, 100, 120, 120),
        engine="two",
    )

    result = align(
        [r1, r2],
        0.5,
    )

    assert len(result) == 2


def test_align_overlapping_different_engines():
    r1 = raw(
        "A",
        (0, 0, 100, 20),
        engine="one",
    )

    r2 = raw(
        "A",
        (0, 0, 100, 20),
        engine="two",
    )

    result = align(
        [r1, r2],
        0.5,
    )

    assert len(result) == 1
    assert result[0] == [
        r1,
        r2,
    ]


def test_align_same_engine_duplicate_dropped():
    r1 = raw(
        "A",
        (0, 0, 100, 20),
        engine="same",
    )

    r2 = raw(
        "B",
        (0, 0, 100, 20),
        engine="same",
    )

    result = align(
        [r1, r2],
        0.5,
    )

    assert len(result) == 1
    assert len(result[0]) == 1


def test_align_sorts_by_y_then_x():
    bottom = raw(
        "BOTTOM",
        (0, 100, 20, 120),
        engine="one",
    )

    right = raw(
        "RIGHT",
        (100, 0, 120, 20),
        engine="two",
    )

    left = raw(
        "LEFT",
        (0, 0, 20, 20),
        engine="three",
    )

    result = align(
        [
            bottom,
            right,
            left,
        ],
        0.5,
    )

    assert result[0][0] is left
    assert result[1][0] is right
    assert result[2][0] is bottom


# ============================================================
# fuse
# ============================================================

def test_fuse_empty():
    assert fuse(
        [],
        iou_thresh=0.5,
        min_conf=0.5,
        n_engines=1,
    ) == []


def test_fuse_filters_low_confidence():
    result = fuse(
        [
            raw(
                conf=0.2
            )
        ],
        iou_thresh=0.5,
        min_conf=0.5,
        n_engines=1,
    )

    assert result == []


def test_fuse_single():
    result = fuse(
        [
            raw(
                text=" ABC ",
                bbox=(1, 2, 30, 40),
                conf=0.9,
            )
        ],
        iou_thresh=0.5,
        min_conf=0.5,
        n_engines=1,
    )

    assert len(result) == 1

    t = result[0]

    assert t.text == "ABC"
    assert t.bbox == (
        1,
        2,
        30,
        40,
    )
    assert t.conf == 0.9
    assert t.agreement == 1.0


def test_fuse_selects_highest_confidence():
    result = fuse(
        [
            raw(
                "LOW",
                conf=0.6,
                engine="one",
            ),
            raw(
                "HIGH",
                conf=0.95,
                engine="two",
            ),
        ],
        iou_thresh=0.5,
        min_conf=0.5,
        n_engines=2,
    )

    assert len(result) == 1
    assert result[0].text == "HIGH"
    assert result[0].conf == 0.95


def test_fuse_same_text_normalized_agreement():
    result = fuse(
        [
            raw(
                "hello world",
                conf=0.8,
                engine="one",
            ),
            raw(
                " HELLO   WORLD ",
                conf=0.9,
                engine="two",
            ),
        ],
        iou_thresh=0.5,
        min_conf=0.5,
        n_engines=2,
    )

    assert result[0].agreement == 1.0


def test_fuse_disagreement():
    result = fuse(
        [
            raw(
                "ABC",
                conf=0.9,
                engine="one",
            ),
            raw(
                "XYZ",
                conf=0.8,
                engine="two",
            ),
        ],
        iou_thresh=0.5,
        min_conf=0.5,
        n_engines=2,
    )

    assert result[0].agreement == 0.5


def test_fuse_penalizes_missing_engine():
    result = fuse(
        [
            raw(
                "ABC",
                conf=0.9,
                engine="one",
            )
        ],
        iou_thresh=0.5,
        min_conf=0.5,
        n_engines=2,
    )

    assert result[0].agreement == 0.5


def test_fuse_union_bbox():
    result = fuse(
        [
            raw(
                "ABC",
                (10, 20, 80, 40),
                0.9,
                "one",
            ),
            raw(
                "ABC",
                (5, 15, 100, 50),
                0.8,
                "two",
            ),
        ],
        iou_thresh=0.2,
        min_conf=0.5,
        n_engines=2,
    )

    assert result[0].bbox == (
        5,
        15,
        100,
        50,
    )


def test_fuse_output_sorted():
    bottom = raw(
        "BOTTOM",
        (0, 100, 50, 120),
        0.9,
        "one",
    )

    top = raw(
        "TOP",
        (0, 0, 50, 20),
        0.9,
        "one",
    )

    result = fuse(
        [
            bottom,
            top,
        ],
        0.5,
        0.5,
        1,
    )

    assert [
        t.text for t in result
    ] == [
        "TOP",
        "BOTTOM",
    ]


# ============================================================
# read_all
# ============================================================

def test_read_all_success():
    gray = np.zeros(
        (100, 200),
        dtype=np.uint8,
    )

    e1 = Mock()
    e1.name = "one"
    e1.read.return_value = [
        raw(
            "ABC",
            engine="one",
        )
    ]

    e2 = Mock()
    e2.name = "two"
    e2.read.return_value = [
        raw(
            "ABC",
            engine="two",
        )
    ]

    tokens, diag = read_all(
        gray,
        [e1, e2],
        upscale=2.0,
        iou_thresh=0.5,
        min_conf=0.5,
    )

    assert len(tokens) == 1

    assert diag[
        "engines_used"
    ] == [
        "one",
        "two",
    ]

    assert diag[
        "engines_configured"
    ] == [
        "one",
        "two",
    ]

    assert diag[
        "tokens_per_engine"
    ] == {
        "one": 1,
        "two": 1,
    }

    assert diag[
        "token_count"
    ] == 1

    assert diag[
        "disagreements"
    ] == 0

    e1.read.assert_called_once_with(
        gray,
        upscale=2.0,
    )


def test_read_all_engine_crash_skipped():
    gray = np.zeros(
        (20, 20),
        dtype=np.uint8,
    )

    broken = Mock()
    broken.name = "broken"
    broken.read.side_effect = (
        RuntimeError("boom")
    )

    good = Mock()
    good.name = "good"
    good.read.return_value = [
        raw(
            "ABC",
            engine="good",
        )
    ]

    tokens, diag = read_all(
        gray,
        [
            broken,
            good,
        ],
        1.0,
        0.5,
        0.5,
    )

    assert len(tokens) == 1

    assert diag[
        "engines_used"
    ] == [
        "good"
    ]

    assert (
        "broken"
        not in diag[
            "tokens_per_engine"
        ]
    )


def test_read_all_all_engines_fail():
    gray = np.zeros(
        (20, 20),
        dtype=np.uint8,
    )

    engine = Mock()
    engine.name = "broken"
    engine.read.side_effect = (
        RuntimeError("boom")
    )

    tokens, diag = read_all(
        gray,
        [engine],
        1.0,
        0.5,
        0.5,
    )

    assert tokens == []

    assert diag[
        "engines_used"
    ] == []

    assert diag[
        "token_count"
    ] == 0


def test_read_all_passes_extra_kwargs():
    gray = np.zeros(
        (20, 20),
        dtype=np.uint8,
    )

    engine = Mock()
    engine.name = "test"
    engine.read.return_value = []

    read_all(
        gray,
        [engine],
        upscale=1.5,
        iou_thresh=0.5,
        min_conf=0.5,
        max_side=3000,
        test_value=123,
    )

    engine.read.assert_called_once_with(
        gray,
        upscale=1.5,
        max_side=3000,
        test_value=123,
    )


def test_read_all_disagreement_count():
    gray = np.zeros(
        (20, 20),
        dtype=np.uint8,
    )

    e1 = Mock()
    e1.name = "one"
    e1.read.return_value = [
        raw(
            "ABC",
            engine="one",
        )
    ]

    e2 = Mock()
    e2.name = "two"
    e2.read.return_value = [
        raw(
            "XYZ",
            engine="two",
        )
    ]

    _, diag = read_all(
        gray,
        [e1, e2],
        1.0,
        0.5,
        0.5,
    )

    assert diag[
        "disagreements"
    ] == 1


# ============================================================
# _same_text
# ============================================================

def test_same_text_empty():
    assert (
        _same_text(
            "",
            "ABC",
        )
        is False
    )


def test_same_text_substring():
    assert (
        _same_text(
            "ABC",
            "ABC DEF",
        )
        is True
    )


def test_same_text_case_and_spaces():
    assert (
        _same_text(
            "hello world",
            " HELLO   WORLD ",
        )
        is True
    )


def test_same_text_fuzzy_true():
    with patch(
        "rapidfuzz.fuzz.ratio",
        return_value=70,
    ):
        assert (
            _same_text(
                "ABC",
                "XYZ",
            )
            is True
        )


def test_same_text_fuzzy_false():
    with patch(
        "rapidfuzz.fuzz.ratio",
        return_value=69,
    ):
        assert (
            _same_text(
                "ABC",
                "XYZ",
            )
            is False
        )


# ============================================================
# _contains
# ============================================================

def test_contains_full():
    assert _contains(
        (0, 0, 100, 100),
        (20, 20, 40, 40),
    )


def test_contains_no_overlap():
    assert not _contains(
        (0, 0, 10, 10),
        (20, 20, 30, 30),
    )


def test_contains_partial_above_threshold():
    assert _contains(
        (0, 0, 80, 100),
        (0, 0, 100, 100),
        ratio=0.7,
    )


def test_contains_partial_below_threshold():
    assert not _contains(
        (0, 0, 50, 100),
        (0, 0, 100, 100),
        ratio=0.7,
    )


def test_contains_zero_area_inner():
    assert not _contains(
        (0, 0, 100, 100),
        (200, 200, 200, 200),
    )


# ============================================================
# merge_passes
# ============================================================

def test_merge_passes_empty():
    result, diag = merge_passes(
        [],
        [],
    )

    assert result == []

    assert diag == {
        "fine_only": 0,
        "conflicts": 0,
        "recovered_from_coarse": 0,
        "merged_total": 0,
    }


def test_merge_passes_matching_text():
    fine = [
        token(
            "ABC",
            (0, 0, 100, 20),
        )
    ]

    coarse = [
        token(
            "ABC",
            (0, 0, 100, 20),
            engine="coarse",
        )
    ]

    result, diag = merge_passes(
        fine,
        coarse,
    )

    assert result == fine

    assert diag[
        "conflicts"
    ] == 0

    assert diag[
        "recovered_from_coarse"
    ] == 0


def test_merge_passes_conflicting_text_high_overlap():
    ft = token(
        "ABC",
        (0, 0, 100, 20),
        engine="fine",
    )

    ct = token(
        "XYZ",
        (0, 0, 100, 20),
        conf=0.8,
        engine="coarse",
    )

    result, diag = merge_passes(
        [ft],
        [ct],
    )

    assert result == [ft]

    assert diag[
        "conflicts"
    ] == 1

    assert len(
        ft.candidates
    ) == 2

    assert (
        ft.candidates[-1].text
        == "XYZ"
    )

    assert (
        ft.candidates[-1].engine
        == "coarse@coarse"
    )


def test_merge_passes_conflict_without_coarse_candidate():
    ft = token(
        "ABC",
        (0, 0, 100, 20),
    )

    ct = token(
        "XYZ",
        (0, 0, 100, 20),
        with_candidate=False,
    )

    result, diag = merge_passes(
        [ft],
        [ct],
    )

    assert diag[
        "conflicts"
    ] == 1

    assert (
        ft.candidates[-1].engine
        == "ocr@coarse"
    )


def test_merge_passes_low_overlap_not_merged():
    ft = token(
        "ABC",
        (0, 0, 100, 20),
    )

    ct = token(
        "ABC",
        (200, 200, 300, 220),
        engine="coarse",
    )

    result, diag = merge_passes(
        [ft],
        [ct],
    )

    assert len(result) == 2

    assert diag[
        "recovered_from_coarse"
    ] == 1


def test_merge_passes_different_text_medium_overlap_skipped():
    ft = token(
        "ABC",
        (0, 0, 100, 100),
    )

    ct = token(
        "XYZ",
        (10, 0, 110, 100),
        engine="coarse",
    )

    with (
        patch(
            "uma.ocr.fusion.io_min",
            return_value=0.7,
        ),
        patch(
            "uma.ocr.fusion._same_text",
            return_value=False,
        ),
    ):
        result, diag = merge_passes(
            [ft],
            [ct],
        )

    assert diag[
        "conflicts"
    ] == 0


def test_merge_passes_contained_coarse_removed():
    ft = token(
        "ABC",
        (20, 20, 40, 40),
    )

    ct = token(
        "LARGE",
        (0, 0, 100, 100),
        engine="coarse",
    )

    with patch(
        "uma.ocr.fusion.io_min",
        return_value=0.0,
    ):
        result, diag = merge_passes(
            [ft],
            [ct],
        )

    assert result == [ft]

    assert diag[
        "recovered_from_coarse"
    ] == 0


def test_merge_passes_recovers_coarse():
    ct = token(
        "RECOVERED",
        (100, 100, 200, 120),
        agreement=0.9,
        engine="coarse",
    )

    result, diag = merge_passes(
        [],
        [ct],
    )

    assert result == [ct]

    assert ct.agreement == 0.5

    assert (
        ct.candidates[0].engine
        == "coarse@coarse"
    )

    assert diag[
        "recovered_from_coarse"
    ] == 1


def test_merge_passes_recovered_without_candidates():
    ct = token(
        "RECOVERED",
        (100, 100, 200, 120),
        agreement=0.8,
        with_candidate=False,
    )

    result, diag = merge_passes(
        [],
        [ct],
    )

    assert len(result) == 1

    assert (
        ct.candidates[0].engine
        == "ocr@coarse"
    )

    assert ct.agreement == 0.5


def test_merge_passes_recovered_keeps_lower_agreement():
    ct = token(
        "ABC",
        agreement=0.3,
    )

    merge_passes(
        [],
        [ct],
    )

    assert ct.agreement == 0.3


def test_merge_passes_sorted_output():
    fine = token(
        "BOTTOM",
        (0, 100, 50, 120),
    )

    coarse = token(
        "TOP",
        (0, 0, 50, 20),
        engine="coarse",
    )

    result, _ = merge_passes(
        [fine],
        [coarse],
    )

    assert [
        t.text for t in result
    ] == [
        "TOP",
        "BOTTOM",
    ]


def test_merge_passes_logs_when_recovered():
    coarse = token(
        "ABC",
        (100, 100, 200, 120),
    )

    with patch(
        "uma.ocr.fusion.log.info"
    ) as mock_info:
        merge_passes(
            [],
            [coarse],
        )

    mock_info.assert_called_once()


def test_merge_passes_no_log_without_changes():
    ft = token(
        "ABC"
    )

    ct = token(
        "ABC",
        engine="coarse",
    )

    with patch(
        "uma.ocr.fusion.log.info"
    ) as mock_info:
        merge_passes(
            [ft],
            [ct],
        )

    mock_info.assert_not_called()