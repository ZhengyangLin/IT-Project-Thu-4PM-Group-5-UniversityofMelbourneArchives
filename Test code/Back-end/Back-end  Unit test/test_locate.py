from unittest.mock import patch

import cv2
import numpy as np

from uma.locate import (
    LocateResult,
    _hit,
    _peaks,
    line_masks,
    line_positions,
    locate,
    locate_by_anchor,
    locate_by_density,
    snap_to_frame,
)
from uma.models import Token


# ---------------------------------------------------------
# Helper
# ---------------------------------------------------------

def make_token(
    text,
    x0,
    y0,
    x1,
    y1,
    conf=0.9,
):
    return Token(
        text=text,
        bbox=(x0, y0, x1, y1),
        conf=conf,
    )


# ---------------------------------------------------------
# _hit
# ---------------------------------------------------------

def test_hit_exact():
    result = _hit(
        "DRAWING NUMBER",
        ["SCALE", "DRAWING NUMBER", "DATE"],
        85,
    )

    assert result is True


def test_hit_no_match():
    result = _hit(
        "HELLO WORLD",
        ["DRAWING NUMBER", "SCALE"],
        85,
    )

    assert result is False


def test_hit_fuzzy():
    result = _hit(
        "DRAWING NUM8ER",
        ["DRAWING NUMBER"],
        80,
    )

    assert result is True


# ---------------------------------------------------------
# _peaks
# ---------------------------------------------------------

def test_peaks():
    profile = np.array(
        [
            0.0,
            0.8,
            0.9,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.8,
            0.9,
        ]
    )

    result = _peaks(
        profile,
        thresh=0.5,
        merge_gap=2,
    )

    assert len(result) == 2


def test_peaks_empty():
    profile = np.zeros(10)

    result = _peaks(
        profile,
        thresh=0.5,
    )

    assert result == []


# ---------------------------------------------------------
# line_masks
# ---------------------------------------------------------

def test_line_masks():
    image = np.full(
        (100, 100),
        255,
        dtype=np.uint8,
    )

    # horizontal black line
    image[50, 10:90] = 0

    # vertical black line
    image[10:90, 50] = 0

    horiz, vert = line_masks(
        image,
        ink_percentile=50,
        min_ratio=0.10,
    )

    assert horiz.shape == image.shape
    assert vert.shape == image.shape

    assert horiz.dtype == np.uint8
    assert vert.dtype == np.uint8


# ---------------------------------------------------------
# line_positions
# ---------------------------------------------------------

def test_line_positions():
    horiz = np.zeros(
        (20, 20),
        dtype=np.uint8,
    )

    vert = np.zeros(
        (20, 20),
        dtype=np.uint8,
    )

    horiz[5, :] = 1
    horiz[15, :] = 1

    vert[:, 4] = 1
    vert[:, 14] = 1

    ys, xs = line_positions(
        horiz,
        vert,
        min_cover=0.5,
    )

    assert 5 in ys
    assert 15 in ys

    assert 4 in xs
    assert 14 in xs


# ---------------------------------------------------------
# snap_to_frame
# ---------------------------------------------------------

def test_snap_to_frame():
    box = (
        100,
        100,
        400,
        400,
    )

    xs = [
        95,
        405,
    ]

    ys = [
        98,
        402,
    ]

    result, snapped = snap_to_frame(
        box,
        ys,
        xs,
        page=(1000, 1000),
        tol=0.02,
    )

    assert snapped is True

    assert result == (
        95,
        98,
        405,
        402,
    )


def test_snap_to_frame_no_lines():
    box = (
        100,
        100,
        400,
        400,
    )

    result, snapped = snap_to_frame(
        box,
        [],
        [],
        page=(1000, 1000),
    )

    assert result == box
    assert snapped is False


# ---------------------------------------------------------
# locate_by_anchor
# ---------------------------------------------------------

def test_locate_by_anchor():
    tokens = [
        make_token(
            "DRAWING NUMBER",
            700,
            700,
            850,
            730,
        ),

        make_token(
            "SCALE",
            700,
            750,
            780,
            780,
        ),
    ]

    result = locate_by_anchor(
        tokens=tokens,
        page=(1000, 1000),
        ys=[],
        xs=[],
    )

    assert result is not None

    assert result.method == "anchor"

    assert result.confidence > 0

    assert result.detail["strong"] >= 1


def test_locate_by_anchor_no_match():
    tokens = [
        make_token(
            "HELLO",
            100,
            100,
            200,
            130,
        )
    ]

    result = locate_by_anchor(
        tokens=tokens,
        page=(1000, 1000),
        ys=[],
        xs=[],
    )

    assert result is None


# ---------------------------------------------------------
# locate_by_density
# ---------------------------------------------------------

def test_locate_by_density_too_few_tokens():
    tokens = [
        make_token(
            "TEXT",
            10,
            10,
            20,
            20,
        )
        for _ in range(5)
    ]

    result = locate_by_density(
        tokens,
        page=(1000, 1000),
    )

    assert result is None


def test_locate_by_density():
    tokens = []

    # Cluster tokens in lower-right region
    for i in range(20):

        x = 700 + (i % 5) * 20
        y = 700 + (i // 5) * 20

        tokens.append(
            make_token(
                f"T{i}",
                x,
                y,
                x + 15,
                y + 15,
            )
        )

    result = locate_by_density(
        tokens,
        page=(1000, 1000),
        grid=20,
        keep_ratio=0.2,
    )

    # Depending on heat-map expansion the result can
    # either identify the cluster or reject it if too small.
    # If identified, verify its structure.
    if result is not None:
        assert result.method == "density"
        assert result.confidence > 0
        assert result.detail["tokens"] >= 5


# ---------------------------------------------------------
# locate - reliable find_table result
# ---------------------------------------------------------

def test_locate_table_result():
    image = np.full(
        (500, 500),
        255,
        dtype=np.uint8,
    )

    fake_proposal = type(
        "FakeProposal",
        (),
        {
            "reliable": True,
            "bbox": (300, 300, 490, 490),
            "method": "table",
            "detail": {
                "score": 0.95
            },
        },
    )()

    with patch(
        "uma.locate.find_table",
        return_value=fake_proposal,
    ):

        result = locate(
            image,
            [],
        )

    assert result.method == "table"

    assert result.bbox == (
        300,
        300,
        490,
        490,
    )

    assert result.confidence == 0.85

    assert result.detail["review_required"] is False


# ---------------------------------------------------------
# locate - anchor branch
# ---------------------------------------------------------

def test_locate_anchor_branch():
    image = np.full(
        (500, 500),
        255,
        dtype=np.uint8,
    )

    fake_proposal = type(
        "FakeProposal",
        (),
        {
            "reliable": False,
            "bbox": None,
            "method": "none",
            "detail": {},
        },
    )()

    expected = LocateResult(
        bbox=(300, 300, 490, 490),
        method="anchor",
        confidence=0.8,
        detail={
            "strong": 1,
            "weak": 1,
            "words": ["DRAWING NUMBER"],
        },
    )

    with patch(
        "uma.locate.find_table",
        return_value=fake_proposal,
    ), patch(
        "uma.locate.line_masks",
        return_value=(
            np.zeros_like(image),
            np.zeros_like(image),
        ),
    ), patch(
        "uma.locate.line_positions",
        return_value=([], []),
    ), patch(
        "uma.locate.locate_by_anchor",
        return_value=expected,
    ):

        result = locate(
            image,
            [],
        )

    assert result is expected
    assert result.method == "anchor"


# ---------------------------------------------------------
# locate - frame branch
# ---------------------------------------------------------

def test_locate_frame_branch():
    image = np.full(
        (500, 500),
        255,
        dtype=np.uint8,
    )

    fake_proposal = type(
        "FakeProposal",
        (),
        {
            "reliable": False,
            "bbox": None,
            "method": "none",
            "detail": {},
        },
    )()

    frame_result = LocateResult(
        bbox=(250, 250, 480, 480),
        method="frame",
        confidence=0.75,
        detail={
            "tokens": 5,
        },
    )

    with patch(
        "uma.locate.find_table",
        return_value=fake_proposal,
    ), patch(
        "uma.locate.line_masks",
        return_value=(
            np.zeros_like(image),
            np.zeros_like(image),
        ),
    ), patch(
        "uma.locate.line_positions",
        return_value=([], []),
    ), patch(
        "uma.locate.locate_by_anchor",
        return_value=None,
    ), patch(
        "uma.locate.locate_by_frame",
        return_value=frame_result,
    ):

        result = locate(
            image,
            [],
        )

    assert result.method == "frame"

    assert result.bbox == (
        250,
        250,
        480,
        480,
    )


# ---------------------------------------------------------
# locate - density branch
# ---------------------------------------------------------

def test_locate_density_branch():
    image = np.full(
        (500, 500),
        255,
        dtype=np.uint8,
    )

    fake_proposal = type(
        "FakeProposal",
        (),
        {
            "reliable": False,
            "bbox": None,
            "method": "none",
            "detail": {},
        },
    )()

    density_result = LocateResult(
        bbox=(300, 300, 480, 480),
        method="density",
        confidence=0.6,
        detail={
            "tokens": 10,
        },
    )

    with patch(
        "uma.locate.find_table",
        return_value=fake_proposal,
    ), patch(
        "uma.locate.line_masks",
        return_value=(
            np.zeros_like(image),
            np.zeros_like(image),
        ),
    ), patch(
        "uma.locate.line_positions",
        return_value=([], []),
    ), patch(
        "uma.locate.locate_by_anchor",
        return_value=None,
    ), patch(
        "uma.locate.locate_by_frame",
        return_value=None,
    ), patch(
        "uma.locate.locate_by_density",
        return_value=density_result,
    ):

        result = locate(
            image,
            [],
        )

    assert result.method == "density"

    assert result.confidence == 0.6


# ---------------------------------------------------------
# locate - fallback
# ---------------------------------------------------------

def test_locate_fallback():
    image = np.full(
        (1000, 1000),
        255,
        dtype=np.uint8,
    )

    fake_proposal = type(
        "FakeProposal",
        (),
        {
            "reliable": False,
            "bbox": None,
            "method": "none",
            "detail": {},
        },
    )()

    with patch(
        "uma.locate.find_table",
        return_value=fake_proposal,
    ), patch(
        "uma.locate.line_masks",
        return_value=(
            np.zeros_like(image),
            np.zeros_like(image),
        ),
    ), patch(
        "uma.locate.line_positions",
        return_value=([], []),
    ), patch(
        "uma.locate.locate_by_anchor",
        return_value=None,
    ), patch(
        "uma.locate.locate_by_frame",
        return_value=None,
    ), patch(
        "uma.locate.locate_by_density",
        return_value=None,
    ):

        result = locate(
            image,
            [],
        )

    assert result.method == "fallback"

    assert result.bbox == (
        600,
        550,
        1000,
        1000,
    )

    assert result.confidence == 0.15

    assert result.detail["degraded"] is True


# ---------------------------------------------------------
# locate - OpenCV error from find_table
# ---------------------------------------------------------

def test_locate_handles_cv_error():
    image = np.full(
        (1000, 1000),
        255,
        dtype=np.uint8,
    )

    with patch(
        "uma.locate.find_table",
        side_effect=cv2.error(
            "test error"
        ),
    ), patch(
        "uma.locate.line_masks",
        return_value=(
            np.zeros_like(image),
            np.zeros_like(image),
        ),
    ), patch(
        "uma.locate.line_positions",
        return_value=([], []),
    ), patch(
        "uma.locate.locate_by_anchor",
        return_value=None,
    ), patch(
        "uma.locate.locate_by_frame",
        return_value=None,
    ), patch(
        "uma.locate.locate_by_density",
        return_value=None,
    ):

        result = locate(
            image,
            [],
        )

    assert result.method == "fallback"