import cv2
import numpy as np
import pytest

from types import SimpleNamespace
from unittest.mock import Mock, patch

from uma.preprocess import (
    load_gray,
    text_mask,
    estimate_skew,
    rotate,
    enhance,
    run,
)


# ============================================================
# Helper
# ============================================================

def make_cfg():
    """
    Create a lightweight configuration containing only
    the values required by preprocess.py.
    """
    return SimpleNamespace(
        min_text_ratio=0.01,
        clahe_clip=2.0,
        clahe_grid=8,
        deskew_deadzone_deg=0.5,
    )


# ============================================================
# load_gray
# ============================================================

def test_load_gray_success(tmp_path):
    image = np.zeros(
        (50, 80, 3),
        dtype=np.uint8,
    )

    image[:, :, 0] = 50
    image[:, :, 1] = 100
    image[:, :, 2] = 150

    path = tmp_path / "test.png"

    assert cv2.imwrite(
        str(path),
        image,
    )

    result = load_gray(str(path))

    assert result.ndim == 2
    assert result.shape == (50, 80)
    assert result.dtype == np.uint8


def test_load_gray_nonexistent_file(tmp_path):
    path = tmp_path / "missing.png"

    with pytest.raises(IOError):
        load_gray(str(path))


def test_load_gray_invalid_image(tmp_path):
    path = tmp_path / "invalid.png"

    path.write_text(
        "this is not an image",
        encoding="utf-8",
    )

    with pytest.raises(IOError):
        load_gray(str(path))


# ============================================================
# text_mask
# ============================================================

def test_text_mask_shape():
    gray = np.full(
        (100, 200),
        255,
        dtype=np.uint8,
    )

    cv2.putText(
        gray,
        "TEST",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        0,
        2,
    )

    result = text_mask(gray)

    assert result.shape == gray.shape
    assert result.dtype == np.uint8


def test_text_mask_contains_foreground():
    gray = np.full(
        (100, 200),
        255,
        dtype=np.uint8,
    )

    cv2.putText(
        gray,
        "ABC",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        0,
        2,
    )

    result = text_mask(gray)

    assert np.any(result > 0)


def test_text_mask_blank_image():
    gray = np.full(
        (100, 100),
        255,
        dtype=np.uint8,
    )

    result = text_mask(gray)

    assert result.shape == gray.shape


# ============================================================
# estimate_skew
# ============================================================

def test_estimate_skew_sparse_ratio():
    cfg = make_cfg()

    gray = np.full(
        (100, 100),
        255,
        dtype=np.uint8,
    )

    with patch(
        "uma.preprocess.text_mask",
        return_value=np.zeros(
            (100, 100),
            dtype=np.uint8,
        ),
    ):
        angle, status = estimate_skew(
            gray,
            cfg,
        )

    assert angle == 0.0
    assert status == "skipped_sparse"


def test_estimate_skew_too_few_points():
    cfg = make_cfg()
    cfg.min_text_ratio = 0.0001

    gray = np.zeros(
        (100, 100),
        dtype=np.uint8,
    )

    mask = np.zeros(
        (100, 100),
        dtype=np.uint8,
    )

    mask[10:15, 10:15] = 1

    with patch(
        "uma.preprocess.text_mask",
        return_value=mask,
    ):
        angle, status = estimate_skew(
            gray,
            cfg,
        )

    assert angle == 0.0
    assert status == "skipped_sparse"


def test_estimate_skew_find_nonzero_none():
    cfg = make_cfg()
    cfg.min_text_ratio = 0.0

    gray = np.zeros(
        (100, 100),
        dtype=np.uint8,
    )

    mask = np.zeros(
        (100, 100),
        dtype=np.uint8,
    )

    with patch(
        "uma.preprocess.text_mask",
        return_value=mask,
    ):
        angle, status = estimate_skew(
            gray,
            cfg,
        )

    assert angle == 0.0
    assert status == "skipped_sparse"


def test_estimate_skew_normal_angle():
    cfg = make_cfg()

    gray = np.zeros(
        (100, 100),
        dtype=np.uint8,
    )

    mask = np.ones(
        (100, 100),
        dtype=np.uint8,
    )

    fake_points = np.zeros(
        (100, 1, 2),
        dtype=np.int32,
    )

    with patch(
        "uma.preprocess.text_mask",
        return_value=mask,
    ), patch(
        "uma.preprocess.cv2.findNonZero",
        return_value=fake_points,
    ), patch(
        "uma.preprocess.cv2.minAreaRect",
        return_value=(
            (50, 50),
            (20, 10),
            5.0,
        ),
    ):
        angle, status = estimate_skew(
            gray,
            cfg,
        )

    assert angle == pytest.approx(5.0)
    assert status == "ok"


def test_estimate_skew_negative_angle_adjustment():
    cfg = make_cfg()

    gray = np.zeros(
        (100, 100),
        dtype=np.uint8,
    )

    mask = np.ones(
        (100, 100),
        dtype=np.uint8,
    )

    fake_points = np.zeros(
        (100, 1, 2),
        dtype=np.int32,
    )

    with patch(
        "uma.preprocess.text_mask",
        return_value=mask,
    ), patch(
        "uma.preprocess.cv2.findNonZero",
        return_value=fake_points,
    ), patch(
        "uma.preprocess.cv2.minAreaRect",
        return_value=(
            (50, 50),
            (20, 10),
            -50.0,
        ),
    ):
        angle, status = estimate_skew(
            gray,
            cfg,
        )

    assert angle == 0.0
    assert status == "rejected_outlier"


def test_estimate_skew_positive_angle_adjustment():
    cfg = make_cfg()

    gray = np.zeros(
        (100, 100),
        dtype=np.uint8,
    )

    mask = np.ones(
        (100, 100),
        dtype=np.uint8,
    )

    fake_points = np.zeros(
        (100, 1, 2),
        dtype=np.int32,
    )

    with patch(
        "uma.preprocess.text_mask",
        return_value=mask,
    ), patch(
        "uma.preprocess.cv2.findNonZero",
        return_value=fake_points,
    ), patch(
        "uma.preprocess.cv2.minAreaRect",
        return_value=(
            (50, 50),
            (20, 10),
            50.0,
        ),
    ):
        angle, status = estimate_skew(
            gray,
            cfg,
        )

    assert angle == 0.0
    assert status == "rejected_outlier"


def test_estimate_skew_outlier():
    cfg = make_cfg()

    gray = np.zeros(
        (100, 100),
        dtype=np.uint8,
    )

    mask = np.ones(
        (100, 100),
        dtype=np.uint8,
    )

    fake_points = np.zeros(
        (100, 1, 2),
        dtype=np.int32,
    )

    with patch(
        "uma.preprocess.text_mask",
        return_value=mask,
    ), patch(
        "uma.preprocess.cv2.findNonZero",
        return_value=fake_points,
    ), patch(
        "uma.preprocess.cv2.minAreaRect",
        return_value=(
            (50, 50),
            (20, 10),
            20.0,
        ),
    ):
        angle, status = estimate_skew(
            gray,
            cfg,
        )

    assert angle == 0.0
    assert status == "rejected_outlier"


def test_estimate_skew_boundary_15_degrees():
    cfg = make_cfg()

    gray = np.zeros(
        (100, 100),
        dtype=np.uint8,
    )

    mask = np.ones(
        (100, 100),
        dtype=np.uint8,
    )

    fake_points = np.zeros(
        (100, 1, 2),
        dtype=np.int32,
    )

    with patch(
        "uma.preprocess.text_mask",
        return_value=mask,
    ), patch(
        "uma.preprocess.cv2.findNonZero",
        return_value=fake_points,
    ), patch(
        "uma.preprocess.cv2.minAreaRect",
        return_value=(
            (50, 50),
            (20, 10),
            15.0,
        ),
    ):
        angle, status = estimate_skew(
            gray,
            cfg,
        )

    assert angle == pytest.approx(15.0)
    assert status == "ok"


# ============================================================
# rotate
# ============================================================

def test_rotate_preserves_shape():
    gray = np.zeros(
        (100, 200),
        dtype=np.uint8,
    )

    result = rotate(
        gray,
        5.0,
    )

    assert result.shape == gray.shape
    assert result.dtype == gray.dtype


def test_rotate_zero_angle():
    gray = np.arange(
        10000,
        dtype=np.uint8,
    ).reshape(100, 100)

    result = rotate(
        gray,
        0.0,
    )

    assert result.shape == gray.shape

    assert np.array_equal(
        result,
        gray,
    )


def test_rotate_calls_opencv():
    gray = np.zeros(
        (100, 200),
        dtype=np.uint8,
    )

    fake_matrix = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
        ],
        dtype=float,
    )

    fake_result = np.ones_like(gray)

    with patch(
        "uma.preprocess.cv2.getRotationMatrix2D",
        return_value=fake_matrix,
    ) as mock_matrix, patch(
        "uma.preprocess.cv2.warpAffine",
        return_value=fake_result,
    ) as mock_warp:

        result = rotate(
            gray,
            3.0,
        )

    mock_matrix.assert_called_once_with(
        (100.0, 50.0),
        3.0,
        1.0,
    )

    mock_warp.assert_called_once()

    assert result is fake_result


# ============================================================
# enhance
# ============================================================

def test_enhance_shape():
    cfg = make_cfg()

    gray = np.full(
        (100, 100),
        120,
        dtype=np.uint8,
    )

    result = enhance(
        gray,
        cfg,
    )

    assert result.shape == gray.shape
    assert result.dtype == np.uint8


def test_enhance_calls_clahe():
    cfg = make_cfg()

    gray = np.zeros(
        (20, 20),
        dtype=np.uint8,
    )

    fake_output = np.ones_like(gray)

    clahe = Mock()
    clahe.apply.return_value = fake_output

    with patch(
        "uma.preprocess.cv2.createCLAHE",
        return_value=clahe,
    ) as mock_create:

        result = enhance(
            gray,
            cfg,
        )

    mock_create.assert_called_once_with(
        clipLimit=cfg.clahe_clip,
        tileGridSize=(
            cfg.clahe_grid,
            cfg.clahe_grid,
        ),
    )

    clahe.apply.assert_called_once_with(
        gray
    )

    assert result is fake_output


# ============================================================
# run
# ============================================================

def test_run_without_deskew():
    cfg = make_cfg()

    gray = np.zeros(
        (100, 200),
        dtype=np.uint8,
    )

    enhanced = np.ones(
        (100, 200),
        dtype=np.uint8,
    )

    with patch(
        "uma.preprocess.load_gray",
        return_value=gray,
    ), patch(
        "uma.preprocess.estimate_skew",
        return_value=(
            0.1,
            "ok",
        ),
    ), patch(
        "uma.preprocess.rotate"
    ) as mock_rotate, patch(
        "uma.preprocess.enhance",
        return_value=enhanced,
    ) as mock_enhance:

        result_enhanced, plain, meta = run(
            "test.png",
            cfg,
        )

    assert result_enhanced is enhanced
    assert plain is gray

    assert meta["width"] == 200
    assert meta["height"] == 100

    assert meta["skew_angle"] == 0.1
    assert meta["skew_status"] == "ok"
    assert meta["deskewed"] is False

    mock_rotate.assert_not_called()

    # FIXED:
    # run() calls enhance(gray, cfg)
    mock_enhance.assert_called_once_with(
        gray,
        cfg,
    )


def test_run_with_deskew():
    cfg = make_cfg()

    gray = np.zeros(
        (100, 200),
        dtype=np.uint8,
    )

    rotated = np.full(
        (100, 200),
        50,
        dtype=np.uint8,
    )

    enhanced = np.full(
        (100, 200),
        100,
        dtype=np.uint8,
    )

    with patch(
        "uma.preprocess.load_gray",
        return_value=gray,
    ), patch(
        "uma.preprocess.estimate_skew",
        return_value=(
            5.12345,
            "ok",
        ),
    ), patch(
        "uma.preprocess.rotate",
        return_value=rotated,
    ) as mock_rotate, patch(
        "uma.preprocess.enhance",
        return_value=enhanced,
    ) as mock_enhance:

        result_enhanced, plain, meta = run(
            "test.png",
            cfg,
        )

    assert result_enhanced is enhanced
    assert plain is rotated

    assert meta["width"] == 200
    assert meta["height"] == 100

    assert meta["skew_angle"] == 5.123
    assert meta["skew_status"] == "ok"
    assert meta["deskewed"] is True

    mock_rotate.assert_called_once_with(
        gray,
        5.12345,
    )

    # FIXED:
    # run() calls enhance(rotated, cfg)
    mock_enhance.assert_called_once_with(
        rotated,
        cfg,
    )


def test_run_skipped_sparse():
    cfg = make_cfg()

    gray = np.zeros(
        (50, 80),
        dtype=np.uint8,
    )

    enhanced = np.ones_like(gray)

    with patch(
        "uma.preprocess.load_gray",
        return_value=gray,
    ), patch(
        "uma.preprocess.estimate_skew",
        return_value=(
            0.0,
            "skipped_sparse",
        ),
    ), patch(
        "uma.preprocess.rotate"
    ) as mock_rotate, patch(
        "uma.preprocess.enhance",
        return_value=enhanced,
    ) as mock_enhance:

        result_enhanced, plain, meta = run(
            "test.png",
            cfg,
        )

    assert result_enhanced is enhanced
    assert plain is gray

    assert meta["deskewed"] is False
    assert meta["skew_status"] == "skipped_sparse"

    mock_rotate.assert_not_called()

    mock_enhance.assert_called_once_with(
        gray,
        cfg,
    )


def test_run_outlier_not_rotated():
    cfg = make_cfg()

    gray = np.zeros(
        (60, 90),
        dtype=np.uint8,
    )

    with patch(
        "uma.preprocess.load_gray",
        return_value=gray,
    ), patch(
        "uma.preprocess.estimate_skew",
        return_value=(
            0.0,
            "rejected_outlier",
        ),
    ), patch(
        "uma.preprocess.rotate"
    ) as mock_rotate, patch(
        "uma.preprocess.enhance",
        return_value=gray,
    ) as mock_enhance:

        result_enhanced, plain, meta = run(
            "test.png",
            cfg,
        )

    assert result_enhanced is gray
    assert plain is gray

    assert meta["deskewed"] is False

    assert (
        meta["skew_status"]
        == "rejected_outlier"
    )

    mock_rotate.assert_not_called()

    mock_enhance.assert_called_once_with(
        gray,
        cfg,
    )


def test_run_exact_deadzone_rotates():
    cfg = make_cfg()

    cfg.deskew_deadzone_deg = 0.5

    gray = np.zeros(
        (100, 100),
        dtype=np.uint8,
    )

    rotated = np.ones_like(gray)

    with patch(
        "uma.preprocess.load_gray",
        return_value=gray,
    ), patch(
        "uma.preprocess.estimate_skew",
        return_value=(
            0.5,
            "ok",
        ),
    ), patch(
        "uma.preprocess.rotate",
        return_value=rotated,
    ) as mock_rotate, patch(
        "uma.preprocess.enhance",
        return_value=rotated,
    ) as mock_enhance:

        result_enhanced, plain, meta = run(
            "test.png",
            cfg,
        )

    assert result_enhanced is rotated
    assert plain is rotated

    assert meta["deskewed"] is True

    mock_rotate.assert_called_once_with(
        gray,
        0.5,
    )

    mock_enhance.assert_called_once_with(
        rotated,
        cfg,
    )