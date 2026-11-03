from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from uma.viz import (
    C_BLOCK,
    C_RESPLIT,
    C_CLUSTER,
    C_TITLE,
    C_LINE,
    _legend,
    draw,
    draw_v2,
    save_titleblock_crop,
    draw_tokens,
)


# ============================================================
# Helpers
# ============================================================

def make_gray(width=400, height=300):
    return np.full(
        (height, width),
        128,
        dtype=np.uint8,
    )


def make_line(
    bbox=(10, 10, 100, 30),
):
    return SimpleNamespace(
        bbox=bbox,
    )


def make_block(
    bbox=(10, 10, 200, 100),
    source="xy_cut",
    lines=None,
):
    if lines is None:
        lines = [
            make_line(
                (20, 20, 180, 40)
            )
        ]

    return SimpleNamespace(
        bbox=bbox,
        source=source,
        lines=lines,
    )


def make_cluster(
    bbox=(50, 50, 150, 120),
):
    return SimpleNamespace(
        bbox=bbox,
    )


def make_token(
    bbox=(20, 20, 80, 40),
):
    return SimpleNamespace(
        bbox=bbox,
    )


def make_locate_result(
    bbox=(100, 100, 300, 200),
    method="anchor",
    confidence=0.95,
):
    return SimpleNamespace(
        bbox=bbox,
        method=method,
        confidence=confidence,
    )


# ============================================================
# Constants
# ============================================================

def test_colors():
    assert C_BLOCK == (200, 160, 60)
    assert C_RESPLIT == (200, 90, 190)
    assert C_CLUSTER == (60, 160, 240)
    assert C_TITLE == (60, 60, 230)
    assert C_LINE == (150, 150, 150)


# ============================================================
# _legend
# ============================================================

def test_legend_draws_background():
    canvas = np.zeros(
        (300, 800, 3),
        dtype=np.uint8,
    )

    _legend(canvas)

    # Legend should modify the image.
    assert np.any(canvas != 0)


def test_legend_calls_rectangle_and_text():
    canvas = np.zeros(
        (300, 800, 3),
        dtype=np.uint8,
    )

    with (
        patch(
            "uma.viz.cv2.rectangle"
        ) as mock_rectangle,
        patch(
            "uma.viz.cv2.putText"
        ) as mock_text,
    ):
        _legend(canvas)

    # 1 background rectangle
    # + 4 colored legend boxes.
    assert mock_rectangle.call_count == 5

    # 4 labels.
    assert mock_text.call_count == 4


def test_legend_labels():
    canvas = np.zeros(
        (300, 800, 3),
        dtype=np.uint8,
    )

    labels = []

    def fake_put_text(
        img,
        text,
        org,
        font,
        scale,
        color,
        thickness,
    ):
        labels.append(text)
        return img

    with patch(
        "uma.viz.cv2.putText",
        side_effect=fake_put_text,
    ):
        _legend(canvas)

    assert "block (xy-cut)" in labels
    assert "block (caption resplit)" in labels
    assert "cluster" in labels
    assert "title block" in labels


# ============================================================
# draw
# ============================================================

def test_draw_basic(tmp_path):
    gray = make_gray()

    block = make_block()
    cluster = make_cluster()

    out = tmp_path / "viz.png"

    result = draw(
        gray,
        [block],
        [cluster],
        None,
        str(out),
    )

    assert result == str(out)
    assert out.exists()

    image = cv2.imread(str(out))

    assert image is not None
    assert image.ndim == 3


def test_draw_empty_objects(tmp_path):
    gray = make_gray()

    out = tmp_path / "empty.png"

    result = draw(
        gray,
        [],
        [],
        None,
        str(out),
    )

    assert result == str(out)
    assert out.exists()


def test_draw_show_lines_false(tmp_path):
    gray = make_gray()

    block = make_block(
        lines=[
            make_line(
                (10, 10, 50, 30)
            )
        ]
    )

    out = tmp_path / "no_lines.png"

    with patch(
        "uma.viz.cv2.rectangle"
    ) as mock_rectangle:
        draw(
            gray,
            [block],
            [],
            None,
            str(out),
            show_lines=False,
        )

    # Block + legend rectangles exist,
    # but no line rectangle should use C_LINE.
    calls = mock_rectangle.call_args_list

    assert not any(
        len(call.args) >= 5
        and call.args[3] == C_LINE
        for call in calls
    )


def test_draw_show_lines_true(tmp_path):
    gray = make_gray()

    line = make_line(
        (20, 20, 100, 40)
    )

    block = make_block(
        lines=[line]
    )

    out = tmp_path / "lines.png"

    with patch(
        "uma.viz.cv2.rectangle"
    ) as mock_rectangle:
        draw(
            gray,
            [block],
            [],
            None,
            str(out),
            show_lines=True,
        )

    assert any(
        len(call.args) >= 5
        and call.args[3] == C_LINE
        and call.args[4] == 1
        for call in mock_rectangle.call_args_list
    )


def test_draw_normal_block_color(tmp_path):
    gray = make_gray()

    block = make_block(
        source="xy_cut"
    )

    out = tmp_path / "block.png"

    with patch(
        "uma.viz.cv2.rectangle"
    ) as mock_rectangle:
        draw(
            gray,
            [block],
            [],
            None,
            str(out),
        )

    assert any(
        len(call.args) >= 5
        and call.args[3] == C_BLOCK
        and call.args[4] == 5
        for call in mock_rectangle.call_args_list
    )


def test_draw_caption_resplit_color(tmp_path):
    gray = make_gray()

    block = make_block(
        source="caption_resplit"
    )

    out = tmp_path / "resplit.png"

    with patch(
        "uma.viz.cv2.rectangle"
    ) as mock_rectangle:
        draw(
            gray,
            [block],
            [],
            None,
            str(out),
        )

    assert any(
        len(call.args) >= 5
        and call.args[3] == C_RESPLIT
        and call.args[4] == 5
        for call in mock_rectangle.call_args_list
    )


def test_draw_block_label(tmp_path):
    gray = make_gray()

    block = make_block(
        lines=[
            make_line(),
            make_line(
                (10, 40, 100, 60)
            ),
        ]
    )

    out = tmp_path / "label.png"

    texts = []

    def fake_text(
        img,
        text,
        *args,
        **kwargs,
    ):
        texts.append(text)
        return img

    with patch(
        "uma.viz.cv2.putText",
        side_effect=fake_text,
    ):
        draw(
            gray,
            [block],
            [],
            None,
            str(out),
        )

    assert "B1(2)" in texts


def test_draw_multiple_blocks_numbered(tmp_path):
    gray = make_gray()

    b1 = make_block(
        (10, 10, 100, 80)
    )

    b2 = make_block(
        (120, 10, 220, 80)
    )

    out = tmp_path / "blocks.png"

    texts = []

    def fake_text(
        img,
        text,
        *args,
        **kwargs,
    ):
        texts.append(text)
        return img

    with patch(
        "uma.viz.cv2.putText",
        side_effect=fake_text,
    ):
        draw(
            gray,
            [b1, b2],
            [],
            None,
            str(out),
        )

    assert "B1(1)" in texts
    assert "B2(1)" in texts


def test_draw_cluster(tmp_path):
    gray = make_gray()

    cluster = make_cluster(
        (50, 60, 150, 120)
    )

    out = tmp_path / "cluster.png"

    with patch(
        "uma.viz.cv2.rectangle"
    ) as mock_rectangle:
        draw(
            gray,
            [],
            [cluster],
            None,
            str(out),
        )

    assert any(
        call.args[1] == (47, 57)
        and call.args[2] == (153, 123)
        and call.args[3] == C_CLUSTER
        for call in mock_rectangle.call_args_list
    )


def test_draw_title_bbox(tmp_path):
    gray = make_gray()

    out = tmp_path / "title.png"

    texts = []

    def fake_text(
        img,
        text,
        *args,
        **kwargs,
    ):
        texts.append(text)
        return img

    with patch(
        "uma.viz.cv2.putText",
        side_effect=fake_text,
    ):
        draw(
            gray,
            [],
            [],
            (100, 100, 300, 200),
            str(out),
        )

    assert "TITLE BLOCK" in texts


def test_draw_no_title_bbox(tmp_path):
    gray = make_gray()

    out = tmp_path / "no_title.png"

    texts = []

    def fake_text(
        img,
        text,
        *args,
        **kwargs,
    ):
        texts.append(text)
        return img

    with patch(
        "uma.viz.cv2.putText",
        side_effect=fake_text,
    ):
        draw(
            gray,
            [],
            [],
            None,
            str(out),
        )

    assert "TITLE BLOCK" not in texts


def test_draw_calls_legend(tmp_path):
    gray = make_gray()

    out = tmp_path / "legend.png"

    with patch(
        "uma.viz._legend"
    ) as mock_legend:
        draw(
            gray,
            [],
            [],
            None,
            str(out),
        )

    mock_legend.assert_called_once()


def test_draw_no_resize_when_small(tmp_path):
    gray = make_gray(
        width=500,
        height=300,
    )

    out = tmp_path / "small.png"

    with patch(
        "uma.viz.cv2.resize"
    ) as mock_resize:
        draw(
            gray,
            [],
            [],
            None,
            str(out),
            max_width=1000,
        )

    mock_resize.assert_not_called()


def test_draw_resize_when_large(tmp_path):
    gray = make_gray(
        width=1000,
        height=500,
    )

    out = tmp_path / "large.png"

    with patch(
        "uma.viz.cv2.resize",
        wraps=cv2.resize,
    ) as mock_resize:
        draw(
            gray,
            [],
            [],
            None,
            str(out),
            max_width=500,
        )

    mock_resize.assert_called_once()

    args = mock_resize.call_args.args

    assert args[1] == (
        500,
        250,
    )


def test_draw_returns_out_path(tmp_path):
    out = tmp_path / "return.png"

    result = draw(
        make_gray(),
        [],
        [],
        None,
        str(out),
    )

    assert result == str(out)


# ============================================================
# draw_v2
# ============================================================

def test_draw_v2_basic(tmp_path):
    plain = make_gray()

    token = make_token()

    result_obj = make_locate_result()

    out = tmp_path / "v2.png"

    with (
        patch(
            "uma.locate.line_masks",
            return_value=(
                np.zeros_like(plain),
                np.zeros_like(plain),
            ),
        ),
        patch(
            "uma.locate.line_positions",
            return_value=(
                [],
                [],
            ),
        ),
    ):
        result = draw_v2(
            plain,
            [token],
            result_obj,
            str(out),
        )

    assert result == str(out)
    assert out.exists()


def test_draw_v2_draws_horizontal_lines(tmp_path):
    plain = make_gray(
        width=400,
        height=300,
    )

    locate_result = make_locate_result()

    out = tmp_path / "horizontal.png"

    with (
        patch(
            "uma.locate.line_masks",
            return_value=(
                np.zeros_like(plain),
                np.zeros_like(plain),
            ),
        ),
        patch(
            "uma.locate.line_positions",
            return_value=(
                [50, 100],
                [],
            ),
        ),
        patch(
            "uma.viz.cv2.line"
        ) as mock_line,
    ):
        draw_v2(
            plain,
            [],
            locate_result,
            str(out),
        )

    assert mock_line.call_count == 2

    mock_line.assert_any_call(
        mock_line.call_args_list[0].args[0],
        (0, 50),
        (400, 50),
        (230, 190, 110),
        2,
    )


def test_draw_v2_draws_vertical_lines(tmp_path):
    plain = make_gray(
        width=400,
        height=300,
    )

    locate_result = make_locate_result()

    out = tmp_path / "vertical.png"

    with (
        patch(
            "uma.locate.line_masks",
            return_value=(
                np.zeros_like(plain),
                np.zeros_like(plain),
            ),
        ),
        patch(
            "uma.locate.line_positions",
            return_value=(
                [],
                [50, 200],
            ),
        ),
        patch(
            "uma.viz.cv2.line"
        ) as mock_line,
    ):
        draw_v2(
            plain,
            [],
            locate_result,
            str(out),
        )

    assert mock_line.call_count == 2


def test_draw_v2_draws_tokens(tmp_path):
    plain = make_gray()

    tokens = [
        make_token(
            (10, 20, 100, 40)
        ),
        make_token(
            (120, 50, 200, 80)
        ),
    ]

    locate_result = make_locate_result()

    out = tmp_path / "tokens.png"

    with (
        patch(
            "uma.locate.line_masks",
            return_value=(
                np.zeros_like(plain),
                np.zeros_like(plain),
            ),
        ),
        patch(
            "uma.locate.line_positions",
            return_value=(
                [],
                [],
            ),
        ),
        patch(
            "uma.viz.cv2.rectangle"
        ) as mock_rectangle,
    ):
        draw_v2(
            plain,
            tokens,
            locate_result,
            str(out),
        )

    assert any(
        call.args[1] == (10, 20)
        and call.args[2] == (100, 40)
        and call.args[3] == (150, 150, 150)
        for call in mock_rectangle.call_args_list
    )


def test_draw_v2_title_label(tmp_path):
    plain = make_gray()

    locate_result = make_locate_result(
        method="frame",
        confidence=0.876,
    )

    out = tmp_path / "label.png"

    texts = []

    def fake_text(
        img,
        text,
        *args,
        **kwargs,
    ):
        texts.append(text)
        return img

    with (
        patch(
            "uma.locate.line_masks",
            return_value=(
                np.zeros_like(plain),
                np.zeros_like(plain),
            ),
        ),
        patch(
            "uma.locate.line_positions",
            return_value=(
                [],
                [],
            ),
        ),
        patch(
            "uma.viz.cv2.putText",
            side_effect=fake_text,
        ),
    ):
        draw_v2(
            plain,
            [],
            locate_result,
            str(out),
        )

    assert "frame  0.88" in texts


def test_draw_v2_legend_text(tmp_path):
    plain = make_gray()

    locate_result = make_locate_result(
        method="anchor",
        confidence=0.9,
    )

    tokens = [
        make_token(),
        make_token(
            (100, 100, 150, 120)
        ),
    ]

    out = tmp_path / "legend_v2.png"

    texts = []

    def fake_text(
        img,
        text,
        *args,
        **kwargs,
    ):
        texts.append(text)
        return img

    with (
        patch(
            "uma.locate.line_masks",
            return_value=(
                np.zeros_like(plain),
                np.zeros_like(plain),
            ),
        ),
        patch(
            "uma.locate.line_positions",
            return_value=(
                [10, 20],
                [30],
            ),
        ),
        patch(
            "uma.viz.cv2.putText",
            side_effect=fake_text,
        ),
    ):
        draw_v2(
            plain,
            tokens,
            locate_result,
            str(out),
        )

    assert any(
        "frame lines  h=2 v=1" == text
        for text in texts
    )

    assert any(
        "ocr tokens  n=2" == text
        for text in texts
    )

    assert any(
        "title block  [anchor]" == text
        for text in texts
    )


def test_draw_v2_no_resize_when_small(tmp_path):
    plain = make_gray(
        width=500,
        height=300,
    )

    locate_result = make_locate_result(
        bbox=(100, 100, 300, 200),
    )

    out = tmp_path / "small_v2.png"

    with (
        patch(
            "uma.locate.line_masks",
            return_value=(
                np.zeros_like(plain),
                np.zeros_like(plain),
            ),
        ),
        patch(
            "uma.locate.line_positions",
            return_value=(
                [],
                [],
            ),
        ),
        patch(
            "uma.viz.cv2.resize"
        ) as mock_resize,
    ):
        draw_v2(
            plain,
            [],
            locate_result,
            str(out),
            max_width=1000,
        )

    mock_resize.assert_not_called()


def test_draw_v2_resize_large(tmp_path):
    plain = make_gray(
        width=1000,
        height=500,
    )

    locate_result = make_locate_result(
        bbox=(100, 100, 500, 300),
    )

    out = tmp_path / "large_v2.png"

    with (
        patch(
            "uma.locate.line_masks",
            return_value=(
                np.zeros_like(plain),
                np.zeros_like(plain),
            ),
        ),
        patch(
            "uma.locate.line_positions",
            return_value=(
                [],
                [],
            ),
        ),
        patch(
            "uma.viz.cv2.resize",
            wraps=cv2.resize,
        ) as mock_resize,
    ):
        draw_v2(
            plain,
            [],
            locate_result,
            str(out),
            max_width=500,
        )

    mock_resize.assert_called_once()

    assert (
        mock_resize.call_args.args[1]
        == (500, 250)
    )


# ============================================================
# save_titleblock_crop
# ============================================================

def test_save_titleblock_crop_basic(tmp_path):
    gray = make_gray(
        width=1200,
        height=800,
    )

    out = tmp_path / "crop.png"

    result = save_titleblock_crop(
        gray,
        (200, 200, 1000, 600),
        str(out),
        pad_ratio=0,
        max_width=1800,
    )

    assert result == str(out)
    assert out.exists()

    image = cv2.imread(str(out))

    assert image is not None


def test_save_titleblock_crop_padding_clamped_top_left(
    tmp_path,
):
    gray = make_gray(
        width=1000,
        height=800,
    )

    out = tmp_path / "crop_edge.png"

    result = save_titleblock_crop(
        gray,
        (0, 0, 300, 200),
        str(out),
        pad_ratio=0.5,
    )

    assert result == str(out)
    assert out.exists()


def test_save_titleblock_crop_padding_clamped_bottom_right(
    tmp_path,
):
    gray = make_gray(
        width=1000,
        height=800,
    )

    out = tmp_path / "crop_edge2.png"

    result = save_titleblock_crop(
        gray,
        (800, 600, 1000, 800),
        str(out),
        pad_ratio=0.5,
    )

    assert result == str(out)
    assert out.exists()


def test_save_titleblock_crop_downscales_large_crop(
    tmp_path,
):
    gray = make_gray(
        width=2000,
        height=1000,
    )

    out = tmp_path / "crop_large.png"

    with patch(
        "uma.viz.cv2.resize",
        wraps=cv2.resize,
    ) as mock_resize:
        save_titleblock_crop(
            gray,
            (0, 0, 2000, 1000),
            str(out),
            pad_ratio=0,
            max_width=1000,
        )

    mock_resize.assert_called_once()

    assert (
        mock_resize.call_args.args[1]
        == (1000, 500)
    )

    assert (
        mock_resize.call_args.kwargs[
            "interpolation"
        ]
        == cv2.INTER_AREA
    )


def test_save_titleblock_crop_upscales_small_crop(
    tmp_path,
):
    gray = make_gray(
        width=500,
        height=500,
    )

    out = tmp_path / "crop_small.png"

    with patch(
        "uma.viz.cv2.resize",
        wraps=cv2.resize,
    ) as mock_resize:
        save_titleblock_crop(
            gray,
            (100, 100, 300, 300),
            str(out),
            pad_ratio=0,
        )

    mock_resize.assert_called_once()

    # 200 wide -> scaled to 700.
    assert (
        mock_resize.call_args.args[1]
        == (700, 700)
    )

    assert (
        mock_resize.call_args.kwargs[
            "interpolation"
        ]
        == cv2.INTER_CUBIC
    )


def test_save_titleblock_crop_medium_no_resize(
    tmp_path,
):
    gray = make_gray(
        width=1200,
        height=800,
    )

    out = tmp_path / "crop_medium.png"

    with patch(
        "uma.viz.cv2.resize"
    ) as mock_resize:
        save_titleblock_crop(
            gray,
            (100, 100, 900, 600),
            str(out),
            pad_ratio=0,
            max_width=1800,
        )

    # Width = 800, which is between
    # 700 and max_width.
    mock_resize.assert_not_called()


def test_save_titleblock_crop_draws_original_bbox(
    tmp_path,
):
    gray = make_gray(
        width=1000,
        height=800,
    )

    out = tmp_path / "crop_rect.png"

    with patch(
        "uma.viz.cv2.rectangle"
    ) as mock_rectangle:
        save_titleblock_crop(
            gray,
            (200, 200, 600, 500),
            str(out),
            pad_ratio=0,
        )

    mock_rectangle.assert_called_once()

    call = mock_rectangle.call_args

    assert call.args[1] == (0, 0)
    assert call.args[2] == (400, 300)
    assert call.args[3] == C_TITLE


# ============================================================
# draw_tokens
# ============================================================

def test_draw_tokens_basic(tmp_path):
    plain = make_gray()

    tokens = [
        make_token(
            (10, 20, 80, 40)
        ),
        make_token(
            (100, 100, 180, 130)
        ),
    ]

    out = tmp_path / "tokens_basic.png"

    result = draw_tokens(
        plain,
        tokens,
        str(out),
    )

    assert result == str(out)
    assert out.exists()


def test_draw_tokens_empty(tmp_path):
    plain = make_gray()

    out = tmp_path / "tokens_empty.png"

    result = draw_tokens(
        plain,
        [],
        str(out),
    )

    assert result == str(out)
    assert out.exists()


def test_draw_tokens_draws_each_bbox(tmp_path):
    plain = make_gray()

    tokens = [
        make_token(
            (10, 20, 80, 40)
        ),
        make_token(
            (100, 100, 180, 130)
        ),
    ]

    out = tmp_path / "bbox.png"

    with patch(
        "uma.viz.cv2.rectangle"
    ) as mock_rectangle:
        draw_tokens(
            plain,
            tokens,
            str(out),
        )

    # Two token boxes + white
    # information background.
    assert mock_rectangle.call_count == 3

    assert any(
        call.args[1] == (10, 20)
        and call.args[2] == (80, 40)
        and call.args[3] == (60, 160, 240)
        for call in mock_rectangle.call_args_list
    )


def test_draw_tokens_displays_count(tmp_path):
    plain = make_gray()

    tokens = [
        make_token(),
        make_token(),
        make_token(),
    ]

    out = tmp_path / "count.png"

    texts = []

    def fake_text(
        img,
        text,
        *args,
        **kwargs,
    ):
        texts.append(text)
        return img

    with patch(
        "uma.viz.cv2.putText",
        side_effect=fake_text,
    ):
        draw_tokens(
            plain,
            tokens,
            str(out),
        )

    assert "ocr tokens: 3" in texts


def test_draw_tokens_empty_count(tmp_path):
    plain = make_gray()

    out = tmp_path / "count_zero.png"

    texts = []

    def fake_text(
        img,
        text,
        *args,
        **kwargs,
    ):
        texts.append(text)
        return img

    with patch(
        "uma.viz.cv2.putText",
        side_effect=fake_text,
    ):
        draw_tokens(
            plain,
            [],
            str(out),
        )

    assert "ocr tokens: 0" in texts


def test_draw_tokens_no_resize_small(tmp_path):
    plain = make_gray(
        width=500,
        height=300,
    )

    out = tmp_path / "tokens_small.png"

    with patch(
        "uma.viz.cv2.resize"
    ) as mock_resize:
        draw_tokens(
            plain,
            [],
            str(out),
            max_width=1000,
        )

    mock_resize.assert_not_called()


def test_draw_tokens_resize_large(tmp_path):
    plain = make_gray(
        width=1000,
        height=500,
    )

    out = tmp_path / "tokens_large.png"

    with patch(
        "uma.viz.cv2.resize",
        wraps=cv2.resize,
    ) as mock_resize:
        draw_tokens(
            plain,
            [],
            str(out),
            max_width=500,
        )

    mock_resize.assert_called_once()

    assert (
        mock_resize.call_args.args[1]
        == (500, 250)
    )

    assert (
        mock_resize.call_args.kwargs[
            "interpolation"
        ]
        == cv2.INTER_AREA
    )


# ============================================================
# imwrite / return behavior
# ============================================================

def test_draw_calls_imwrite(tmp_path):
    gray = make_gray()

    out = tmp_path / "write.png"

    with patch(
        "uma.viz.cv2.imwrite",
        return_value=True,
    ) as mock_write:
        result = draw(
            gray,
            [],
            [],
            None,
            str(out),
        )

    mock_write.assert_called_once()

    assert (
        mock_write.call_args.args[0]
        == str(out)
    )

    assert result == str(out)


def test_draw_v2_calls_imwrite(tmp_path):
    plain = make_gray()

    locate_result = make_locate_result()

    out = tmp_path / "write_v2.png"

    with (
        patch(
            "uma.locate.line_masks",
            return_value=(
                np.zeros_like(plain),
                np.zeros_like(plain),
            ),
        ),
        patch(
            "uma.locate.line_positions",
            return_value=(
                [],
                [],
            ),
        ),
        patch(
            "uma.viz.cv2.imwrite",
            return_value=True,
        ) as mock_write,
    ):
        result = draw_v2(
            plain,
            [],
            locate_result,
            str(out),
        )

    mock_write.assert_called_once()

    assert result == str(out)


def test_save_titleblock_crop_calls_imwrite(
    tmp_path,
):
    gray = make_gray(
        width=1000,
        height=800,
    )

    out = tmp_path / "write_crop.png"

    with patch(
        "uma.viz.cv2.imwrite",
        return_value=True,
    ) as mock_write:
        result = save_titleblock_crop(
            gray,
            (100, 100, 900, 600),
            str(out),
            pad_ratio=0,
        )

    mock_write.assert_called_once()

    assert result == str(out)


def test_draw_tokens_calls_imwrite(tmp_path):
    plain = make_gray()

    out = tmp_path / "write_tokens.png"

    with patch(
        "uma.viz.cv2.imwrite",
        return_value=True,
    ) as mock_write:
        result = draw_tokens(
            plain,
            [],
            str(out),
        )

    mock_write.assert_called_once()

    assert result == str(out)