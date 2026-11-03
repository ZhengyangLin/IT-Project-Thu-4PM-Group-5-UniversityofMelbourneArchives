from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from uma.triage import (
    RELIABLE,
    HOPELESS,
    IMG_EXT,
    PROBE_ANCHORS,
    char_height,
    verdict,
    noise_ratio,
    assess,
    probe_anchors,
    sha256,
    infer_series,
    build_manifest,
    summarize,
)


# ============================================================
# Constants
# ============================================================

def test_constants():
    assert RELIABLE == 20
    assert HOPELESS == 8

    assert ".png" in IMG_EXT
    assert ".jpg" in IMG_EXT
    assert ".jpeg" in IMG_EXT
    assert ".tif" in IMG_EXT
    assert ".tiff" in IMG_EXT
    assert ".bmp" in IMG_EXT
    assert ".webp" in IMG_EXT

    assert "SCALE" in PROBE_ANCHORS
    assert "DATE" in PROBE_ANCHORS
    assert "DRAWN" in PROBE_ANCHORS


# ============================================================
# verdict
# ============================================================

@pytest.mark.parametrize(
    "height,expected",
    [
        (30.0, "reliable"),
        (20.0, "reliable"),
        (19.9, "marginal"),
        (15.0, "marginal"),
        (8.0, "marginal"),
        (7.9, "unrecoverable"),
        (0.0, "unrecoverable"),
        (-1.0, "unrecoverable"),
    ],
)
def test_verdict(height, expected):
    assert verdict(height) == expected


# ============================================================
# noise_ratio
# ============================================================

def test_noise_ratio_empty():
    assert noise_ratio([]) == 1.0


def test_noise_ratio_all_junk():
    texts = [
        "A",
        "AB",
        "XYZ",
        "!",
    ]

    assert noise_ratio(texts) == 1.0


def test_noise_ratio_no_junk():
    texts = [
        "DRAWING",
        "SCALE",
        "DATE",
        "PLAN",
    ]

    assert noise_ratio(texts) == 0.0


def test_noise_ratio_short_digits_not_junk():
    texts = [
        "1",
        "12",
        "A1",
        "99",
    ]

    assert noise_ratio(texts) == 0.0


def test_noise_ratio_mixed():
    texts = [
        "A",          # junk
        "AB",         # junk
        "DRAWING",    # not junk
        "12",         # digit -> not junk
    ]

    assert noise_ratio(texts) == pytest.approx(0.5)


def test_noise_ratio_whitespace_short():
    texts = [
        " A ",
        " LONG ",
    ]

    assert noise_ratio(texts) == pytest.approx(0.5)


def test_noise_ratio_empty_strings():
    texts = [
        "",
        " ",
        "DRAWING",
    ]

    assert noise_ratio(texts) == pytest.approx(2 / 3)


# ============================================================
# probe_anchors
# ============================================================

def test_probe_anchors_empty():
    assert probe_anchors([]) == 0


def test_probe_anchors_single():
    assert probe_anchors(
        ["Scale 1:100"]
    ) == 1


def test_probe_anchors_case_insensitive():
    result = probe_anchors([
        "scale",
        "date",
        "drawn",
    ])

    assert result == 3


def test_probe_anchors_multiple_in_one_string():
    result = probe_anchors([
        "DRAWN BY JOHN DATE 2024 SCALE 1:100"
    ])

    assert result >= 3


def test_probe_anchors_all():
    text = " ".join(PROBE_ANCHORS)

    assert probe_anchors([text]) == len(
        PROBE_ANCHORS
    )


def test_probe_anchors_unknown():
    assert probe_anchors([
        "HELLO WORLD"
    ]) == 0


def test_probe_anchor_counted_once():
    result = probe_anchors([
        "SCALE SCALE SCALE"
    ])

    # probe_anchors counts matching keywords,
    # not the number of occurrences.
    assert result == 1


# ============================================================
# char_height
# ============================================================

def test_char_height_empty_roi():
    gray = np.zeros(
        (100, 100),
        dtype=np.uint8,
    )

    # roi produces an empty slice.
    result, count = char_height(
        gray,
        roi=(1.0, 1.0, 1.0, 1.0),
    )

    assert result == 0.0
    assert count == 0


def test_char_height_no_components():
    gray = np.full(
        (200, 200),
        255,
        dtype=np.uint8,
    )

    height, count = char_height(
        gray,
        roi=(0, 0, 1, 1),
    )

    assert height == 0.0
    assert count < 10


def test_char_height_fewer_than_10_components():
    gray = np.full(
        (300, 300),
        255,
        dtype=np.uint8,
    )

    # Draw only a few valid components.
    for i in range(5):
        x = 20 + i * 40

        cv2.rectangle(
            gray,
            (x, 100),
            (x + 10, 120),
            0,
            -1,
        )

    height, count = char_height(
        gray,
        roi=(0, 0, 1, 1),
    )

    assert height == 0.0
    assert count < 10


def test_char_height_valid_components():
    gray = np.full(
        (500, 500),
        255,
        dtype=np.uint8,
    )

    # Create 12 separated components.
    for i in range(12):
        row = i // 6
        col = i % 6

        x = 20 + col * 60
        y = 100 + row * 80

        cv2.rectangle(
            gray,
            (x, y),
            (x + 10, y + 24),
            0,
            -1,
        )

    height, count = char_height(
        gray,
        roi=(0, 0, 1, 1),
    )

    assert count >= 10
    assert height > 0


def test_char_height_mocked_valid_stats():
    gray = np.zeros(
        (100, 100),
        dtype=np.uint8,
    )

    # connectedComponentsWithStats includes
    # background as the first row.
    background = [
        0,
        0,
        100,
        100,
        10000,
    ]

    components = []

    for i in range(10):
        components.append([
            i * 5,
            10,
            5,
            20 + i,
            100,
        ])

    stats = np.array(
        [background] + components,
        dtype=np.int32,
    )

    with patch(
        "uma.triage.cv2.connectedComponentsWithStats",
        return_value=(
            11,
            None,
            stats,
            None,
        ),
    ):
        height, count = char_height(
            gray,
            roi=(0, 0, 1, 1),
        )

    assert count == 10

    expected = float(
        np.percentile(
            [20 + i for i in range(10)],
            90,
        )
    )

    assert height == pytest.approx(
        expected
    )


def test_char_height_rejects_too_small():
    gray = np.zeros(
        (100, 100),
        dtype=np.uint8,
    )

    background = [
        0,
        0,
        100,
        100,
        10000,
    ]

    # height < 3
    bad = [
        1,
        1,
        5,
        2,
        10,
    ]

    stats = np.array(
        [background] + [bad] * 10,
        dtype=np.int32,
    )

    with patch(
        "uma.triage.cv2.connectedComponentsWithStats",
        return_value=(
            11,
            None,
            stats,
            None,
        ),
    ):
        height, count = char_height(
            gray,
            roi=(0, 0, 1, 1),
        )

    assert height == 0.0
    assert count == 0


def test_char_height_rejects_too_large():
    gray = np.zeros(
        (100, 100),
        dtype=np.uint8,
    )

    background = [
        0,
        0,
        100,
        100,
        10000,
    ]

    bad = [
        1,
        1,
        20,
        121,
        2000,
    ]

    stats = np.array(
        [background] + [bad] * 10,
        dtype=np.int32,
    )

    with patch(
        "uma.triage.cv2.connectedComponentsWithStats",
        return_value=(
            11,
            None,
            stats,
            None,
        ),
    ):
        height, count = char_height(
            gray,
            roi=(0, 0, 1, 1),
        )

    assert height == 0.0
    assert count == 0


def test_char_height_rejects_low_fill_ratio():
    gray = np.zeros(
        (100, 100),
        dtype=np.uint8,
    )

    background = [
        0,
        0,
        100,
        100,
        10000,
    ]

    # w*h = 400
    # 15% = 60
    # area=10 -> reject
    bad = [
        1,
        1,
        20,
        20,
        10,
    ]

    stats = np.array(
        [background] + [bad] * 10,
        dtype=np.int32,
    )

    with patch(
        "uma.triage.cv2.connectedComponentsWithStats",
        return_value=(
            11,
            None,
            stats,
            None,
        ),
    ):
        height, count = char_height(
            gray,
            roi=(0, 0, 1, 1),
        )

    assert height == 0.0
    assert count == 0


def test_char_height_rejects_extreme_width_ratio():
    gray = np.zeros(
        (100, 100),
        dtype=np.uint8,
    )

    background = [
        0,
        0,
        100,
        100,
        10000,
    ]

    # w/h > 8
    bad = [
        1,
        1,
        100,
        10,
        500,
    ]

    stats = np.array(
        [background] + [bad] * 10,
        dtype=np.int32,
    )

    with patch(
        "uma.triage.cv2.connectedComponentsWithStats",
        return_value=(
            11,
            None,
            stats,
            None,
        ),
    ):
        height, count = char_height(
            gray,
            roi=(0, 0, 1, 1),
        )

    assert height == 0.0
    assert count == 0


def test_char_height_rejects_extreme_height_ratio():
    gray = np.zeros(
        (100, 100),
        dtype=np.uint8,
    )

    background = [
        0,
        0,
        100,
        100,
        10000,
    ]

    # h/w > 8
    bad = [
        1,
        1,
        5,
        50,
        150,
    ]

    stats = np.array(
        [background] + [bad] * 10,
        dtype=np.int32,
    )

    with patch(
        "uma.triage.cv2.connectedComponentsWithStats",
        return_value=(
            11,
            None,
            stats,
            None,
        ),
    ):
        height, count = char_height(
            gray,
            roi=(0, 0, 1, 1),
        )

    assert height == 0.0
    assert count == 0


# ============================================================
# assess
# ============================================================

def test_assess_reliable():
    gray = np.zeros(
        (600, 800),
        dtype=np.uint8,
    )

    with patch(
        "uma.triage.char_height",
        return_value=(25.46, 50),
    ):
        result = assess(gray)

    assert result[
        "char_height_p90_px"
    ] == 25.5

    assert result["components"] == 50
    assert result["verdict"] == "reliable"

    assert (
        result["note"]
        == "Key fields should be legible"
    )

    assert result["width"] == 800
    assert result["height"] == 600


def test_assess_marginal():
    gray = np.zeros(
        (100, 200),
        dtype=np.uint8,
    )

    with patch(
        "uma.triage.char_height",
        return_value=(12.0, 20),
    ):
        result = assess(gray)

    assert result["verdict"] == "marginal"

    assert (
        "Large text is legible"
        in result["note"]
    )

    assert result["width"] == 200
    assert result["height"] == 100


def test_assess_unrecoverable():
    gray = np.zeros(
        (300, 400),
        dtype=np.uint8,
    )

    with patch(
        "uma.triage.char_height",
        return_value=(5.0, 2),
    ):
        result = assess(gray)

    assert (
        result["verdict"]
        == "unrecoverable"
    )

    assert (
        "severely downsampled"
        in result["note"]
    )


# ============================================================
# sha256
# ============================================================

def test_sha256_known_content(tmp_path):
    path = tmp_path / "test.txt"

    path.write_bytes(
        b"hello world"
    )

    result = sha256(path)

    assert result == (
        "b94d27b9934d3e08"
        "a52e52d7da7dabfa"
        "c484efe37a5380ee"
        "9088f7ace2efcde9"
    )


def test_sha256_empty_file(tmp_path):
    path = tmp_path / "empty.bin"

    path.write_bytes(b"")

    result = sha256(path)

    assert result == (
        "e3b0c44298fc1c14"
        "9afbf4c8996fb924"
        "27ae41e4649b934c"
        "a495991b7852b855"
    )


def test_sha256_small_chunks(tmp_path):
    path = tmp_path / "chunks.bin"

    path.write_bytes(
        b"abcdefghijk"
    )

    normal = sha256(path)

    chunked = sha256(
        path,
        chunk=2,
    )

    assert chunked == normal


# ============================================================
# infer_series
# ============================================================

@pytest.mark.parametrize(
    "path,expected",
    [
        (
            Path("root/BSM/file.png"),
            "BSM",
        ),
        (
            Path("root/bsm/file.png"),
            "BSM",
        ),
        (
            Path("root/CABANA/file.png"),
            "YFA-Cabana",
        ),
        (
            Path("root/cabana/file.png"),
            "YFA-Cabana",
        ),
        (
            Path("root/STATE/file.png"),
            "YFA-StateGov",
        ),
        (
            Path("root/state/file.png"),
            "YFA-StateGov",
        ),
        (
            Path("root/YFA/file.png"),
            "YFA",
        ),
        (
            Path("root/yfa/file.png"),
            "YFA",
        ),
        (
            Path("root/OTHER/file.png"),
            "unknown",
        ),
    ],
)
def test_infer_series(
    path,
    expected,
):
    assert infer_series(path) == expected


def test_infer_series_priority_bsm():
    path = Path(
        "YFA/STATE/CABANA/BSM/file.png"
    )

    # BSM is checked first.
    assert infer_series(path) == "BSM"


def test_infer_series_priority_cabana():
    path = Path(
        "YFA/STATE/CABANA/file.png"
    )

    assert (
        infer_series(path)
        == "YFA-Cabana"
    )


def test_infer_series_priority_state():
    path = Path(
        "YFA/STATE/file.png"
    )

    assert (
        infer_series(path)
        == "YFA-StateGov"
    )


# ============================================================
# build_manifest
# ============================================================

def test_build_manifest_empty_directory(
    tmp_path,
):
    result = build_manifest(
        tmp_path
    )

    assert result == []


def test_build_manifest_ignores_non_images(
    tmp_path,
):
    (tmp_path / "test.txt").write_text(
        "hello",
        encoding="utf-8",
    )

    (tmp_path / "data.csv").write_text(
        "a,b,c",
        encoding="utf-8",
    )

    result = build_manifest(
        tmp_path
    )

    assert result == []


def test_build_manifest_valid_image(
    tmp_path,
):
    folder = (
        tmp_path
        / "BSM"
        / "NODE001"
    )

    folder.mkdir(
        parents=True
    )

    path = (
        folder
        / "drawing.png"
    )

    image = np.full(
        (100, 200),
        255,
        dtype=np.uint8,
    )

    assert cv2.imwrite(
        str(path),
        image,
    )

    with patch(
        "uma.triage.assess",
        return_value={
            "char_height_p90_px": 25.0,
            "components": 20,
            "verdict": "reliable",
            "note": "test",
            "width": 200,
            "height": 100,
        },
    ):
        result = build_manifest(
            tmp_path
        )

    assert len(result) == 1

    row = result[0]

    assert row[
        "node_id"
    ] == "NODE001"

    assert row[
        "path"
    ] == str(path)

    assert len(
        row["sha256"]
    ) == 16

    assert row[
        "series"
    ] == "BSM"

    assert row[
        "filesize_kb"
    ] >= 0

    assert row[
        "verdict"
    ] == "reliable"

    assert row[
        "width"
    ] == 200

    assert row[
        "height"
    ] == 100


def test_build_manifest_unreadable_image(
    tmp_path,
):
    folder = (
        tmp_path
        / "NODE_BAD"
    )

    folder.mkdir()

    path = (
        folder
        / "bad.png"
    )

    # Has a supported extension,
    # but is not actually an image.
    path.write_bytes(
        b"not a real image"
    )

    result = build_manifest(
        tmp_path
    )

    assert len(result) == 1

    row = result[0]

    assert (
        row["verdict"]
        == "unreadable_file"
    )

    assert (
        row["node_id"]
        == "NODE_BAD"
    )


def test_build_manifest_multiple_formats(
    tmp_path,
):
    folder = (
        tmp_path
        / "NODE"
    )

    folder.mkdir()

    image = np.full(
        (20, 20),
        255,
        dtype=np.uint8,
    )

    png = folder / "a.png"
    jpg = folder / "b.jpg"
    bmp = folder / "c.bmp"

    assert cv2.imwrite(
        str(png),
        image,
    )

    assert cv2.imwrite(
        str(jpg),
        image,
    )

    assert cv2.imwrite(
        str(bmp),
        image,
    )

    with patch(
        "uma.triage.assess",
        return_value={
            "verdict": "reliable",
        },
    ):
        result = build_manifest(
            tmp_path
        )

    assert len(result) == 3


def test_build_manifest_sorted_files(
    tmp_path,
):
    folder = (
        tmp_path
        / "NODE"
    )

    folder.mkdir()

    image = np.full(
        (20, 20),
        255,
        dtype=np.uint8,
    )

    assert cv2.imwrite(
        str(folder / "z.png"),
        image,
    )

    assert cv2.imwrite(
        str(folder / "a.png"),
        image,
    )

    with patch(
        "uma.triage.assess",
        return_value={
            "verdict": "reliable",
        },
    ):
        result = build_manifest(
            tmp_path
        )

    assert len(result) == 2

    assert result[0][
        "path"
    ].endswith("a.png")

    assert result[1][
        "path"
    ].endswith("z.png")


def test_build_manifest_sample_parameter_does_not_change_current_behavior(
    tmp_path,
):
    folder = tmp_path / "NODE"
    folder.mkdir()

    image = np.full(
        (20, 20),
        255,
        dtype=np.uint8,
    )

    for i in range(3):
        assert cv2.imwrite(
            str(
                folder
                / f"{i}.png"
            ),
            image,
        )

    with patch(
        "uma.triage.assess",
        return_value={
            "verdict": "reliable",
        },
    ):
        result = build_manifest(
            tmp_path,
            sample=1,
        )

    # In the current triage.py implementation,
    # sample is accepted but not used to limit files.
    assert len(result) == 3


# ============================================================
# summarize
# ============================================================

def test_summarize_empty():
    result = summarize([])

    assert result == {
        "total": 0,
        "series": {},
    }


def test_summarize_single_series():
    rows = [
        {
            "series": "BSM",
            "verdict": "reliable",
            "char_height_p90_px": 25.0,
        },
        {
            "series": "BSM",
            "verdict": "marginal",
            "char_height_p90_px": 15.0,
        },
        {
            "series": "BSM",
            "verdict": "unrecoverable",
            "char_height_p90_px": 5.0,
        },
    ]

    result = summarize(rows)

    assert result["total"] == 3

    bsm = result[
        "series"
    ]["BSM"]

    assert bsm["count"] == 3
    assert bsm["reliable"] == 1
    assert bsm["marginal"] == 1
    assert bsm["unrecoverable"] == 1

    assert (
        bsm["char_height_p90_median"]
        == 15.0
    )

    assert (
        bsm["unrecoverable_pct"]
        == pytest.approx(33.3)
    )


def test_summarize_multiple_series():
    rows = [
        {
            "series": "BSM",
            "verdict": "reliable",
            "char_height_p90_px": 25,
        },
        {
            "series": "YFA",
            "verdict": "marginal",
            "char_height_p90_px": 12,
        },
    ]

    result = summarize(rows)

    assert result["total"] == 2

    assert (
        result["series"]["BSM"]["count"]
        == 1
    )

    assert (
        result["series"]["YFA"]["count"]
        == 1
    )


def test_summarize_missing_series():
    rows = [
        {
            "verdict": "reliable",
            "char_height_p90_px": 25,
        }
    ]

    result = summarize(rows)

    assert (
        "unknown"
        in result["series"]
    )


def test_summarize_missing_char_height():
    rows = [
        {
            "series": "TEST",
            "verdict": "reliable",
        }
    ]

    result = summarize(rows)

    assert (
        result["series"]["TEST"][
            "char_height_p90_median"
        ]
        == 0
    )


def test_summarize_all_unrecoverable():
    rows = [
        {
            "series": "TEST",
            "verdict": "unrecoverable",
            "char_height_p90_px": 4,
        },
        {
            "series": "TEST",
            "verdict": "unrecoverable",
            "char_height_p90_px": 6,
        },
    ]

    result = summarize(rows)

    test = result[
        "series"
    ]["TEST"]

    assert test[
        "unrecoverable"
    ] == 2

    assert test[
        "unrecoverable_pct"
    ] == 100.0

    assert test[
        "char_height_p90_median"
    ] == 5.0


def test_summarize_no_unrecoverable():
    rows = [
        {
            "series": "TEST",
            "verdict": "reliable",
            "char_height_p90_px": 30,
        },
        {
            "series": "TEST",
            "verdict": "marginal",
            "char_height_p90_px": 10,
        },
    ]

    result = summarize(rows)

    assert (
        result["series"]["TEST"][
            "unrecoverable_pct"
        ]
        == 0.0
    )