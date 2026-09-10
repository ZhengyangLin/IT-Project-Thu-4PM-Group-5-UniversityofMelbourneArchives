from pathlib import Path
import pytest



TEST_DIR = Path(__file__).parent


FIXTURES_DIR = TEST_DIR / "fixtrues"


@pytest.mark.parametrize(
    "image_name",
    [
        "drawing1.jpg",
        "drawing2.jpg",
    ]
)
def test_drawing_exists(image_name):
    """Test that the drawing exists."""

    image_path = FIXTURES_DIR / image_name

    assert image_path.exists(), f"Image not found: {image_path}"


@pytest.mark.parametrize(
    "image_name",
    [
        "drawing1.jpg",
        "drawing2.jpg",
    ]
)
def test_drawing_not_empty(image_name):
    """Test that the drawing file is not empty."""

    image_path = FIXTURES_DIR / image_name

    assert image_path.exists(), f"Image not found: {image_path}"
    assert image_path.stat().st_size > 0, f"Image is empty: {image_path}"
