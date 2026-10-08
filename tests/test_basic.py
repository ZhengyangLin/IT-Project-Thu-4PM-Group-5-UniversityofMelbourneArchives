from pathlib import Path
import importlib
import py_compile

import pytest

from uma.config import PreprocessCfg
from uma.preprocess import run


ROOT_DIR = Path(__file__).parent.parent
UMA_DIR = ROOT_DIR / "uma"
FIXTURES_DIR = Path(__file__).parent / "fixtrues"

def test_all_python_files_compile():
    python_files = list(UMA_DIR.rglob("*.py"))

    assert len(python_files) > 0, "No Python files found in uma/"

    for file_path in python_files:
        py_compile.compile(
            str(file_path),
            doraise=True
        )


def get_uma_modules():
    modules = []

    for file_path in UMA_DIR.rglob("*.py"):

        # Ignore __init__.py
        if file_path.name == "__init__.py":
            continue

        relative = file_path.relative_to(ROOT_DIR)

        module_name = ".".join(
            relative.with_suffix("").parts
        )

        modules.append(module_name)

    return modules


@pytest.mark.parametrize(
    "module_name",
    get_uma_modules()
)
def test_all_modules_import(module_name):

    module = importlib.import_module(module_name)

    assert module is not None


@pytest.mark.parametrize(
    "image_name",
    [
        "drawing1.jpg",
        "drawing2.jpg",
    ]
)
def test_ci_images_exist(image_name):

    image_path = FIXTURES_DIR / image_name

    assert image_path.exists(), (
        f"Test image not found: {image_path}"
    )

    assert image_path.stat().st_size > 0, (
        f"Test image is empty: {image_path}"
    )


@pytest.mark.parametrize(
    "image_name",
    [
        "drawing1.jpg",
        "drawing2.jpg",
    ]
)
def test_preprocessing_runs(image_name):

    image_path = FIXTURES_DIR / image_name

    assert image_path.exists()

    cfg = PreprocessCfg()

    enhanced, gray, meta = run(
        str(image_path),
        cfg
    )

    assert enhanced is not None
    assert gray is not None
    assert meta is not None
