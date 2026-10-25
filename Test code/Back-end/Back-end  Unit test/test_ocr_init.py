import uma.ocr as ocr

from uma.ocr.engines import (
    build_engines,
    Engine,
    RawToken,
)

from uma.ocr.fusion import (
    read_all,
    fuse,
)


def test_exports_build_engines():
    assert ocr.build_engines is build_engines


def test_exports_engine():
    assert ocr.Engine is Engine


def test_exports_raw_token():
    assert ocr.RawToken is RawToken


def test_exports_read_all():
    assert ocr.read_all is read_all


def test_exports_fuse():
    assert ocr.fuse is fuse


def test_all_exports():
    assert ocr.__all__ == [
        "build_engines",
        "Engine",
        "RawToken",
        "read_all",
        "fuse",
    ]


def test_all_names_exist():
    for name in ocr.__all__:
        assert hasattr(ocr, name)