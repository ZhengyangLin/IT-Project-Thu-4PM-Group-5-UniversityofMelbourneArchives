import sys
from types import (
    ModuleType,
    SimpleNamespace,
)
from unittest.mock import (
    Mock,
    patch,
)

import numpy as np
import pytest

from uma.ocr.engines import (
    RawToken,
    Engine,
    PaddleEngine,
    TesseractEngine,
    DoctrEngine,
    _check_extent,
    build_engines,
)


# ============================================================
# RawToken
# ============================================================

def test_raw_token():
    token = RawToken(
        text="ABC",
        bbox=(1, 2, 3, 4),
        conf=0.9,
        engine="test",
    )

    assert token.text == "ABC"
    assert token.bbox == (1, 2, 3, 4)
    assert token.conf == 0.9
    assert token.engine == "test"


# ============================================================
# Engine base class
# ============================================================

def test_engine_available_not_implemented():
    with pytest.raises(
        NotImplementedError
    ):
        Engine().available()


def test_engine_read_not_implemented():
    gray = np.zeros(
        (10, 10),
        dtype=np.uint8,
    )

    with pytest.raises(
        NotImplementedError
    ):
        Engine().read(gray)


# ============================================================
# _scale_up
# ============================================================

def test_scale_up_factor_one_returns_original():
    gray = np.zeros(
        (10, 20),
        dtype=np.uint8,
    )

    result, eff = Engine._scale_up(
        gray,
        1.0,
    )

    assert result is gray
    assert eff == 1.0


def test_scale_up_enlarge():
    gray = np.zeros(
        (10, 20),
        dtype=np.uint8,
    )

    result, eff = Engine._scale_up(
        gray,
        2.0,
    )

    assert result.shape == (
        20,
        40,
    )

    assert eff == 2.0


def test_scale_up_shrink():
    gray = np.zeros(
        (20, 40),
        dtype=np.uint8,
    )

    result, eff = Engine._scale_up(
        gray,
        0.5,
    )

    assert result.shape == (
        10,
        20,
    )

    assert eff == 0.5


def test_scale_up_max_side_limits_factor():
    gray = np.zeros(
        (100, 200),
        dtype=np.uint8,
    )

    result, eff = Engine._scale_up(
        gray,
        2.0,
        max_side=200,
    )

    assert eff == pytest.approx(
        1.0
    )

    assert result is gray


def test_scale_up_max_side_resize():
    gray = np.zeros(
        (100, 200),
        dtype=np.uint8,
    )

    result, eff = Engine._scale_up(
        gray,
        4.0,
        max_side=400,
    )

    assert eff == pytest.approx(
        2.0
    )

    assert result.shape == (
        200,
        400,
    )


# ============================================================
# _scale_box
# ============================================================

def test_scale_box():
    box = Engine._scale_box(
        (
            10,
            20,
            31,
            41,
        ),
        2.0,
    )

    assert box == (
        5,
        10,
        16,
        21,
    )


def test_scale_box_factor_one():
    assert Engine._scale_box(
        (1, 2, 3, 4),
        1.0,
    ) == (
        1,
        2,
        3,
        4,
    )


# ============================================================
# PaddleEngine init
# ============================================================

def test_paddle_init():
    engine = PaddleEngine(
        lang="en",
        use_angle=False,
        max_side_limit=3000,
    )

    assert engine.name == "paddle"
    assert engine._ocr is None
    assert engine._lang == "en"
    assert engine._use_angle is False
    assert engine._max_side == 3000


# ============================================================
# Paddle _lazy
# ============================================================

def test_paddle_lazy_returns_existing():
    engine = PaddleEngine()

    fake = object()

    engine._ocr = fake

    assert engine._lazy() is fake


def test_paddle_lazy_first_attempt_success():
    fake_module = ModuleType(
        "paddleocr"
    )

    fake_ocr = object()
    constructor = Mock(
        return_value=fake_ocr
    )

    fake_module.PaddleOCR = constructor

    engine = PaddleEngine()

    with patch.dict(
        sys.modules,
        {
            "paddleocr": fake_module,
        },
    ):
        result = engine._lazy()

    assert result is fake_ocr
    assert engine._ocr is fake_ocr
    assert engine._api == "v3"

    assert constructor.call_count == 1


def test_paddle_lazy_falls_back():
    fake_module = ModuleType(
        "paddleocr"
    )

    fake_ocr = object()

    constructor = Mock(
        side_effect=[
            TypeError("1"),
            TypeError("2"),
            TypeError("3"),
            fake_ocr,
        ]
    )

    fake_module.PaddleOCR = constructor

    engine = PaddleEngine()

    with patch.dict(
        sys.modules,
        {
            "paddleocr": fake_module,
        },
    ):
        result = engine._lazy()

    assert result is fake_ocr
    assert constructor.call_count == 4
    assert engine._api == "v2"


def test_paddle_lazy_all_attempts_fail():
    fake_module = ModuleType(
        "paddleocr"
    )

    fake_module.PaddleOCR = Mock(
        side_effect=TypeError(
            "unsupported"
        )
    )

    engine = PaddleEngine()

    with patch.dict(
        sys.modules,
        {
            "paddleocr": fake_module,
        },
    ):
        with pytest.raises(
            RuntimeError,
            match="initialization failed",
        ):
            engine._lazy()


# ============================================================
# Paddle available
# ============================================================

def test_paddle_available_true():
    fake_module = ModuleType(
        "paddleocr"
    )

    with patch.dict(
        sys.modules,
        {
            "paddleocr": fake_module,
        },
    ):
        assert (
            PaddleEngine().available()
            is True
        )


def test_paddle_available_false():
    engine = PaddleEngine()

    original_import = __import__

    def fake_import(
        name,
        *args,
        **kwargs,
    ):
        if name == "paddleocr":
            raise ImportError(
                "missing"
            )

        return original_import(
            name,
            *args,
            **kwargs,
        )

    with patch(
        "builtins.__import__",
        side_effect=fake_import,
    ):
        assert (
            engine.available()
            is False
        )


# ============================================================
# Paddle _parse
# ============================================================

def test_paddle_parse_empty():
    engine = PaddleEngine()

    assert engine._parse(
        None,
        1.0,
    ) == []


def test_paddle_parse_old_style():
    engine = PaddleEngine()

    res = [[
        [
            [
                [10, 20],
                [30, 20],
                [30, 40],
                [10, 40],
            ],
            (
                "HELLO",
                0.9,
            ),
        ]
    ]]

    tokens = engine._parse(
        res,
        1.0,
    )

    assert len(tokens) == 1

    assert tokens[0].text == "HELLO"
    assert tokens[0].bbox == (
        10,
        20,
        30,
        40,
    )
    assert tokens[0].conf == 0.9
    assert tokens[0].engine == "paddle"


def test_paddle_parse_scales_coordinates():
    engine = PaddleEngine()

    res = [[
        [
            [
                [20, 40],
                [60, 40],
                [60, 80],
                [20, 80],
            ],
            (
                "TEST",
                0.8,
            ),
        ]
    ]]

    tokens = engine._parse(
        res,
        2.0,
    )

    assert tokens[0].bbox == (
        10,
        20,
        30,
        40,
    )


def test_paddle_parse_ignores_blank_text():
    engine = PaddleEngine()

    res = [[
        [
            [
                [0, 0],
                [10, 0],
                [10, 10],
                [0, 10],
            ],
            (
                "   ",
                0.9,
            ),
        ]
    ]]

    assert engine._parse(
        res,
        1.0,
    ) == []


def test_paddle_parse_dict():
    engine = PaddleEngine()

    res = [{
        "rec_texts": [
            "ABC",
            "DEF",
        ],
        "rec_scores": [
            0.9,
            0.8,
        ],
        "rec_polys": [
            [
                [0, 0],
                [20, 0],
                [20, 10],
                [0, 10],
            ],
            [
                [30, 20],
                [50, 20],
                [50, 40],
                [30, 40],
            ],
        ],
    }]

    tokens = engine._parse(
        res,
        1.0,
    )

    assert [
        t.text for t in tokens
    ] == [
        "ABC",
        "DEF",
    ]


def test_paddle_parse_json_object():
    engine = PaddleEngine()

    page = SimpleNamespace(
        json={
            "res": {
                "rec_texts": [
                    "ABC"
                ],
                "rec_scores": [
                    0.75
                ],
                "rec_polys": [[
                    [1, 2],
                    [10, 2],
                    [10, 8],
                    [1, 8],
                ]],
            }
        }
    )

    tokens = engine._parse(
        [page],
        1.0,
    )

    assert len(tokens) == 1
    assert tokens[0].text == "ABC"


def test_paddle_parse_skips_none_page():
    engine = PaddleEngine()

    assert engine._parse(
        [None],
        1.0,
    ) == []


def test_paddle_parse_skips_invalid_item():
    engine = PaddleEngine()

    res = [[
        "bad item"
    ]]

    assert engine._parse(
        res,
        1.0,
    ) == []


# ============================================================
# Paddle read
# ============================================================

def test_paddle_read_predict_success():
    gray = np.zeros(
        (100, 200),
        dtype=np.uint8,
    )

    engine = PaddleEngine()

    fake_ocr = Mock()
    fake_ocr.predict.return_value = []

    with (
        patch.object(
            engine,
            "_lazy",
            return_value=fake_ocr,
        ),
        patch.object(
            engine,
            "_parse",
            return_value=[],
        ) as mock_parse,
        patch(
            "uma.ocr.engines._check_extent"
        ) as mock_extent,
    ):
        result = engine.read(
            gray
        )

    assert result == []

    fake_ocr.predict.assert_called_once()

    mock_parse.assert_called_once()

    mock_extent.assert_called_once()


def test_paddle_read_predict_typeerror_fallback():
    gray = np.zeros(
        (50, 50),
        dtype=np.uint8,
    )

    engine = PaddleEngine()

    fake_ocr = Mock()

    fake_ocr.predict.side_effect = [
        TypeError("bad arg"),
        [],
    ]

    with (
        patch.object(
            engine,
            "_lazy",
            return_value=fake_ocr,
        ),
        patch.object(
            engine,
            "_parse",
            return_value=[],
        ),
    ):
        result = engine.read(
            gray
        )

    assert result == []
    assert (
        fake_ocr.predict.call_count
        == 2
    )


def test_paddle_read_ocr_fallback():
    gray = np.zeros(
        (50, 50),
        dtype=np.uint8,
    )

    engine = PaddleEngine()

    class FakeOCR:
        def ocr(
            self,
            *args,
            **kwargs,
        ):
            return []

    with (
        patch.object(
            engine,
            "_lazy",
            return_value=FakeOCR(),
        ),
        patch.object(
            engine,
            "_parse",
            return_value=[],
        ),
    ):
        result = engine.read(
            gray
        )

    assert result == []


def test_paddle_read_all_calls_fail():
    gray = np.zeros(
        (50, 50),
        dtype=np.uint8,
    )

    engine = PaddleEngine()

    fake_ocr = Mock()

    fake_ocr.predict.side_effect = (
        TypeError("bad")
    )

    fake_ocr.ocr.side_effect = (
        RuntimeError("failure")
    )

    with patch.object(
        engine,
        "_lazy",
        return_value=fake_ocr,
    ):
        result = engine.read(
            gray
        )

    assert result == []


def test_paddle_read_coordinate_error_path():
    gray = np.zeros(
        (100, 100),
        dtype=np.uint8,
    )

    engine = PaddleEngine()

    fake_ocr = Mock()
    fake_ocr.predict.return_value = []

    bad_token = RawToken(
        "BAD",
        (0, 0, 200, 200),
        0.9,
        "paddle",
    )

    with (
        patch.object(
            engine,
            "_lazy",
            return_value=fake_ocr,
        ),
        patch.object(
            engine,
            "_parse",
            return_value=[
                bad_token
            ],
        ),
        patch(
            "uma.ocr.engines.log.error"
        ) as mock_error,
    ):
        result = engine.read(
            gray
        )

    assert result == [
        bad_token
    ]

    assert mock_error.called


# ============================================================
# Tesseract
# ============================================================

def test_tesseract_init():
    engine = TesseractEngine(
        lang="eng",
        psms=(3, 6),
    )

    assert engine.name == "tesseract"
    assert engine._lang == "eng"
    assert engine._psms == (
        3,
        6,
    )


def test_tesseract_available_true():
    fake = ModuleType(
        "pytesseract"
    )

    fake.get_tesseract_version = (
        Mock(return_value="5")
    )

    with patch.dict(
        sys.modules,
        {
            "pytesseract": fake,
        },
    ):
        assert (
            TesseractEngine().available()
            is True
        )


def test_tesseract_available_false():
    fake = ModuleType(
        "pytesseract"
    )

    fake.get_tesseract_version = Mock(
        side_effect=RuntimeError(
            "missing binary"
        )
    )

    with patch.dict(
        sys.modules,
        {
            "pytesseract": fake,
        },
    ):
        assert (
            TesseractEngine().available()
            is False
        )


def test_tesseract_read():
    fake = ModuleType(
        "pytesseract"
    )

    fake.Output = SimpleNamespace(
        DICT="DICT"
    )

    fake.image_to_data = Mock(
        return_value={
            "text": [
                "HELLO",
                "",
                "BAD",
            ],
            "conf": [
                "90",
                "80",
                "-1",
            ],
            "left": [
                10,
                20,
                30,
            ],
            "top": [
                10,
                20,
                30,
            ],
            "width": [
                20,
                20,
                20,
            ],
            "height": [
                10,
                10,
                10,
            ],
        }
    )

    gray = np.zeros(
        (100, 100),
        dtype=np.uint8,
    )

    engine = TesseractEngine(
        psms=(6,)
    )

    with (
        patch.dict(
            sys.modules,
            {
                "pytesseract": fake,
            },
        ),
        patch(
            "uma.ocr.engines._check_extent"
        ),
    ):
        result = engine.read(
            gray
        )

    assert len(result) == 1

    assert result[0].text == "HELLO"
    assert result[0].conf == 0.9
    assert result[0].engine == "tesseract"


def test_tesseract_read_whitelist():
    fake = ModuleType(
        "pytesseract"
    )

    fake.Output = SimpleNamespace(
        DICT="DICT"
    )

    fake.image_to_data = Mock(
        return_value={
            "text": [],
            "conf": [],
            "left": [],
            "top": [],
            "width": [],
            "height": [],
        }
    )

    gray = np.zeros(
        (20, 20),
        dtype=np.uint8,
    )

    with patch.dict(
        sys.modules,
        {
            "pytesseract": fake,
        },
    ):
        TesseractEngine(
            psms=(6,)
        ).read(
            gray,
            whitelist="ABC123",
        )

    config = (
        fake.image_to_data
        .call_args.kwargs[
            "config"
        ]
    )

    assert "--psm 6" in config
    assert (
        "tessedit_char_whitelist=ABC123"
        in config
    )


def test_tesseract_duplicate_removed():
    fake = ModuleType(
        "pytesseract"
    )

    fake.Output = SimpleNamespace(
        DICT="DICT"
    )

    fake.image_to_data = Mock(
        return_value={
            "text": ["ABC"],
            "conf": ["90"],
            "left": [10],
            "top": [10],
            "width": [20],
            "height": [10],
        }
    )

    gray = np.zeros(
        (100, 100),
        dtype=np.uint8,
    )

    engine = TesseractEngine(
        psms=(6, 11)
    )

    with (
        patch.dict(
            sys.modules,
            {
                "pytesseract": fake,
            },
        ),
        patch(
            "uma.ocr.engines._check_extent"
        ),
    ):
        result = engine.read(
            gray
        )

    assert len(result) == 1


def test_tesseract_exception_skipped():
    fake = ModuleType(
        "pytesseract"
    )

    fake.Output = SimpleNamespace(
        DICT="DICT"
    )

    fake.image_to_data = Mock(
        side_effect=RuntimeError(
            "ocr failure"
        )
    )

    gray = np.zeros(
        (20, 20),
        dtype=np.uint8,
    )

    with patch.dict(
        sys.modules,
        {
            "pytesseract": fake,
        },
    ):
        result = TesseractEngine(
            psms=(6,)
        ).read(gray)

    assert result == []


# ============================================================
# DoctrEngine
# ============================================================

def test_doctr_init():
    engine = DoctrEngine()

    assert engine.name == "doctr"
    assert engine._model is None
    assert engine._failed is False


def test_doctr_lazy_success():
    doctr = ModuleType("doctr")
    models = ModuleType(
        "doctr.models"
    )

    model = object()

    models.ocr_predictor = Mock(
        return_value=model
    )

    doctr.models = models

    engine = DoctrEngine()

    with patch.dict(
        sys.modules,
        {
            "doctr": doctr,
            "doctr.models": models,
        },
    ):
        result = engine._lazy()

    assert result is model
    assert engine._model is model
    assert engine._failed is False


def test_doctr_lazy_failure():
    doctr = ModuleType("doctr")
    models = ModuleType(
        "doctr.models"
    )

    models.ocr_predictor = Mock(
        side_effect=RuntimeError(
            "download failed"
        )
    )

    doctr.models = models

    engine = DoctrEngine()

    with patch.dict(
        sys.modules,
        {
            "doctr": doctr,
            "doctr.models": models,
        },
    ):
        result = engine._lazy()

    assert result is None
    assert engine._failed is True


def test_doctr_lazy_does_not_retry_after_failure():
    engine = DoctrEngine()

    engine._failed = True

    assert engine._lazy() is None


def test_doctr_available_true():
    doctr = ModuleType(
        "doctr"
    )

    with patch.dict(
        sys.modules,
        {
            "doctr": doctr,
        },
    ):
        assert (
            DoctrEngine().available()
            is True
        )


def test_doctr_read_no_model():
    gray = np.zeros(
        (100, 100),
        dtype=np.uint8,
    )

    engine = DoctrEngine()

    with patch.object(
        engine,
        "_lazy",
        return_value=None,
    ):
        assert engine.read(
            gray
        ) == []


def test_doctr_read_model_exception():
    gray = np.zeros(
        (100, 100),
        dtype=np.uint8,
    )

    engine = DoctrEngine()

    model = Mock(
        side_effect=RuntimeError(
            "inference failed"
        )
    )

    with patch.object(
        engine,
        "_lazy",
        return_value=model,
    ):
        assert engine.read(
            gray
        ) == []


def test_doctr_read_success():
    gray = np.zeros(
        (100, 200),
        dtype=np.uint8,
    )

    word = SimpleNamespace(
        geometry=(
            (0.1, 0.2),
            (0.5, 0.4),
        ),
        value="HELLO",
        confidence=0.88,
    )

    line = SimpleNamespace(
        words=[word]
    )

    block = SimpleNamespace(
        lines=[line]
    )

    page = SimpleNamespace(
        blocks=[block]
    )

    response = SimpleNamespace(
        pages=[page]
    )

    model = Mock(
        return_value=response
    )

    engine = DoctrEngine()

    with (
        patch.object(
            engine,
            "_lazy",
            return_value=model,
        ),
        patch(
            "uma.ocr.engines._check_extent"
        ) as mock_extent,
    ):
        result = engine.read(
            gray
        )

    assert len(result) == 1

    token = result[0]

    assert token.text == "HELLO"
    assert token.bbox == (
        20,
        20,
        100,
        40,
    )
    assert token.conf == 0.88
    assert token.engine == "doctr"

    mock_extent.assert_called_once()


# ============================================================
# _check_extent
# ============================================================

def test_check_extent_empty():
    with patch(
        "uma.ocr.engines.log"
    ) as mock_log:
        _check_extent(
            [],
            (100, 100),
            "test",
        )

    mock_log.error.assert_not_called()
    mock_log.warning.assert_not_called()
    mock_log.info.assert_not_called()


def test_check_extent_out_of_bounds():
    tokens = [
        RawToken(
            "A",
            (0, 0, 120, 120),
            0.9,
            "test",
        )
    ]

    with patch(
        "uma.ocr.engines.log.error"
    ) as mock_error:
        _check_extent(
            tokens,
            (100, 100),
            "test",
        )

    mock_error.assert_called_once()


def test_check_extent_low_coverage():
    tokens = [
        RawToken(
            "A",
            (0, 0, 20, 20),
            0.9,
            "test",
        )
    ]

    with patch(
        "uma.ocr.engines.log.warning"
    ) as mock_warning:
        _check_extent(
            tokens,
            (100, 100),
            "test",
        )

    mock_warning.assert_called_once()


def test_check_extent_valid():
    tokens = [
        RawToken(
            "A",
            (0, 0, 80, 80),
            0.9,
            "test",
        )
    ]

    with patch(
        "uma.ocr.engines.log.info"
    ) as mock_info:
        _check_extent(
            tokens,
            (100, 100),
            "test",
        )

    mock_info.assert_called_once()


# ============================================================
# build_engines
# ============================================================

def test_build_engines_unknown_name():
    with pytest.raises(
        RuntimeError
    ):
        build_engines([
            "unknown"
        ])


def test_build_engines_paddle():
    fake_engine = Mock()
    fake_engine.available.return_value = True

    fake_cls = Mock(
        return_value=fake_engine
    )

    with patch.dict(
        "uma.ocr.engines.REGISTRY",
        {
            "paddle": fake_cls
        },
        clear=True,
    ):
        result = build_engines(
            ["paddle"],
            lang="fr",
            paddle_max_side=3000,
        )

    fake_cls.assert_called_once_with(
        lang="fr",
        max_side_limit=3000,
    )

    assert result == [
        fake_engine
    ]


def test_build_engines_tesseract():
    fake_engine = Mock()
    fake_engine.available.return_value = True

    fake_cls = Mock(
        return_value=fake_engine
    )

    with patch.dict(
        "uma.ocr.engines.REGISTRY",
        {
            "tesseract": fake_cls
        },
        clear=True,
    ):
        result = build_engines(
            ["tesseract"],
            tess_psms=(3, 6),
        )

    fake_cls.assert_called_once_with(
        lang="eng",
        psms=(3, 6),
    )

    assert result == [
        fake_engine
    ]


def test_build_engines_doctr():
    fake_engine = Mock()
    fake_engine.available.return_value = True

    fake_cls = Mock(
        return_value=fake_engine
    )

    with patch.dict(
        "uma.ocr.engines.REGISTRY",
        {
            "doctr": fake_cls
        },
        clear=True,
    ):
        result = build_engines(
            ["doctr"]
        )

    fake_cls.assert_called_once_with()

    assert result == [
        fake_engine
    ]


def test_build_engines_skips_unavailable():
    unavailable = Mock()
    unavailable.available.return_value = (
        False
    )

    available = Mock()
    available.available.return_value = (
        True
    )

    cls1 = Mock(
        return_value=unavailable
    )

    cls2 = Mock(
        return_value=available
    )

    with patch.dict(
        "uma.ocr.engines.REGISTRY",
        {
            "paddle": cls1,
            "doctr": cls2,
        },
        clear=True,
    ):
        result = build_engines([
            "paddle",
            "doctr",
        ])

    assert result == [
        available
    ]


def test_build_engines_none_available():
    engine = Mock()
    engine.available.return_value = False

    cls = Mock(
        return_value=engine
    )

    with patch.dict(
        "uma.ocr.engines.REGISTRY",
        {
            "doctr": cls
        },
        clear=True,
    ):
        with pytest.raises(
            RuntimeError,
            match="No OCR engine",
        ):
            build_engines([
                "doctr"
            ])