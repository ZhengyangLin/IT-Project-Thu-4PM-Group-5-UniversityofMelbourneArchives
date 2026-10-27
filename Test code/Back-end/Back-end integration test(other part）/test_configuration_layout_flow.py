from pathlib import Path

from uma.config import Config, LayoutCfg, ViewCfg
from uma.layout import group_rows, split_rotated
from uma.models import Candidate, Token
from uma.views import build_views


def test_yaml_config_is_loaded_into_application_config(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
preprocess:
  clahe_clip: 3.0
debug_viz: false
""",
        encoding="utf-8",
    )

    config = Config.load(config_path)

    assert config.preprocess.clahe_clip == 3.0
    assert config.debug_viz is False


def test_default_config_is_available_when_no_file_is_given():
    config = Config.load()

    assert config.preprocess.clahe_clip == 2.0
    assert config.ocr.lang == "en"


def test_tokens_are_grouped_and_converted_to_both_text_views():
    tokens = [
        Token("DRAWING", (10, 10, 70, 30), 0.98),
        Token("PLAN", (78, 10, 125, 30), 0.97),
        Token("SCALE", (10, 55, 60, 75), 0.96),
        Token("1:100", (68, 55, 120, 75), 0.95),
    ]

    lines = group_rows(tokens, LayoutCfg())
    grid, index, flat = build_views(lines, ViewCfg())

    assert len(lines) == 2
    assert [line.text for line in lines] == ["DRAWING PLAN", "SCALE 1:100"]
    assert [token.tid for token in flat] == [0, 1, 2, 3]
    assert "DRAWING PLAN" in grid
    assert "[0] row 1  DRAWING" in index
    assert "[3] row 2  1:100" in index


def test_rotated_tokens_are_separated_from_upright_tokens():
    upright = Token("TITLE", (10, 10, 60, 30), 0.9)
    second_upright = Token("TEXT", (10, 40, 50, 60), 0.9)
    rotated = Token("SIDE", (80, 10, 90, 60), 0.9)

    normal, vertical = split_rotated([upright, second_upright, rotated])

    assert normal == [upright, second_upright]
    assert vertical == [rotated]


def test_index_view_keeps_ocr_alternatives():
    token = Token(
        "S24",
        (10, 10, 45, 30),
        0.8,
        candidates=[
            Candidate("S24", "paddle", 0.8),
            Candidate("524", "tesseract", 0.7),
        ],
    )

    lines = group_rows([token], LayoutCfg())
    _, index, _ = build_views(lines, ViewCfg())

    assert "alternatives tesseract:524" in index
