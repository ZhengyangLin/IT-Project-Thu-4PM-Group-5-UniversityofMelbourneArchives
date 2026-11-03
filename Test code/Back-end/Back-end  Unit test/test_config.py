import logging

import yaml

from uma.config import (
    PreprocessCfg,
    OcrCfg,
    LayoutCfg,
    ViewCfg,
    RuleCfg,
    LlmCfg,
    ConfidenceCfg,
    Config,
    _merge,
)


# ============================================================
# PreprocessCfg
# ============================================================

def test_preprocess_cfg_defaults():
    cfg = PreprocessCfg()

    assert cfg.deskew_deadzone_deg == 0.5
    assert cfg.clahe_clip == 2.0
    assert cfg.clahe_grid == 8
    assert cfg.min_text_ratio == 0.005


# ============================================================
# OcrCfg
# ============================================================

def test_ocr_cfg_defaults():
    cfg = OcrCfg()

    assert cfg.engines == [
        "paddle",
        "tesseract",
        "doctr",
    ]

    assert cfg.coarse_upscale == 2.0
    assert cfg.coarse_target_char_px == 26.0
    assert cfg.fine_upscale == 2.5
    assert cfg.target_char_px == 42.0
    assert cfg.max_upscale == 6.0
    assert cfg.min_conf == 0.30
    assert cfg.max_long_side == 6000
    assert cfg.paddle_max_side_limit == 0
    assert cfg.iou_merge == 0.50
    assert cfg.tesseract_psm_coarse == (6, 11)
    assert cfg.lang == "en"


def test_ocr_cfg_engines_not_shared():
    first = OcrCfg()
    second = OcrCfg()

    first.engines.append("custom")

    assert "custom" in first.engines
    assert "custom" not in second.engines


# ============================================================
# LayoutCfg
# ============================================================

def test_layout_cfg_defaults():
    cfg = LayoutCfg()

    assert cfg.line_ink_percentile == 15.0
    assert cfg.line_min_ratio == 0.10
    assert cfg.row_y_tol == 0.5
    assert cfg.row_max_x_gap == 6.0


# ============================================================
# ViewCfg
# ============================================================

def test_view_cfg_defaults():
    cfg = ViewCfg()

    assert cfg.grid_max_cols == 120
    assert cfg.grid_max_blank_rows == 2
    assert cfg.show_row_no is True


# ============================================================
# RuleCfg
# ============================================================

def test_rule_cfg_defaults():
    cfg = RuleCfg()

    assert cfg.lock_threshold == 0.85
    assert cfg.anchor_fuzz == 85
    assert cfg.year_range == (1900, 2000)

    assert cfg.architects == [
        "YUNCKEN FREEMAN ARCHITECTS PTY LTD",
        "YUNCKEN FREEMAN",
        "BATES SMART MCCUTCHEON",
        "BATES SMART",
    ]


def test_rule_cfg_architects_not_shared():
    first = RuleCfg()
    second = RuleCfg()

    first.architects.append("TEST ARCHITECT")

    assert "TEST ARCHITECT" in first.architects
    assert "TEST ARCHITECT" not in second.architects


# ============================================================
# LlmCfg
# ============================================================

def test_llm_cfg_defaults():
    cfg = LlmCfg()

    assert cfg.enabled is True
    assert cfg.base_url == "http://localhost:8000/v1"
    assert cfg.api_key_env == "OPENAI_API_KEY"
    assert cfg.model == "qwen3-32b-instruct"
    assert cfg.temperature == 0.0
    assert cfg.max_tokens == 1500
    assert cfg.timeout == 60
    assert cfg.max_retries == 0


# ============================================================
# ConfidenceCfg
# ============================================================

def test_confidence_cfg_defaults():
    cfg = ConfidenceCfg()

    assert cfg.w_agreement == 0.15
    assert cfg.w_ocr_conf == 0.15
    assert cfg.w_format == 0.15
    assert cfg.w_llm_conf == 0.15

    assert cfg.llm_conf_map == {
        "high": 1.0,
        "medium": 0.7,
        "low": 0.35,
    }

    assert cfg.agreement_after_adjudication == 0.85
    assert cfg.t_high == 0.75
    assert cfg.t_low == 0.45
    assert cfg.correction_conf_ceiling == 0.85


def test_confidence_map_not_shared():
    first = ConfidenceCfg()
    second = ConfidenceCfg()

    first.llm_conf_map["test"] = 0.5

    assert "test" in first.llm_conf_map
    assert "test" not in second.llm_conf_map


# ============================================================
# Config
# ============================================================

def test_config_defaults():
    cfg = Config()

    assert isinstance(cfg.preprocess, PreprocessCfg)
    assert isinstance(cfg.ocr, OcrCfg)
    assert isinstance(cfg.layout, LayoutCfg)
    assert isinstance(cfg.view, ViewCfg)
    assert isinstance(cfg.rule, RuleCfg)
    assert isinstance(cfg.llm, LlmCfg)
    assert isinstance(cfg.conf, ConfidenceCfg)

    assert cfg.debug_viz is True
    assert cfg.write_trace is True


def test_config_instances_not_shared():
    first = Config()
    second = Config()

    first.preprocess.clahe_clip = 99
    first.ocr.engines.append("new")

    assert second.preprocess.clahe_clip == 2.0
    assert "new" not in second.ocr.engines


# ============================================================
# Config.dump()
# ============================================================

def test_config_dump():
    cfg = Config()

    result = cfg.dump()

    assert isinstance(result, dict)

    assert result["debug_viz"] is True
    assert result["write_trace"] is True

    assert result["preprocess"]["clahe_clip"] == 2.0
    assert result["layout"]["row_y_tol"] == 0.5
    assert result["llm"]["enabled"] is True

    assert result["ocr"]["engines"] == [
        "paddle",
        "tesseract",
        "doctr",
    ]


def test_config_dump_reflects_changes():
    cfg = Config()

    cfg.preprocess.clahe_clip = 3.5
    cfg.debug_viz = False

    result = cfg.dump()

    assert result["preprocess"]["clahe_clip"] == 3.5
    assert result["debug_viz"] is False


# ============================================================
# Config.load()
# ============================================================

def test_load_without_path():
    cfg = Config.load()

    assert isinstance(cfg, Config)
    assert cfg.preprocess.clahe_clip == 2.0
    assert cfg.debug_viz is True


def test_load_none_path():
    cfg = Config.load(None)

    assert isinstance(cfg, Config)
    assert cfg.ocr.min_conf == 0.30


def test_load_nonexistent_file(tmp_path):
    path = tmp_path / "does_not_exist.yaml"

    cfg = Config.load(path)

    assert isinstance(cfg, Config)

    # Defaults must remain unchanged.
    assert cfg.preprocess.clahe_clip == 2.0
    assert cfg.layout.row_y_tol == 0.5


def test_load_valid_yaml(tmp_path):
    path = tmp_path / "config.yaml"

    path.write_text(
        """
preprocess:
  clahe_clip: 4.5
  clahe_grid: 16

ocr:
  min_conf: 0.75
  lang: fr

layout:
  row_y_tol: 0.8

debug_viz: false
write_trace: false
""",
        encoding="utf-8",
    )

    cfg = Config.load(path)

    assert cfg.preprocess.clahe_clip == 4.5
    assert cfg.preprocess.clahe_grid == 16

    assert cfg.ocr.min_conf == 0.75
    assert cfg.ocr.lang == "fr"

    assert cfg.layout.row_y_tol == 0.8

    assert cfg.debug_viz is False
    assert cfg.write_trace is False


def test_load_empty_yaml(tmp_path):
    path = tmp_path / "empty.yaml"

    path.write_text(
        "",
        encoding="utf-8",
    )

    cfg = Config.load(path)

    # Empty YAML becomes {}, so defaults remain.
    assert isinstance(cfg, Config)
    assert cfg.preprocess.clahe_clip == 2.0
    assert cfg.debug_viz is True


def test_load_yaml_null(tmp_path):
    path = tmp_path / "null.yaml"

    path.write_text(
        "null",
        encoding="utf-8",
    )

    cfg = Config.load(path)

    assert isinstance(cfg, Config)
    assert cfg.write_trace is True


def test_load_nested_values(tmp_path):
    path = tmp_path / "nested.yaml"

    path.write_text(
        """
llm:
  enabled: false
  model: test-model
  timeout: 120

conf:
  t_high: 0.9
  t_low: 0.3
""",
        encoding="utf-8",
    )

    cfg = Config.load(path)

    assert cfg.llm.enabled is False
    assert cfg.llm.model == "test-model"
    assert cfg.llm.timeout == 120

    assert cfg.conf.t_high == 0.9
    assert cfg.conf.t_low == 0.3


# ============================================================
# _merge()
# ============================================================

def test_merge_known_nested_value():
    cfg = Config()

    result = _merge(
        cfg,
        {
            "preprocess": {
                "clahe_clip": 5.0,
            }
        },
    )

    assert result is cfg
    assert cfg.preprocess.clahe_clip == 5.0


def test_merge_multiple_nested_values():
    cfg = Config()

    _merge(
        cfg,
        {
            "preprocess": {
                "clahe_clip": 5.0,
                "clahe_grid": 32,
            },
            "layout": {
                "row_y_tol": 1.0,
                "row_max_x_gap": 10.0,
            },
        },
    )

    assert cfg.preprocess.clahe_clip == 5.0
    assert cfg.preprocess.clahe_grid == 32

    assert cfg.layout.row_y_tol == 1.0
    assert cfg.layout.row_max_x_gap == 10.0


def test_merge_top_level_value():
    cfg = Config()

    _merge(
        cfg,
        {
            "debug_viz": False,
            "write_trace": False,
        },
    )

    assert cfg.debug_viz is False
    assert cfg.write_trace is False


def test_merge_unknown_section(caplog):
    cfg = Config()

    with caplog.at_level(logging.WARNING):
        result = _merge(
            cfg,
            {
                "unknown_section": {
                    "value": 123,
                }
            },
        )

    assert result is cfg

    assert (
        "unrecognized keys"
        in caplog.text
    )

    assert (
        "unknown_section"
        in caplog.text
    )


def test_merge_unknown_nested_key(caplog):
    cfg = Config()

    with caplog.at_level(logging.WARNING):
        _merge(
            cfg,
            {
                "preprocess": {
                    "clahe_clip": 7.0,
                    "does_not_exist": 123,
                }
            },
        )

    # Valid key is still applied.
    assert cfg.preprocess.clahe_clip == 7.0

    assert (
        "preprocess.does_not_exist"
        in caplog.text
    )


def test_merge_multiple_unknown_keys(caplog):
    cfg = Config()

    with caplog.at_level(logging.WARNING):
        _merge(
            cfg,
            {
                "unknown_section": 123,
                "layout": {
                    "bad_key": 456,
                },
            },
        )

    assert "unknown_section" in caplog.text
    assert "layout.bad_key" in caplog.text


def test_merge_known_keys_no_warning(caplog):
    cfg = Config()

    with caplog.at_level(logging.WARNING):
        _merge(
            cfg,
            {
                "preprocess": {
                    "clahe_clip": 3.0,
                },
                "debug_viz": False,
            },
        )

    assert cfg.preprocess.clahe_clip == 3.0
    assert cfg.debug_viz is False

    assert (
        "unrecognized keys"
        not in caplog.text
    )


def test_merge_replace_dataclass_section():
    cfg = Config()

    replacement = PreprocessCfg(
        clahe_clip=9.0,
        clahe_grid=20,
    )

    _merge(
        cfg,
        {
            "preprocess": replacement,
        },
    )

    assert cfg.preprocess is replacement
    assert cfg.preprocess.clahe_clip == 9.0
    assert cfg.preprocess.clahe_grid == 20


def test_merge_empty_data():
    cfg = Config()

    original = cfg.dump()

    result = _merge(
        cfg,
        {},
    )

    assert result is cfg
    assert cfg.dump() == original