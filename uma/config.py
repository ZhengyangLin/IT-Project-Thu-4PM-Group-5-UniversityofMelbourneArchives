from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


# Image preprocessing settings
@dataclass
class PreprocessCfg:
    deskew_deadzone_deg: float = 0.5
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    min_text_ratio: float = 0.005


# OCR engine and recognition settings
@dataclass
class OcrCfg:
    engines: list[str] = field(default_factory=lambda: ["paddle", "tesseract", "doctr"])
    coarse_upscale: float = 2.0
    coarse_target_char_px: float = 26.0
    fine_upscale: float = 2.5
    target_char_px: float = 42.0
    max_upscale: float = 6.0
    min_conf: float = 0.30
    max_long_side: int = 6000
    paddle_max_side_limit: int = 0
    probe_gaps: bool = True
    iou_merge: float = 0.50
    tesseract_psm_coarse: tuple = (6, 11)
    lang: str = "en"


# Text layout analysis settings
@dataclass
class LayoutCfg:

    line_ink_percentile: float = 15.0
    line_min_ratio: float = 0.10
    row_y_tol: float = 0.5
    row_max_x_gap: float = 6.0


# Output view settings
@dataclass
class ViewCfg:
    grid_max_cols: int = 120
    grid_max_blank_rows: int = 2
    show_row_no: bool = True


# Rule-based extraction settings
@dataclass
class RuleCfg:
    lock_threshold: float = 0.85
    anchor_fuzz: int = 85
    year_range: tuple = (1900, 2000)

    # Known architect names
    architects: list[str] = field(default_factory=lambda: [
        "YUNCKEN FREEMAN ARCHITECTS PTY LTD",
        "YUNCKEN FREEMAN",
        "BATES SMART MCCUTCHEON",
        "BATES SMART",
    ])


# LLM service settings
@dataclass
class LlmCfg:
    enabled: bool = True
    base_url: str = "http://localhost:8000/v1"
    api_key_env: str = "OPENAI_API_KEY"
    model: str = "qwen3-32b-instruct"
    temperature: float = 0.0
    max_tokens: int = 1500
    timeout: int = 60
    max_retries: int = 0


# Confidence scoring settings
@dataclass
class ConfidenceCfg:
    w_agreement: float = 0.15
    w_ocr_conf: float = 0.15
    w_format: float = 0.15
    w_cross_sheet: float = 0.15
    w_reread: float = 0.15
    w_cluster: float = 0.10
    w_llm_conf: float = 0.15

    # Map LLM confidence labels to numeric values
    llm_conf_map: dict = field(default_factory=lambda: {
        "high": 1.0, "medium": 0.7, "low": 0.35})

    agreement_after_adjudication: float = 0.85
    t_high: float = 0.75
    t_low: float = 0.45
    reread_fields: list[str] = field(default_factory=lambda: ["drawing_number", "date", "scale"])
    correction_conf_ceiling: float = 0.85


# Main configuration container
@dataclass
class Config:
    preprocess: PreprocessCfg = field(default_factory=PreprocessCfg)
    ocr: OcrCfg = field(default_factory=OcrCfg)
    layout: LayoutCfg = field(default_factory=LayoutCfg)
    view: ViewCfg = field(default_factory=ViewCfg)
    rule: RuleCfg = field(default_factory=RuleCfg)
    llm: LlmCfg = field(default_factory=LlmCfg)
    conf: ConfidenceCfg = field(default_factory=ConfidenceCfg)
    debug_viz: bool = True
    write_trace: bool = True

    # Load configuration from a YAML file
    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        cfg = cls()
        if path and Path(path).exists():
            data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
            cfg = _merge(cfg, data)
        return cfg

    # Convert configuration to a dictionary
    def dump(self) -> dict[str, Any]:
        return asdict(self)


# Merge YAML values into the default configuration
def _merge(cfg: Config, data: dict) -> Config:
    import logging
    log = logging.getLogger(__name__)
    unknown: list[str] = []

    for section, values in data.items():
        if not hasattr(cfg, section):
            unknown.append(section)
            continue

        target = getattr(cfg, section)

        if isinstance(values, dict) and hasattr(target, "__dataclass_fields__"):
            for k, v in values.items():
                if hasattr(target, k):
                    setattr(target, k, v)
                else:
                    unknown.append(f"{section}.{k}")
        else:
            setattr(cfg, section, values)

    # Warn about unknown configuration keys
    if unknown:
        log.warning("Ignored %d unrecognized keys in config.yaml: %s",
                    len(unknown), ", ".join(unknown))

    return cfg