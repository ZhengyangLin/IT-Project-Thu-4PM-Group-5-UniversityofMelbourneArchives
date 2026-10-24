from __future__ import annotations
import gc
from dataclasses import dataclass, field
import logging
import os
import sys
from typing import Any

from ...unimelb_result_codec import extract_ocr_result


log = logging.getLogger(__name__)
_PIPELINE: Any = None


@dataclass(frozen=True)
class OcrImageJob:

    image_path: str
    node_id: str
    item_output_dir: str


@dataclass
class OcrImageOutcome:

    ok: bool
    result: dict[str, Any] = field(default_factory=dict)
    summary_row: dict[str, Any] | None = None
    error: str | None = None


def initialize_ocr_worker(config_path: str,engines: tuple[str, ...],verbose_prompt: bool,log_dir: str | None = None,) -> None:
    global _PIPELINE

    os.environ.update({
        "FLAGS_allocator_strategy": "auto_growth",
        "FLAGS_fraction_of_gpu_memory_to_use": "0.3",
        "FLAGS_initial_gpu_memory_in_mb": "512",
        "FLAGS_reallocate_gpu_memory_in_mb": "128",
        "FLAGS_cudnn_exhaustive_search": "0",
        "FLAGS_conv_workspace_size_limit": "64",
    })

    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    os.environ.setdefault("FLAGS_enable_pir_api", "0")
    # os.environ.setdefault("PADDLE_PDX_CPU_NUM_THREADS", "1")
    # os.environ.setdefault("OMP_NUM_THREADS", "1")
    # os.environ.setdefault("MKL_NUM_THREADS", "1")
    # os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    # os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

    if log_dir:
        from ...logging_utils import setup_logging
        setup_logging(log_dir)

    from ...config import Config
    from ...pipeline import Pipeline

    cfg = Config.load(config_path)
    cfg.ocr.engines = list(engines)
    _PIPELINE = Pipeline(cfg, verbose_prompt=verbose_prompt)
    log.info("OCR worker initialized: pid=%s engines=%s", os.getpid(), engines)


def _clear_paddle_gpu_cache() -> None:
    try:
        paddle = sys.modules.get("paddle")
        if paddle is None or not paddle.is_compiled_with_cuda():
            return
        if paddle.device.cuda.device_count() == 0:
            return
        paddle.device.synchronize("gpu:0")
        gc.collect()
        paddle.device.cuda.empty_cache()
        log.info("Paddle GPU cache cleanup finished: pid=%s device=gpu:0", os.getpid())
    except Exception:
        log.warning("Paddle GPU cache cleanup failed: pid=%s", os.getpid(), exc_info=True)

def process_ocr_image(job: OcrImageJob) -> OcrImageOutcome:
    if _PIPELINE is None:
        return OcrImageOutcome(ok=False, error="OCR worker is not initialized")

    try:
        from ...pipeline import record_to_row, write_outputs

        record = _PIPELINE.process(
            job.image_path,
            job.node_id,
            out_dir=job.item_output_dir,
        )
        write_outputs([record], job.item_output_dir)
        failure = (
            record.diagnostics.get("processing_error")
            or record.diagnostics.get("fatal")
        )
        return OcrImageOutcome(
            ok=not bool(failure),
            result=extract_ocr_result(record),
            summary_row=record_to_row(record),
            error=str(failure) if failure else None,
        )
    except Exception as exc:
        log.exception(
            "OCR worker failed to process image: node_id=%s image=%s",
            job.node_id,
            job.image_path,
        )
        return OcrImageOutcome(
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        _clear_paddle_gpu_cache()


def write_batch_summary(output_dir: str, rows: list[dict[str, Any]]) -> str:
    from ...pipeline import write_summary_rows

    return write_summary_rows(rows, output_dir)
