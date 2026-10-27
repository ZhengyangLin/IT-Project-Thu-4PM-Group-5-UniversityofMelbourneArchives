"""Worker adapter -> real metadata codec, sidecars and CSV output."""
import csv
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from uma import pipeline
from uma.models import FieldValue, SheetRecord
from uma.web.workers import Unimelb_ocr_worker as worker


def sample_record(job, diagnostics=None):
    return SheetRecord(
        node_id=job.node_id, path=job.image_path,
        fields={"drawing_number": FieldValue(
            name="drawing_number", verbatim="003/A1", corrected="003/A2",
            confidence=.95, tokens=[7], source="rule", status="locked",
        )},
        token_map={"7": {"bbox": [10, 20, 30, 40]}},
        diagnostics=diagnostics or {},
    )


@pytest.mark.parametrize("diagnostics,error", [
    ({}, None), ({"fatal": "unreadable"}, "unreadable"),
    ({"processing_error": "image failed"}, "image failed"),
])
def test_worker_record_flows_to_codec_sidecar_and_batch_csv(tmp_path, monkeypatch, diagnostics, error):
    job = worker.OcrImageJob("drawing.png", "123", str(tmp_path / "item"))
    process = Mock(return_value=sample_record(job, diagnostics))
    monkeypatch.setattr(worker, "_PIPELINE", SimpleNamespace(process=process))
    # No installed Paddle module/GPU is used by these adapter tests.
    monkeypatch.setitem(sys.modules, "paddle", None)
    outcome = worker.process_ocr_image(job)
    assert outcome.ok is (error is None) and outcome.error == error
    assert outcome.result["drawing_number"] == "003/A2"
    assert outcome.result["drawing_number_tokens_bbox"] == {"x0": 10, "y0": 20, "x1": 30, "y1": 40}
    process.assert_called_once_with(job.image_path, job.node_id, out_dir=job.item_output_dir)
    sidecar = json.loads((Path(job.item_output_dir) / "sidecar/123.json").read_text())
    assert sidecar["fields"]["drawing_number"]["corrected"] == "003/A2"
    summary_path = worker.write_batch_summary(str(tmp_path / "batch"), [outcome.summary_row])
    with open(summary_path, encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["drawing_number"] == "003/A2"
    assert rows[0]["node_id"] == "123"


def test_uninitialized_worker_and_pipeline_exception_return_failed_outcomes(tmp_path, monkeypatch):
    job = worker.OcrImageJob("missing.png", "123", str(tmp_path))
    monkeypatch.setattr(worker, "_PIPELINE", None)
    assert worker.process_ocr_image(job).error == "OCR worker is not initialized"
    cleanup = Mock()
    monkeypatch.setattr(worker, "_clear_paddle_gpu_cache", cleanup)
    monkeypatch.setattr(worker, "_PIPELINE", SimpleNamespace(process=Mock(side_effect=ValueError("bad image"))))
    outcome = worker.process_ocr_image(job)
    assert not outcome.ok and outcome.error == "ValueError: bad image"
    assert outcome.result == {} and outcome.summary_row is None
    cleanup.assert_called_once()


def test_worker_initialization_loads_config_without_constructing_real_ocr(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text("debug_viz: false\nocr:\n  lang: en\n")
    constructor = Mock(return_value=SimpleNamespace(process=Mock()))
    monkeypatch.setattr(pipeline, "Pipeline", constructor)
    monkeypatch.setattr(worker, "_PIPELINE", None)
    from uma import logging_utils
    setup = Mock()
    monkeypatch.setattr(logging_utils, "setup_logging", setup)
    for key in [
        "FLAGS_allocator_strategy", "FLAGS_fraction_of_gpu_memory_to_use",
        "FLAGS_initial_gpu_memory_in_mb", "FLAGS_reallocate_gpu_memory_in_mb",
        "FLAGS_cudnn_exhaustive_search", "FLAGS_conv_workspace_size_limit",
        "FLAGS_use_mkldnn", "FLAGS_enable_pir_api", "OMP_WAIT_POLICY",
    ]:
        monkeypatch.delenv(key, raising=False)
    worker.initialize_ocr_worker(str(config), ("fake-engine",), False, str(tmp_path / "logs"))
    loaded = constructor.call_args.args[0]
    assert loaded.ocr.engines == ["fake-engine"] and loaded.debug_viz is False
    assert constructor.call_args.kwargs == {"verbose_prompt": False}
    assert worker._PIPELINE is constructor.return_value
    assert os.environ["FLAGS_fraction_of_gpu_memory_to_use"] == "0.3"
    setup.assert_called_once_with(str(tmp_path / "logs"))


def test_worker_gpu_cleanup_adapter_uses_mocked_device_only(monkeypatch):
    device = SimpleNamespace(
        cuda=SimpleNamespace(device_count=Mock(return_value=1), empty_cache=Mock()),
        synchronize=Mock(),
    )
    monkeypatch.setitem(sys.modules, "paddle", SimpleNamespace(
        is_compiled_with_cuda=lambda: True, device=device,
    ))
    collect = Mock()
    monkeypatch.setattr(worker.gc, "collect", collect)
    worker._clear_paddle_gpu_cache()
    device.synchronize.assert_called_once_with("gpu:0")
    device.cuda.empty_cache.assert_called_once()
    collect.assert_called_once()


def test_cleanup_failure_is_logged_without_losing_worker_result(tmp_path, monkeypatch):
    job = worker.OcrImageJob("drawing.png", "123", str(tmp_path))
    monkeypatch.setattr(worker, "_PIPELINE", SimpleNamespace(process=lambda *a, **k: sample_record(job)))
    monkeypatch.setitem(sys.modules, "paddle", SimpleNamespace(is_compiled_with_cuda=Mock(side_effect=RuntimeError("device gone"))))
    warning = Mock()
    monkeypatch.setattr(worker.log, "warning", warning)
    assert worker.process_ocr_image(job).ok
    warning.assert_called_once()
