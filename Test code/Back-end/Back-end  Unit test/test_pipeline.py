"""High-coverage unit tests for uma/pipeline.py.

No large payloads are supplied as pytest parameter values (important on
Windows, where pytest's test IDs also appear in environment variables).
OCR, LLM, image processing and tracing are mocked. File outputs use tmp_path.
Run: python -m pytest tests/test_pipeline.py -v
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

import uma.pipeline as mod


@dataclass
class FakeValue:
    name: str = "drawing_title"
    verbatim: str | None = "SITE PLAN"
    corrected: str | None = None
    correction_basis: str | None = None
    confidence: float = 0.92
    status: str = "auto_accept"
    reason: str | None = None
    source: str = "rule"
    tokens: list[int] = field(default_factory=list)
    signals: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass
class FakeRecord:
    node_id: str
    path: str
    series: str = ""
    title_bbox: tuple | None = None
    localization_method: str = ""
    localization_conf: float = 0.0
    grid: str = ""
    index_view: str = ""
    token_map: dict = field(default_factory=dict)
    fields: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)


class FakeToken:
    def __init__(self, text="SITE PLAN", bbox=(12, 7, 25, 14), tid=1,
                 conf=0.91, agreement=0.98, candidates=None):
        self.text = text
        self.bbox = bbox
        self.tid = tid
        self.conf = conf
        self.agreement = agreement
        self.candidates = candidates if candidates is not None else [
            NS(text=text, engine="mock_ocr", conf=conf)
        ]

    @property
    def cx(self):
        return (self.bbox[0] + self.bbox[2]) / 2

    @property
    def cy(self):
        return (self.bbox[1] + self.bbox[3]) / 2


class FakeLine:
    def __init__(self, tokens, row_no=0):
        self.tokens = list(tokens)
        self.row_no = row_no
        self.x0 = min((t.bbox[0] for t in tokens), default=0)
        self.y0 = min((t.bbox[1] for t in tokens), default=0)


class FakeTracer:
    def __init__(self, *args):
        self.calls = []
        self.closed = False
        self.args = args

    def section(self, *args):
        self.calls.append(("section", args))

    def tokens(self, tokens):
        self.calls.append(("tokens", list(tokens)))

    def text(self, text):
        self.calls.append(("text", text))

    def close(self):
        self.closed = True
        return "trace/fake.txt"


def make_cfg(**overrides):
    cfg = NS(
        ocr=NS(engines=["fake"], lang="eng", paddle_max_side_limit=1000,
               tesseract_psm_coarse=11, coarse_upscale=1.5,
               coarse_target_char_px=20.0, iou_merge=0.5, min_conf=0.1,
               max_long_side=1000, fine_upscale=1.2,
               target_char_px=16.0, max_upscale=4.0),
        preprocess=NS(),
        layout=NS(line_ink_percentile=80, line_min_ratio=0.3),
        rule=NS(anchor_fuzz=85),
        view=NS(),
        conf=NS(t_low=0.35),
        llm=NS(enabled=False),
        write_trace=False,
        debug_viz=False,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


@pytest.fixture
def env(monkeypatch):
    """Patch only pipeline dependencies; exercise real pipeline control flow."""
    cfg = make_cfg()
    gray = np.zeros((50, 90), dtype=np.uint8)
    coarse = FakeToken("COARSE", (11, 6, 20, 12), tid=1)
    fine = FakeToken("SITE PLAN", (2, 2, 15, 9), tid=2)
    trace = FakeTracer()
    engines = [NS(name="fake")]
    build = Mock(return_value=engines)
    llm = Mock(return_value=NS(name="fake_llm"))
    monkeypatch.setattr(mod, "build_engines", build)
    monkeypatch.setattr(mod, "LlmClient", llm)
    monkeypatch.setattr(mod, "SheetRecord", FakeRecord)
    monkeypatch.setattr(mod.tr, "Tracer", lambda *a: trace)
    monkeypatch.setattr(mod.preprocess, "run", Mock(return_value=(gray, gray.copy(), {"gray": True})))
    monkeypatch.setattr(mod.triage, "assess", Mock(return_value={
        "verdict": "reliable", "char_height_p90_px": 8, "note": "clear"}))
    monkeypatch.setattr(mod.triage, "infer_series", Mock(return_value="A"))
    monkeypatch.setattr(mod.triage, "char_height", Mock(return_value=(8.0, None)))
    monkeypatch.setattr(mod.triage, "noise_ratio", Mock(return_value=0.05))
    monkeypatch.setattr(mod.triage, "probe_anchors", Mock(return_value=2))
    reads = Mock(side_effect=lambda *a, **k: (  # coarse / fine selected by call number
        ([coarse], {"engines_used": ["fake"]}) if reads.call_count == 1
        else ([fine], {"engines_used": ["fake"]})
    ))
    monkeypatch.setattr(mod, "read_all", reads)
    location = NS(bbox=(10, 5, 50, 30), method="ocr", confidence=0.93,
                  detail={"candidate_count": 1})
    monkeypatch.setattr(mod.loc, "locate", Mock(return_value=location))
    monkeypatch.setattr(mod, "merge_passes", Mock(side_effect=lambda fine, coarse: (fine, {"merged": len(coarse)})))
    monkeypatch.setattr(mod.layout, "split_rotated", Mock(side_effect=lambda toks: (toks, [])))
    monkeypatch.setattr(mod.layout, "group_rows", Mock(side_effect=lambda toks, cfg: [FakeLine(toks)] if toks else []))
    monkeypatch.setattr(mod.layout, "Line", FakeLine)
    monkeypatch.setattr(mod.views, "assign_ids", Mock())
    monkeypatch.setattr(mod.views, "build_views", Mock(side_effect=lambda rows, cfg, hints:
                        ("GRID", "INDEX", [t for row in rows for t in row.tokens])))
    monkeypatch.setattr(mod.rules, "analyze_candidates", Mock(return_value=None))
    monkeypatch.setattr(mod.rules, "extract", Mock(return_value=({}, [], {})))
    monkeypatch.setattr(mod, "assign_fields", Mock(return_value=({}, {})))
    evidence = Mock()
    monkeypatch.setattr(mod, "record_evidence_regions", evidence)
    fuse = Mock()
    monkeypatch.setattr(mod.verify, "fuse", fuse)
    monkeypatch.setattr(mod, "is_vague_scale", Mock(return_value=False))
    process = mod.Pipeline(cfg)
    process._last_debug = {}
    return NS(cfg=cfg, pipeline=process, reads=reads, coarse=coarse,
              fine=fine, loc=location, trace=trace, evidence=evidence,
              fuse=fuse, engine_build=build, llm=llm, gray=gray)


@pytest.mark.parametrize("content", [b"", b"hello"])
def test_sha256(tmp_path, content):
    p = tmp_path / "sample.png"
    p.write_bytes(content)
    assert mod.sha256(str(p)) == hashlib.sha256(content).hexdigest()


def test_sha256_multiple_reads_without_large_parameters(monkeypatch):
    """Check the read-until-EOF loop without putting megabytes in test IDs."""
    chunks = iter((b"first", b"second", b""))
    requested_sizes = []

    class FakeStream:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, size):
            requested_sizes.append(size)
            return next(chunks)

    monkeypatch.setattr(mod, "open", lambda _path, _mode: FakeStream(), raising=False)
    actual = mod.sha256("virtual-image.png")
    assert actual == hashlib.sha256(b"firstsecond").hexdigest()
    assert requested_sizes == [1 << 20] * 3


def test_sha256_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        mod.sha256(str(tmp_path / "missing.png"))


def test_node_id_from_path():
    assert mod.node_id_from_path(Path("plans/N123/image.tif")) == "N123"


def test_pipeline_initialization(env):
    e = env
    assert e.pipeline.engines[0].name == "fake"
    assert e.pipeline.force is False
    assert e.pipeline.verbose_prompt is True
    e.engine_build.assert_called_once_with(["fake"], "eng", 1000, 11)
    e.llm.assert_called_once_with(e.cfg.llm)


@pytest.mark.parametrize("height,scale", [(0, 1.5), (5, 4.0), (10, 2.0)])
def test_coarse_scale_selection(env, monkeypatch, height, scale):
    monkeypatch.setattr(mod.triage, "char_height", Mock(return_value=(height, None)))
    env.reads.side_effect = [([], {"engines_used": ["fake"]})]
    result = env.pipeline.process("folder/image.png")
    assert result.diagnostics["fatal"].startswith("No OCR engine")
    assert result.diagnostics["coarse_ocr"]["coarse_scale_used"] == scale
    assert env.reads.call_args.args[2] == pytest.approx(scale)
    assert not env.trace.closed


@pytest.mark.parametrize("engines,expected", [([], "none"), (["paddle"], "paddle")])
def test_no_ocr_results_reports_engine_failure(env, engines, expected):
    env.reads.side_effect = [([], {"engines_used": engines})]
    result = env.pipeline.process("node/image.png", node_id="CUSTOM")
    assert result.node_id == "CUSTOM"
    assert expected in result.diagnostics["fatal"]
    assert result.diagnostics["prompt_version"] == mod.PROMPT_VERSION
    assert result.series == "A"
    assert result.fields == {}


def test_nonreliable_triage_warning(env, monkeypatch, caplog):
    monkeypatch.setattr(mod.triage, "assess", Mock(return_value={
        "verdict": "poor", "char_height_p90_px": 2, "note": "low resolution"}))
    env.reads.side_effect = [([], {"engines_used": []})]
    with caplog.at_level("WARNING"):
        result = env.pipeline.process("N/image.png")
    assert result.diagnostics["triage_warning"] == "low resolution"
    assert "low resolution" in caplog.text


@pytest.mark.parametrize("noise,hits,force,should_stop", [
    (0.9, 0, False, True),
    (0.75, 0, False, False),
    (0.9, 1, False, False),
    (0.9, 0, True, False),
    (0.1, 0, False, False),
])
def test_noise_rejection_and_force(env, monkeypatch, noise, hits, force, should_stop):
    monkeypatch.setattr(mod.triage, "noise_ratio", Mock(return_value=noise))
    monkeypatch.setattr(mod.triage, "probe_anchors", Mock(return_value=hits))
    env.pipeline.force = force
    rec = env.pipeline.process("N/image.png")
    assert ("fatal" in rec.diagnostics) is should_stop
    if should_stop:
        assert "noise" in rec.diagnostics["fatal"]
        assert env.reads.call_count == 1
    else:
        assert env.reads.call_count == 2
        assert rec.diagnostics["processing_status"] == "success"


def test_full_success_result_and_coordinate_transform(env):
    rec = env.pipeline.process("N005/image.png")
    assert rec.node_id == "N005"
    assert Path(rec.path) == Path("N005/image.png")
    assert rec.title_bbox == (10, 5, 50, 30)
    assert rec.localization_method == "ocr"
    assert rec.localization_conf == 0.93
    assert rec.grid == "GRID" and rec.index_view == "INDEX"
    assert rec.diagnostics["processing_status"] == "success"
    assert rec.diagnostics["crop_size"] == [40, 25]
    assert rec.diagnostics["pass_merge"] == {"merged": 1}
    assert rec.diagnostics["fine_ocr"]["fine_upscale_used"] == 2.0
    assert rec.token_map["2"]["bbox"] == [12, 7, 25, 14]
    assert rec.token_map["2"]["text"] == "SITE PLAN"
    assert env.trace.closed
    assert rec.diagnostics["elapsed_sec"] >= 0
    assert "debug_images" in rec.diagnostics


@pytest.mark.parametrize("height,expected", [
    (0.0, 1.2), (1.0, 4.0), (20.0, 1.2), (10.0, 1.6),
])
def test_fine_upscale_clamps(env, monkeypatch, height, expected):
    monkeypatch.setattr(mod.triage, "char_height", Mock(side_effect=[(8, None), (height, None)]))
    result = env.pipeline.process("N/image.png")
    assert result.diagnostics["fine_ocr"]["fine_upscale_used"] == expected
    assert env.reads.call_args.args[2] == expected


def test_fine_upscale_image_size_cap_and_floor(env):
    env.cfg.ocr.max_long_side = 2
    r = env.pipeline.process("N/image.png")
    assert r.diagnostics["fine_ocr"]["fine_upscale_used"] == 0.1


def test_debug_hooks_and_trace_file(env, monkeypatch):
    env.cfg.debug_viz = True
    env.cfg.write_trace = True
    tok_save = Mock()
    loc_save = Mock(return_value={"overview": "preview.png"})
    monkeypatch.setattr(env.pipeline, "_save_tokens_only", tok_save)
    monkeypatch.setattr(env.pipeline, "_save_debug", loc_save)
    rec = env.pipeline.process("N/image.png", out_dir="somewhere")
    tok_save.assert_called_once()
    loc_save.assert_called_once()
    assert rec.diagnostics["trace_file"] == "trace/fake.txt"


def test_debug_token_hook_runs_before_noise_abort(env, monkeypatch):
    env.cfg.debug_viz = True
    monkeypatch.setattr(mod.triage, "noise_ratio", lambda _texts: 0.92)
    monkeypatch.setattr(mod.triage, "probe_anchors", lambda _texts: 0)
    draw = Mock()
    monkeypatch.setattr(env.pipeline, "_save_tokens_only", draw)
    rec = env.pipeline.process("N/image.png")
    assert "fatal" in rec.diagnostics
    draw.assert_called_once()


def test_rotated_text_creates_extra_row(env, monkeypatch):
    rotated = FakeToken("DATE", (3, 2, 4, 15), tid=3)
    monkeypatch.setattr(mod.layout, "split_rotated", Mock(return_value=([], [rotated])))
    rec = env.pipeline.process("N/image.png")
    assert "3" in rec.token_map
    assert rec.token_map["3"]["text"] == "DATE"
    assert rec.diagnostics["processing_status"] == "success"


def test_candidate_hint_and_alternative_evidence(env, monkeypatch):
    env.fine.candidates.append(NS(text="SITE PLAM", engine="second", conf=0.62))
    analyzer = Mock(side_effect=lambda token, fld: {"preferred": "SITE PLAN"} if fld == "scale" else None)
    monkeypatch.setattr(mod.rules, "analyze_candidates", analyzer)
    rec = env.pipeline.process("N/image.png")
    assert rec.diagnostics["candidate_hints"] == {"2": {"preferred": "SITE PLAN"}}
    assert len(rec.token_map["2"]["candidates"]) == 2
    assert analyzer.call_count == 3


def test_candidate_hint_preferred_first_and_stops_search(env, monkeypatch):
    analyzer = Mock(return_value={"preferred": "ALT"})
    monkeypatch.setattr(mod.rules, "analyze_candidates", analyzer)
    rec = env.pipeline.process("N/image.png")
    assert rec.diagnostics["candidate_hints"]["2"]["preferred"] == "ALT"
    assert analyzer.call_count == 1


@pytest.mark.parametrize("details,expected", [
    ({"error": "timeout"}, "Model request failed"),
    ({"parse_error": "bad JSON"}, "Failed to parse"),
    ({"skipped": "llm_unavailable"}, "enabled but unavailable"),
])
def test_llm_failures_keep_rule_results(env, monkeypatch, details, expected):
    env.cfg.llm.enabled = True
    value = FakeValue(name="drawing_title", verbatim="SITE PLAN")
    monkeypatch.setattr(mod.rules, "extract", Mock(return_value=(
        {"drawing_title": value}, ["scale"], {"scale": "low confidence"})))
    monkeypatch.setattr(mod, "assign_fields", Mock(return_value=({}, details)))
    rec = env.pipeline.process("N/image.png")
    assert rec.fields["drawing_title"] is value
    assert rec.diagnostics["processing_status"] == "failed"
    assert expected in rec.diagnostics["processing_error"]
    assert rec.diagnostics["rules"]["rejected"]["scale"] == "low confidence"
    assert env.evidence.call_count == 1
    assert env.fuse.call_count == 1


def test_llm_unavailable_does_not_fail_if_disabled_or_no_pending(env, monkeypatch):
    monkeypatch.setattr(mod, "assign_fields", Mock(return_value=({}, {"skipped": "llm_unavailable"})))
    rec = env.pipeline.process("N/image.png")
    assert rec.diagnostics["processing_status"] == "success"
    env.cfg.llm.enabled = True
    rec2 = env.pipeline.process("N/image.png")
    assert rec2.diagnostics["processing_status"] == "success"


def test_llm_prompt_and_raw_response_saved_to_trace(env, monkeypatch):
    monkeypatch.setattr(mod, "assign_fields", Mock(return_value=(
        {}, {"prompt": "my question", "raw_response": "raw answer", "usage": {"tokens": 12}})))
    env.pipeline.process("N/image.png")
    texts = [v for typ, v in env.trace.calls if typ == "text"]
    assert any("my question" in v for v in texts)
    assert any("raw answer" in v for v in texts)
    assert any("system" in v for v in texts)


def test_rule_precedence_and_llm_merging(env, monkeypatch):
    rule = FakeValue(name="drawing_title", verbatim="SITE PLAN", tokens=[2])
    model_override = FakeValue(name="drawing_title", verbatim="OTHER")
    model_extra = FakeValue(name="architect", verbatim="ACME", tokens=[])
    monkeypatch.setattr(mod.rules, "extract", Mock(return_value=(
        {"drawing_title": rule}, ["architect"], {})))
    assign = Mock(return_value=({"drawing_title": model_override,
                                 "architect": model_extra}, {}))
    monkeypatch.setattr(mod, "assign_fields", assign)
    result = env.pipeline.process("N/image.png")
    assert result.fields["drawing_title"] is rule
    assert result.fields["architect"] is model_extra
    assert any("takes precedence" in n for n in rule.notes)
    assert rule.signals["localization_conf"] == 0.93
    assert model_extra.signals["localization_conf"] == 0.93
    assert env.evidence.call_count == 2
    assert env.fuse.call_count == 2
    assert assign.call_args.kwargs["rejected"] == {}


def test_fallback_localization_degrades_fields(env, monkeypatch):
    env.loc.method = "fallback"
    env.loc.confidence = 0.9
    value = FakeValue(name="drawing_number", verbatim="A101")
    monkeypatch.setattr(mod.rules, "extract", Mock(return_value=({"drawing_number": value}, [], {})))
    rec = env.pipeline.process("N/image.png")
    assert rec.fields["drawing_number"].status == "needs_review"
    assert rec.fields["drawing_number"].confidence == pytest.approx(0.35)
    assert any("fell back" in reason.lower() for reason in rec.diagnostics["degraded_reasons"])


@pytest.mark.parametrize("method,confidence,degraded", [
    ("ocr", 0.4, True),
    ("fallback", 0.4, True),
    ("ocr", 0.5, False),
    ("ocr", 0.95, False),
])
def test_localization_confidence_boundary(env, monkeypatch, method, confidence, degraded):
    env.loc.method = method
    env.loc.confidence = confidence
    value = FakeValue(name="drawing_title", tokens=[2])
    monkeypatch.setattr(mod.rules, "extract", Mock(return_value=({"drawing_title": value}, [], {})))
    result = env.pipeline.process("N/image.png")
    assert (result.fields["drawing_title"].status == "needs_review") is degraded
    if degraded:
        assert result.fields["drawing_title"].confidence == 0.35
    else:
        assert result.fields["drawing_title"].confidence == 0.92


def test_vague_scale_is_flagged(env, monkeypatch):
    value = FakeValue(name="scale", verbatim="AS SHOWN", tokens=[999])
    monkeypatch.setattr(mod.rules, "extract", Mock(return_value=({"scale": value}, [], {})))
    monkeypatch.setattr(mod, "is_vague_scale", Mock(return_value=True))
    rec = env.pipeline.process("N/image.png")
    assert "non-specific" in rec.fields["scale"].notes[0]
    assert rec.fields["scale"].status == "auto_accept"


def test_field_logging_missing_tokens_and_correction_basis(env, monkeypatch, caplog):
    missing = FakeValue(name="architect", verbatim=None, corrected=None,
                        reason="not_present", tokens=[], confidence=0.1)
    corrected = FakeValue(name="drawing_number", verbatim="A 10", corrected="A10",
                          correction_basis="OCR correction", tokens=[2, 999], confidence=0.9)
    monkeypatch.setattr(mod.rules, "extract", Mock(return_value=(
        {"architect": missing, "drawing_number": corrected}, [], {})))
    with caplog.at_level("INFO"):
        rec = env.pipeline.process("N/image.png")
    assert rec.fields["drawing_number"].corrected == "A10"
    assert "Token ID does not exist" in caplog.text
    assert "OCR correction" in caplog.text
    assert "not_present" in caplog.text
    assert any("(empty: not_present)" in v for typ, v in env.trace.calls if typ == "text")


def test_debug_draw_tokens_success_and_failure(env, tmp_path, monkeypatch):
    draw = Mock()
    monkeypatch.setattr(mod.viz, "draw_tokens", draw)
    env.pipeline._save_tokens_only("N001", str(tmp_path), env.gray, [env.coarse])
    assert (tmp_path / "debug").is_dir()
    assert draw.call_args.args[2].endswith("N001_0_tokens.jpg")
    draw.side_effect = OSError("render failed")
    env.pipeline._save_tokens_only("N001", str(tmp_path), env.gray, [env.coarse])
    assert draw.call_count == 2


def test_debug_draw_localization_success_and_error(env, tmp_path, monkeypatch):
    overview = Mock()
    crop = Mock()
    monkeypatch.setattr(mod.viz, "draw_v2", overview)
    monkeypatch.setattr(mod.viz, "save_titleblock_crop", crop)
    result = env.pipeline._save_debug("N001", str(tmp_path), env.gray,
                                      env.gray, [env.coarse], env.loc)
    assert result["overview"].endswith("N001_1_overview.jpg")
    assert result["titleblock"].endswith("N001_2_titleblock.jpg")
    assert env.pipeline._last_debug == result
    crop.side_effect = RuntimeError("cannot save crop")
    again = env.pipeline._save_debug("N002", str(tmp_path), env.gray,
                                     env.gray, [env.coarse], env.loc)
    assert again == {"overview": str(tmp_path / "debug" / "N002_1_overview.jpg")}
    assert env.pipeline._last_debug == again
    overview.side_effect = OSError("cannot draw")
    empty = env.pipeline._save_debug("N003", str(tmp_path), env.gray,
                                     env.gray, [env.coarse], env.loc)
    assert empty == {}


def test_record_to_row_all_fields_absent():
    rec = FakeRecord("N1", "img.png", series="A", diagnostics={})
    row = mod.record_to_row(rec)
    for name in mod.FIELD_ORDER:
        assert row[name] == ""
        assert row[name + "_status"] == ""
        assert row[name + "_reason"] == ""
    assert row["review_status"] == "auto_accept"
    assert row["processing_status"] == "success"
    assert row["prompt_version"] == mod.PROMPT_VERSION


@pytest.mark.parametrize("diagnostics,expected", [
    ({"processing_error": "model down"}, "model down"),
    ({"fatal": "bad image"}, "bad image"),
    ({"processing_error": "", "fatal": "fallback failure"}, "fallback failure"),
])
def test_record_to_row_failures(diagnostics, expected):
    rec = FakeRecord("N1", "img.png", diagnostics=diagnostics)
    row = mod.record_to_row(rec)
    assert row["review_status"] == "needs_review"
    assert row["processing_status"] == "failed"
    assert row["processing_error"] == expected


def test_record_to_row_fields_status_and_corrections():
    rec = FakeRecord("N1", "drawing.jpg", series="B",
        localization_method="ocr", localization_conf=0.83,
        fields={
            "scale": FakeValue(name="scale", verbatim="1 : 100", corrected="1:100",
                               correction_basis="normalized", confidence=0.77, status="needs_review", reason="vague"),
            "date": FakeValue(name="date", verbatim="2024", confidence=0.95),
            "architect": FakeValue(name="architect", verbatim=None, corrected="ACME", confidence=0.8),
        })
    row = mod.record_to_row(rec)
    assert row["scale"] == "1:100"
    assert row["scale_verbatim"] == "1 : 100"
    assert row["scale_correction_basis"] == "normalized"
    assert row["scale_conf"] == 0.77
    assert row["scale_reason"] == "vague"
    assert row["date"] == "2024"
    assert row["architect"] == "ACME"
    assert row["review_status"] == "needs_review"
    assert row["localization_conf"] == 0.83


def test_record_to_row_auto_accept_with_all_values():
    rec = FakeRecord("N2", "a.png", fields={
        name: FakeValue(name=name, verbatim="OK", status="auto_accept")
        for name in mod.FIELD_ORDER})
    assert mod.record_to_row(rec)["review_status"] == "auto_accept"


def test_write_outputs_sidecars_and_csv(tmp_path):
    first = FakeRecord("A", "folder/a.png", series="A", title_bbox=(1, 2, 30, 40),
        fields={"drawing_title": FakeValue(name="drawing_title", verbatim="平面图")},
        diagnostics={"ok": True})
    second = FakeRecord("B", "folder/b.png", diagnostics={"fatal": "no OCR"})
    files = mod.write_outputs([first, second], str(tmp_path))
    assert Path(files["csv"]).exists()
    assert Path(files["sidecar_dir"]).is_dir()
    sidecar = json.loads((tmp_path / "sidecar" / "A.json").read_text(encoding="utf-8"))
    assert sidecar["fields"]["drawing_title"]["verbatim"] == "平面图"
    assert sidecar["localization"]["score"] == 0.0
    assert sidecar["title_bbox"] == [1, 2, 30, 40]
    frame = pd.read_csv(files["csv"])
    assert len(frame) == 2
    assert list(frame["node_id"]) == ["A", "B"]
    assert frame.loc[1, "processing_status"] == "failed"


def test_write_outputs_empty_records(tmp_path):
    paths = mod.write_outputs([], str(tmp_path))
    assert Path(paths["csv"]).exists()
    assert (tmp_path / "sidecar").exists()


def test_write_summary_rows_creates_parent(tmp_path):
    folder = tmp_path / "new" / "nested"
    file_path = mod.write_summary_rows([{"node_id": "N5", "value": "áé"}], str(folder))
    assert Path(file_path).name == "sheets.csv"
    assert pd.read_csv(file_path).iloc[0]["value"] == "áé"


def test_write_csv_normal(tmp_path):
    p = tmp_path / "summary.csv"
    result = mod._write_csv(pd.DataFrame([{"x": 42}]), p)
    assert result == p
    assert "42" in p.read_text(encoding="utf-8-sig")


def test_write_csv_permission_error_uses_timestamped_fallback(tmp_path, monkeypatch, caplog):
    p = tmp_path / "sheets.csv"
    real_open = open
    attempted = []

    def flaky_open(path, *args, **kwargs):
        attempted.append(Path(path))
        if Path(path) == p:
            raise PermissionError("file in use")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(mod, "open", flaky_open, raising=False)
    with caplog.at_level("WARNING"):
        result = mod._write_csv(pd.DataFrame([{"x": 1}]), p)
    assert result != p
    assert result.exists()
    assert result.name.startswith("sheets_")
    assert len(attempted) == 2
    assert "in use" in caplog.text


def test_write_csv_both_names_locked_raises(tmp_path, monkeypatch):
    def always_locked(*a, **k):
        raise PermissionError("locked")

    monkeypatch.setattr(mod, "open", always_locked, raising=False)
    with pytest.raises(PermissionError, match="locked"):
        mod._write_csv(pd.DataFrame([{"x": 1}]), tmp_path / "sheets.csv")
