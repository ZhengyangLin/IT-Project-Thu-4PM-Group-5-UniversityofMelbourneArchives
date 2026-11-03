"""Adapter-to-fusion integration tests using deterministic external SDK doubles.

All uma engine adapters, image resizing, response parsing, bounding-box mapping,
candidate alignment, fusion, diagnostics, and pass merging execute real code.
Only pytesseract, paddleocr, and doctr (external inference/model-download
boundaries) are substituted. These tests do not measure model accuracy.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from uma.ocr import Engine, RawToken, build_engines, fuse, read_all
from uma.ocr.fusion import merge_passes


@pytest.fixture
def gray():
    """A reproducible source image, independent of OCR software installations."""
    image = np.full((100, 200), 255, dtype=np.uint8)
    image[20:40, 20:100] = 0
    image[65:80, 120:180] = 0
    return image


def _data(rows):
    """Build the actual pytesseract image_to_data dictionary shape."""
    return {
        "text": [r[0] for r in rows],
        "conf": [r[1] for r in rows],
        "left": [r[2][0] for r in rows],
        "top": [r[2][1] for r in rows],
        "width": [r[2][2] - r[2][0] for r in rows],
        "height": [r[2][3] - r[2][1] for r in rows],
    }


def _tesseract(monkeypatch, responses, *, version_error=None):
    module = ModuleType("pytesseract")
    module.Output = SimpleNamespace(DICT="dict")
    calls = []

    def version():
        if version_error is not None:
            raise version_error
        return "test-sdk"

    def image_to_data(image, *, lang, config, output_type):
        calls.append({"image": image.copy(), "lang": lang,
                      "config": config, "output_type": output_type})
        psm = int(config.split()[1])
        response = responses[psm] if isinstance(responses, dict) else responses
        if isinstance(response, BaseException):
            raise response
        return _data(response)

    module.get_tesseract_version = version
    module.image_to_data = image_to_data
    monkeypatch.setitem(sys.modules, "pytesseract", module)
    return calls


def _paddle(monkeypatch, sdk, *, accept=None):
    module = ModuleType("paddleocr")
    constructors = []

    def factory(**kwargs):
        constructors.append(kwargs)
        if accept is not None and not accept(kwargs):
            raise TypeError("SDK version does not support these constructor options")
        return sdk

    module.PaddleOCR = factory
    monkeypatch.setitem(sys.modules, "paddleocr", module)
    return constructors


def _poly(box):
    x0, y0, x1, y1 = box
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _paddle_v3(rows):
    return [{"rec_texts": [r[0] for r in rows],
             "rec_scores": [r[1] for r in rows],
             "rec_polys": [_poly(r[2]) for r in rows]}]


def _doctr(monkeypatch, words=(), *, load_error=None, inference_error=None):
    module = ModuleType("doctr")
    models = ModuleType("doctr.models")
    calls = {"loads": [], "images": []}

    def model(images):
        calls["images"].append([image.copy() for image in images])
        if inference_error:
            raise inference_error
        result_words = [SimpleNamespace(value=t, confidence=c, geometry=b)
                        for t, c, b in words]
        line = SimpleNamespace(words=result_words)
        block = SimpleNamespace(lines=[line])
        return SimpleNamespace(pages=[SimpleNamespace(blocks=[block])])

    def predictor(**kwargs):
        calls["loads"].append(kwargs)
        if load_error:
            raise load_error
        return model

    models.ocr_predictor = predictor
    module.models = models
    monkeypatch.setitem(sys.modules, "doctr", module)
    monkeypatch.setitem(sys.modules, "doctr.models", models)
    return calls


def _read(gray, engines, *, upscale=1.0, min_conf=0.5, **kwargs):
    return read_all(gray, engines, upscale=upscale, iou_thresh=0.4,
                    min_conf=min_conf, **kwargs)


def _check_partial_overlap_merge(monkeypatch, gray, fine_text, coarse_text,
                                 expected_count, expected_conflicts):
    """Check a real two-adapter fine/coarse text-matching scenario."""
    _tesseract(monkeypatch, {
        6: [(fine_text, 96, (20, 20, 100, 40))],
        11: [(coarse_text, 90, (52, 20, 132, 40))]})
    paddle_responses = iter([
        _paddle_v3([(fine_text, 0.94, (20, 20, 100, 40))]),
        _paddle_v3([(coarse_text, 0.89, (52, 20, 132, 40))])])
    _paddle(monkeypatch, SimpleNamespace(predict=lambda image, **kwargs:
        next(paddle_responses)))
    engines = build_engines(["tesseract", "paddle"])
    fine, _ = _read(gray, engines, psms=(6,))
    coarse, _ = _read(gray, engines, psms=(11,))
    # Similarity matching receives real two-engine fused tokens, not hand-made tokens.
    assert fine[0].agreement == coarse[0].agreement == 1.0
    assert all({c.engine for c in token.candidates} == {"tesseract", "paddle"}
               for token in fine + coarse)
    merged, diag = merge_passes(fine, coarse)
    assert len(merged) == expected_count
    assert diag["conflicts"] == expected_conflicts
    assert diag["recovered_from_coarse"] == expected_count - 1
    assert merged[0].text == fine_text

    assert merged[0].bbox == (20, 20, 100, 40)
    if expected_conflicts:
        assert merged[0].candidates[-1].text == coarse_text.strip()
        assert merged[0].candidates[-1].engine == "tesseract@coarse"
        assert merged[0].agreement == pytest.approx(0.667)
    elif expected_count == 2:
        assert merged[1].text == coarse_text
        assert {c.engine for c in merged[1].candidates} == {
            "tesseract@coarse", "paddle@coarse"}
        assert merged[1].agreement == 0.5
    else:
        assert merged[0].agreement == 1.0
        assert len(merged[0].candidates) == 2


def test_tesseract_multipass_dedup_filter_scale_and_fusion(monkeypatch, gray):
    """
    OCR-IT-001: Verify that Tesseract filters blank, invalid, and low-confidence text, removes
    duplicates across page segmentation modes, forwards its whitelist, and restores enlarged-image
    coordinates. Read a separate coarse Paddle pass through the public OCR interface and merge it
    with the fine result. Only ARCHIVE must remain, with the original coordinates and confidence;
    rejected text must not be reintroduced.
    """
    rows = [(" ARCHIVE ", "92", (40, 40, 200, 80)),
            ("", "99", (0, 0, 1, 1)),
            ("invalid", "-1", (0, 0, 2, 2)),
            ("weak", "20", (240, 130, 360, 160))]
    calls = _tesseract(monkeypatch, {6: rows, 11: rows})
    engines = build_engines(["tesseract"], tess_psms=(6, 11))
    tokens, diag = _read(gray, engines, upscale=2, whitelist="ARCHIVE")
    assert [token.text for token in tokens] == ["ARCHIVE"]
    assert tokens[0].bbox == (20, 20, 100, 40)
    assert tokens[0].conf == pytest.approx(0.92)
    assert tokens[0].agreement == 1.0
    assert len(tokens[0].candidates) == 1
    assert diag["tokens_per_engine"] == {"tesseract": 2}
    assert diag["token_count"] == 1
    assert diag["disagreements"] == 0
    assert [call["image"].shape for call in calls] == [(200, 400)] * 2
    assert [call["config"] for call in calls] == [
        "--psm 6 -c tessedit_char_whitelist=ARCHIVE",
        "--psm 11 -c tessedit_char_whitelist=ARCHIVE"]
    assert all(call["lang"] == "eng" and call["output_type"] == "dict"
               for call in calls)

    # A separate coarse adapter must corroborate the deduplicated fine result
    # without bringing the rejected low-confidence word back into the output.
    _paddle(monkeypatch, SimpleNamespace(predict=lambda image, **kw: _paddle_v3([
        ("archive", 0.90, (20, 20, 100, 40)),
        ("weak", 0.20, (120, 65, 180, 80))])))
    coarse, coarse_diag = _read(gray, build_engines(["paddle"]))
    merged, merge_diag = merge_passes(tokens, coarse)
    assert coarse_diag["tokens_per_engine"] == {"paddle": 2}
    assert [t.text for t in coarse] == ["archive"]
    assert [t.text for t in merged] == ["ARCHIVE"]
    assert merged[0].bbox == (20, 20, 100, 40)
    assert merged[0].conf == pytest.approx(0.92)
    assert merge_diag == {"fine_only": 1, "conflicts": 0,
                          "recovered_from_coarse": 0, "merged_total": 1}


def test_tesseract_failed_psm_recovers_with_remaining_pass(monkeypatch, gray):
    """
    OCR-IT-002: Verify that a failed Tesseract page segmentation mode does not discard TITLE from
    the surviving mode. Read a separate coarse Paddle pass and send both real adapter results
    through fusion and pass merging. Preserve the fine title and its confidence, recover the missing
    SHEET region, retain its coordinates and coarse-pass provenance, and cap the recovered token
    agreement at 0.5.
    """
    calls = _tesseract(monkeypatch, {
        6: RuntimeError("segmentation failed"),
        11: [("TITLE", 88, (20, 20, 100, 40))]})
    tokens, diag = _read(gray, build_engines(["tesseract"]))
    assert [t.text for t in tokens] == ["TITLE"]
    assert len(calls) == 2
    assert diag["engines_used"] == ["tesseract"]
    assert diag["tokens_per_engine"] == {"tesseract": 1}

    # A coarse scan from another adapter can recover text missed after the
    # failed segmentation mode, while preserving the surviving fine title.
    _paddle(monkeypatch, SimpleNamespace(predict=lambda image, **kw: _paddle_v3([
        ("TITLE", 0.85, (20, 20, 100, 40)),
        ("SHEET", 0.90, (120, 65, 180, 80))])))
    coarse, coarse_diag = _read(gray, build_engines(["paddle"]))
    merged, merge_diag = merge_passes(tokens, coarse)
    assert coarse_diag["token_count"] == 2
    assert [t.text for t in merged] == ["TITLE", "SHEET"]
    assert merged[0].conf == pytest.approx(0.88)
    assert merged[1].bbox == (120, 65, 180, 80)
    assert merged[1].candidates[0].engine == "paddle@coarse"
    assert merged[1].agreement == 0.5
    assert merge_diag == {"fine_only": 1, "conflicts": 0,
                          "recovered_from_coarse": 1, "merged_total": 2}


def test_scaled_coordinates_across_resize_policies(monkeypatch, gray):
    """
    OCR-IT-003: Exercise capped enlargement, forced reduction, and unchanged image size within one
    integration test. For each policy, run Tesseract, Paddle, and docTR adapters through the public
    OCR interface and verify actual image dimensions, outward coordinate rounding, matching source-
    image boxes, and three-engine consensus. Merge a native-resolution coarse scan to retain one
    SCALE token and recover only the missing SHEET region.
    """
    scenarios = [
        ("capped-upscale-rounds-outwards", 3.0, 300, (150, 300),
         (31, 31, 149, 89), (20, 20, 100, 60)),
        ("downscale-restores-source", 1.0, 100, (50, 100),
         (10, 10, 50, 30), (20, 20, 100, 60)),
        ("native-scale", 1.0, 0, (100, 200),
         (20, 20, 100, 60), (20, 20, 100, 60)),
    ]
    for scenario_name, upscale, max_side, expected_shape, box, expected_box in scenarios:
        with monkeypatch.context() as scenario_patch:
            calls = _tesseract(scenario_patch, [("SCALE", 95, box)])
            tokens, _ = _read(gray, build_engines(["tesseract"], tess_psms=(6,)),
                              upscale=upscale, max_side=max_side)
            assert calls[0]["image"].shape == expected_shape
            assert tokens[0].bbox == expected_box
            assert gray.shape == (100, 200)

            # Pixel polygons (Paddle) and normalized geometry (docTR) must map to the
            # same original-image coordinates as Tesseract under this resize policy.
            paddle_shapes = []

            def predict(image, **kwargs):
                paddle_shapes.append(image.shape)
                return _paddle_v3([("SCALE", 0.94, box)])

            _paddle(scenario_patch, SimpleNamespace(predict=predict))
            height, width = expected_shape
            doctr_calls = _doctr(scenario_patch, [("SCALE", 0.93,
                ((box[0] / width, box[1] / height),
                 (box[2] / width, box[3] / height)))])
            combined_engines = build_engines(["tesseract", "paddle", "doctr"],
                                             tess_psms=(6,))
            for engine in combined_engines:
                raw = engine.read(gray, upscale=upscale, max_side=max_side)
                assert len(raw) == 1
                assert raw[0].bbox == expected_box
            combined, diag = _read(gray, combined_engines,
                                   upscale=upscale, max_side=max_side)
            assert len(combined) == 1
            assert combined[0].bbox == expected_box
            assert combined[0].text == "SCALE"
            assert combined[0].agreement == 1.0
            assert {c.engine for c in combined[0].candidates} == {
                "tesseract", "paddle", "doctr"}
            assert diag["tokens_per_engine"] == {"tesseract": 1, "paddle": 1, "doctr": 1}
            assert paddle_shapes == [(*expected_shape, 3)] * 2
            assert all(images[0].shape == (*expected_shape, 3)
                       for images in doctr_calls["images"])
            assert len(doctr_calls["loads"]) == 1

            # The native-resolution coarse scan must align with the restored fine
            # coordinates, deduplicate SCALE and recover only the genuinely new line.
            _tesseract(scenario_patch, [("SCALE", 90, expected_box),
                                     ("SHEET", 88, (120, 65, 180, 80))])
            coarse, coarse_diag = _read(gray, [combined_engines[0]], upscale=1)
            merged, merge_diag = merge_passes(combined, coarse)
            assert coarse_diag["token_count"] == 2
            assert [t.text for t in merged] == ["SCALE", "SHEET"]
            assert merged[0].bbox == expected_box
            assert merged[0].agreement == 1.0
            assert len(merged[0].candidates) == 3
            assert merged[1].bbox == (120, 65, 180, 80)
            assert merged[1].candidates[0].engine == "tesseract@coarse"
            assert merge_diag == {"fine_only": 1, "conflicts": 0,
                                  "recovered_from_coarse": 1, "merged_total": 2}


def test_registry_configuration_runs_three_real_adapters(monkeypatch, gray):
    """
    OCR-IT-004: Verify language, segmentation, size-limit, image-format, and model-loading
    configuration across Paddle, Tesseract, and docTR adapters. Their real parsing and coordinate
    conversion must fuse a shared title into one token containing all three engine candidates. Reuse
    the configured Paddle client for a native-resolution coarse pass, then merge results to preserve
    the established title consensus and recover SHEET without duplicate initialization.
    """
    paddle_calls = []

    def predict(image, **kwargs):
        paddle_calls.append((image.copy(), kwargs))
        if image.shape[:2] == (100, 200):
            return _paddle_v3([
                (" Archive  Title ", 0.91, (20, 20, 100, 40)),
                ("SHEET", 0.90, (120, 65, 180, 80))])
        return _paddle_v3([(" Archive  Title ", 0.93, (40, 40, 200, 80))])

    constructors = _paddle(monkeypatch, SimpleNamespace(predict=predict))
    tess_calls = _tesseract(monkeypatch, [("ARCHIVE TITLE", 90, (40, 40, 200, 80))])
    doctr_calls = _doctr(monkeypatch, [
        ("archive title", 0.89, ((0.1, 0.2), (0.5, 0.4)))])
    engines = build_engines(["paddle", "tesseract", "doctr"],
                            lang="en", paddle_max_side=1024, tess_psms=(11,))
    tokens, diag = _read(gray, engines, upscale=2)
    assert len(tokens) == 1
    assert tokens[0].text == "Archive  Title"
    assert tokens[0].bbox == (20, 20, 100, 40)
    assert tokens[0].agreement == 1.0
    assert {c.engine for c in tokens[0].candidates} == {"paddle", "tesseract", "doctr"}
    assert diag == {"engines_used": ["paddle", "tesseract", "doctr"],
                    "engines_configured": ["paddle", "tesseract", "doctr"],
                    "tokens_per_engine": {"paddle": 1, "tesseract": 1, "doctr": 1},
                    "token_count": 1, "disagreements": 0}
    assert constructors[0]["lang"] == "en"
    assert constructors[0]["use_textline_orientation"] is True
    assert paddle_calls[0][0].shape == (200, 400, 3)
    assert paddle_calls[0][1] == {"max_side_limit": 1024}
    assert tess_calls[0]["config"] == "--psm 11"
    assert doctr_calls["loads"] == [{"pretrained": True}]
    assert doctr_calls["images"][0][0].shape == (200, 400, 3)

    # Reuse the configured Paddle adapter for the native-resolution coarse
    # scan: merging it must retain the three-engine consensus and add SHEET.
    coarse, coarse_diag = _read(gray, [engines[0]], upscale=1)
    merged, merge_diag = merge_passes(tokens, coarse)
    assert coarse_diag["token_count"] == 2
    assert len(constructors) == 1
    assert paddle_calls[1][0].shape == (100, 200, 3)
    assert paddle_calls[1][1] == {"max_side_limit": 1024}
    assert [t.text for t in merged] == ["Archive  Title", "SHEET"]
    assert merged[0].bbox == (20, 20, 100, 40)
    assert merged[0].agreement == 1.0
    assert len(merged[0].candidates) == 3
    assert merged[1].candidates[0].engine == "paddle@coarse"
    assert merge_diag == {"fine_only": 1, "conflicts": 0,
                          "recovered_from_coarse": 1, "merged_total": 2}


def test_registry_unavailability_and_dependency_recovery(monkeypatch, gray, caplog):
    """
    OCR-IT-005: Exercise two isolated registry scenarios: unknown or missing optional engines with a
    working Tesseract fallback, and complete engine unavailability requiring an explicit error.
    Restore the controlled SDK dependencies in each scenario and rebuild the configured adapters
    through the public interface. Verify resumed three-engine fusion, correct candidate origins and
    diagnostics, and repeated-pass merging without stale failure state or duplicate text.
    """
    # Missing optional engines leave a working fallback.
    with monkeypatch.context() as scenario_patch:
        caplog.clear()
        scenario_patch.setitem(sys.modules, "paddleocr", None)
        scenario_patch.setitem(sys.modules, "doctr", None)
        _tesseract(scenario_patch, [("FALLBACK", 90, (20, 20, 100, 40))])
        engines = build_engines(["unknown", "paddle", "doctr", "tesseract"])
        tokens, diag = _read(gray, engines)
        assert [e.name for e in engines] == ["tesseract"]
        assert tokens[0].text == "FALLBACK"
        assert diag["engines_configured"] == ["tesseract"]
        assert "Unknown OCR engine: unknown" in caplog.text
        assert "PaddleOCR is unavailable" in caplog.text
        assert "docTR is unavailable" in caplog.text

        # Rebuilding the same configuration after dependencies become available
        # must restore the optional adapters and their contribution to consensus.
        _paddle(scenario_patch, SimpleNamespace(predict=lambda image, **kw: _paddle_v3([
            ("FALLBACK", 0.94, (20, 20, 100, 40))])))
        _doctr(scenario_patch, [("FALLBACK", 0.92, ((0.1, 0.2), (0.5, 0.4)))])
        restored = build_engines(["unknown", "paddle", "doctr", "tesseract"])
        restored_tokens, restored_diag = _read(gray, restored)
        assert [e.name for e in restored] == ["paddle", "doctr", "tesseract"]
        assert len(restored_tokens) == 1
        assert restored_tokens[0].text == tokens[0].text
        assert restored_tokens[0].bbox == tokens[0].bbox
        assert restored_tokens[0].agreement == 1.0
        assert {c.engine for c in restored_tokens[0].candidates} == {
            "paddle", "doctr", "tesseract"}
        assert restored_diag["tokens_per_engine"] == {
            "paddle": 1, "doctr": 1, "tesseract": 1}

        # Results produced before and after dependency recovery must merge without
        # counting the same fallback word twice.
        merged, merge_diag = merge_passes(tokens, restored_tokens)
        assert len(merged) == 1
        assert merged[0].text == "FALLBACK"
        assert merged[0].bbox == (20, 20, 100, 40)
        assert merge_diag == {"fine_only": 1, "conflicts": 0,
                              "recovered_from_coarse": 0, "merged_total": 1}

    # Complete unavailability fails explicitly and can recover.
    with monkeypatch.context() as scenario_patch:
        caplog.clear()
        scenario_patch.setitem(sys.modules, "paddleocr", None)
        scenario_patch.setitem(sys.modules, "doctr", None)
        _tesseract(scenario_patch, [], version_error=OSError("binary not installed"))
        with pytest.raises(RuntimeError, match="No OCR engine is available"):
            build_engines(["paddle", "tesseract", "doctr", "unknown"])

        # Dependency failure must not poison a later build: once the SDK/binary
        # boundaries recover, the same registry configuration can complete OCR.
        _tesseract(scenario_patch, [("READY", 90, (20, 20, 100, 40))])
        _paddle(scenario_patch, SimpleNamespace(predict=lambda image, **kw: _paddle_v3([
            ("READY", 0.94, (20, 20, 100, 40))])))
        _doctr(scenario_patch, [("READY", 0.92, ((0.1, 0.2), (0.5, 0.4)))])
        engines = build_engines(["paddle", "tesseract", "doctr", "unknown"])
        tokens, diag = _read(gray, engines)
        assert [e.name for e in engines] == ["paddle", "tesseract", "doctr"]
        assert len(tokens) == 1
        assert tokens[0].text == "READY"
        assert tokens[0].bbox == (20, 20, 100, 40)
        assert tokens[0].agreement == 1.0
        assert {c.engine for c in tokens[0].candidates} == {
            "paddle", "tesseract", "doctr"}
        assert diag["tokens_per_engine"] == {"paddle": 1, "tesseract": 1, "doctr": 1}
        assert diag["token_count"] == 1
        assert diag["disagreements"] == 0

        # A fresh request after recovery must remain usable, then merge with the
        # previous result without a stale unavailable-engine state or duplication.
        coarse, coarse_diag = _read(gray, engines)
        merged, merge_diag = merge_passes(tokens, coarse)
        assert coarse_diag == diag
        assert len(merged) == 1
        assert merged[0].text == "READY"
        assert merged[0].bbox == (20, 20, 100, 40)
        assert {c.engine for c in merged[0].candidates} == {
            "paddle", "tesseract", "doctr"}
        assert merge_diag == {"fine_only": 1, "conflicts": 0,
                              "recovered_from_coarse": 0, "merged_total": 1}


def test_paddle_api_compatibility_across_fusion_and_merge(monkeypatch, gray):
    """
    OCR-IT-006: Exercise legacy Paddle constructor and ocr argument fallbacks separately from
    predict retry without an unsupported max_side_limit keyword. Preserve constructor and invocation
    counts, malformed-result filtering, missing-confidence filtering, and coordinate restoration. In
    each compatibility scenario, corroborate the real Paddle adapter output with Tesseract, then
    merge repeated passes to retain one correctly located token with both engine candidates and a
    reused client.
    """
    # Legacy constructor and ocr signatures remain compatible.
    with monkeypatch.context() as scenario_patch:
        calls = []

        def legacy_ocr(image, **kwargs):
            calls.append((image.copy(), kwargs))
            if kwargs:
                raise TypeError("cls was removed")
            return [None, [[_poly((40, 40, 200, 80)), (" LEGACY ", 0.94)],
                           [None], [None, ("broken", 0.1)],
                           [_poly((0, 0, 1, 1)), (" ", 0.99)]]]

        constructors = _paddle(scenario_patch, SimpleNamespace(ocr=legacy_ocr),
                                accept=lambda options: set(options) == {"lang"})
        engines = build_engines(["paddle"])
        tokens, diag = _read(gray, engines, upscale=2)
        assert len(constructors) == 5
        assert [call[1] for call in calls] == [{"cls": True}, {}]
        assert tokens[0].text == "LEGACY"
        assert tokens[0].bbox == (20, 20, 100, 40)
        assert diag["tokens_per_engine"] == {"paddle": 1}
        # Repeating the real adapter must reuse the initialized external client.
        _read(gray, engines, upscale=2)
        assert len(constructors) == 5

        # The compatibility fallback must still corroborate another real adapter.
        _tesseract(scenario_patch, [("legacy", 91, (40, 40, 200, 80))])
        engines.extend(build_engines(["tesseract"], tess_psms=(6,)))
        corroborated, combined_diag = _read(gray, engines, upscale=2)
        assert [token.text for token in corroborated] == ["LEGACY"]
        assert corroborated[0].bbox == (20, 20, 100, 40)
        assert corroborated[0].agreement == 1.0
        assert {candidate.engine for candidate in corroborated[0].candidates} == {
            "paddle", "tesseract"}
        assert combined_diag["tokens_per_engine"] == {"paddle": 1, "tesseract": 1}
        assert len(constructors) == 5
        assert [call[1] for call in calls] == [{"cls": True}, {}] * 3
        repeated, repeated_diag = _read(gray, engines, upscale=2)
        merged, merge_diag = merge_passes(corroborated, repeated)
        assert [token.text for token in merged] == ["LEGACY"]
        assert merged[0].agreement == 1.0
        assert {candidate.engine for candidate in merged[0].candidates} == {
            "paddle", "tesseract"}
        assert repeated_diag == combined_diag
        assert merge_diag == {"fine_only": 1, "conflicts": 0,
                              "recovered_from_coarse": 0, "merged_total": 1}
        assert len(constructors) == 5

    # The predict call retries without an unsupported keyword.
    with monkeypatch.context() as scenario_patch:
        calls = []

        def predict(image, **kwargs):
            calls.append(kwargs)
            if kwargs:
                raise TypeError("max_side_limit unsupported")
            return [SimpleNamespace(json={"res": {
                "rec_texts": ["PLAN", "MISSING CONFIDENCE"],
                "rec_scores": [0.91],
                "dt_polys": [_poly((20, 20, 100, 40))]}})]

        constructors = _paddle(scenario_patch, SimpleNamespace(predict=predict))
        engines = build_engines(["paddle"])
        tokens, diag = _read(gray, engines)
        assert calls == [{"max_side_limit": 236}, {}]
        assert [t.text for t in tokens] == ["PLAN"]
        assert diag["tokens_per_engine"] == {"paddle": 2}

        # An incompatible keyword must not prevent cross-engine agreement on retry.
        _tesseract(scenario_patch, [("plan", 89, (20, 20, 100, 40))])
        engines.extend(build_engines(["tesseract"], tess_psms=(6,)))
        corroborated, combined_diag = _read(gray, engines)
        assert [token.text for token in corroborated] == ["PLAN"]
        assert corroborated[0].bbox == (20, 20, 100, 40)
        assert corroborated[0].agreement == 1.0
        assert {candidate.engine for candidate in corroborated[0].candidates} == {
            "paddle", "tesseract"}
        assert combined_diag["tokens_per_engine"] == {"paddle": 2, "tesseract": 1}
        assert len(constructors) == 1
        assert calls == [{"max_side_limit": 236}, {}] * 2
        repeated, repeated_diag = _read(gray, engines)
        merged, merge_diag = merge_passes(corroborated, repeated)
        assert [token.text for token in merged] == ["PLAN"]
        assert merged[0].agreement == 1.0
        assert {candidate.engine for candidate in merged[0].candidates} == {
            "paddle", "tesseract"}
        assert repeated_diag == combined_diag
        assert merge_diag == {"fine_only": 1, "conflicts": 0,
                              "recovered_from_coarse": 0, "merged_total": 1}
        assert len(constructors) == 1
        assert calls == [{"max_side_limit": 236}, {}] * 3


def test_paddle_initialization_and_inference_failure_recovery(monkeypatch, gray, caplog):
    """
    OCR-IT-007: Exercise Paddle inference failure and complete initialization failure as separate
    recovery scenarios. Preserve Tesseract output and the different engines-used and agreement
    diagnostics for each failure type. Restore the controlled external dependency or device, reuse
    the same real adapter, and verify resumed fusion. Merge degraded and recovered passes to retain
    original text, recover the new region, and preserve coarse-pass candidate provenance.
    """
    # A transient inference failure preserves and then restores output.
    with monkeypatch.context() as scenario_patch:
        caplog.clear()
        device = {"available": False}

        def predict(image, **kwargs):
            if not device["available"]:
                raise RuntimeError("device unavailable")
            return _paddle_v3([("survivor", 0.93, (20, 20, 100, 40)),
                               ("RECOVERED", 0.92, (120, 65, 180, 80))])

        constructors = _paddle(scenario_patch, SimpleNamespace(predict=predict))
        _tesseract(scenario_patch, [("SURVIVOR", 95, (20, 20, 100, 40))])
        engines = build_engines(["paddle", "tesseract"])
        tokens, diag = _read(gray, engines)
        assert tokens[0].text == "SURVIVOR"
        assert tokens[0].agreement == 0.5
        assert diag["tokens_per_engine"] == {"paddle": 0, "tesseract": 1}
        assert diag["engines_used"] == ["paddle", "tesseract"]
        assert "PaddleOCR text recognition failed: device unavailable" in caplog.text

        # A transient inference error must not permanently disable the cached client.
        device["available"] = True
        recovered, recovered_diag = _read(gray, engines)
        assert [token.text for token in recovered] == ["SURVIVOR", "RECOVERED"]
        assert recovered[0].agreement == 1.0
        assert {candidate.engine for candidate in recovered[0].candidates} == {
            "paddle", "tesseract"}
        assert recovered[1].bbox == (120, 65, 180, 80)
        assert recovered[1].agreement == 0.5
        assert recovered_diag["tokens_per_engine"] == {"paddle": 2, "tesseract": 1}
        assert len(constructors) == 1
        # Merge the degraded first pass with the now-recovered second pass.
        merged, merge_diag = merge_passes(tokens, recovered)
        assert [token.text for token in merged] == ["SURVIVOR", "RECOVERED"]
        assert merged[0].candidates[0].engine == "tesseract"
        assert merged[1].candidates[0].engine == "paddle@coarse"
        assert merged[1].bbox == (120, 65, 180, 80)
        assert merge_diag == {"fine_only": 1, "conflicts": 0,
                              "recovered_from_coarse": 1, "merged_total": 2}

    # Initialization failure is isolated and the same adapter can recover.
    with monkeypatch.context() as scenario_patch:
        caplog.clear()
        constructors = _paddle(scenario_patch, object(), accept=lambda options: False)
        _tesseract(scenario_patch, [("AVAILABLE", 95, (20, 20, 100, 40))])
        engines = build_engines(["paddle", "tesseract"])
        tokens, diag = _read(gray, engines)
        assert len(constructors) == 5
        assert tokens[0].text == "AVAILABLE"
        assert tokens[0].agreement == 1.0
        assert diag["engines_configured"] == ["paddle", "tesseract"]
        assert diag["engines_used"] == ["tesseract"]
        assert "paddle crashed and will be skipped" in caplog.text

        # After the SDK becomes usable, the same adapter can retry initialization.
        recovered_constructors = _paddle(scenario_patch, SimpleNamespace(
            predict=lambda image, **kwargs: _paddle_v3([
                ("available", 0.93, (20, 20, 100, 40)),
                ("RESTORED", 0.9, (120, 65, 180, 80))])))
        recovered, recovered_diag = _read(gray, engines)
        assert [token.text for token in recovered] == ["AVAILABLE", "RESTORED"]
        assert recovered[0].agreement == 1.0
        assert {candidate.engine for candidate in recovered[0].candidates} == {
            "paddle", "tesseract"}
        assert recovered[1].bbox == (120, 65, 180, 80)
        assert recovered[1].agreement == 0.5
        assert recovered_diag["engines_used"] == ["paddle", "tesseract"]
        assert recovered_diag["tokens_per_engine"] == {"paddle": 2, "tesseract": 1}
        repeated, repeated_diag = _read(gray, engines)
        assert repeated == recovered
        assert repeated_diag == recovered_diag
        assert len(recovered_constructors) == 1
        merged, merge_diag = merge_passes(tokens, repeated)
        assert [token.text for token in merged] == ["AVAILABLE", "RESTORED"]
        assert merged[0].candidates[0].engine == "tesseract"
        assert merged[1].candidates[0].engine == "paddle@coarse"
        assert merged[1].bbox == (120, 65, 180, 80)
        assert merge_diag == {"fine_only": 1, "conflicts": 0,
                              "recovered_from_coarse": 1, "merged_total": 2}


def test_doctr_loading_and_inference_failure_isolation(monkeypatch, gray, caplog):
    """
    OCR-IT-008: Exercise docTR model-loading failure and inference exceptions in isolated scenarios.
    Loading failure must remain cached across passes without model calls; inference failure must
    reuse the loaded model while reporting empty docTR results. In both scenarios, real Tesseract
    and Paddle adapters continue recognition and fusion. Merge repeated passes while preserving
    text, engine provenance, agreement values, and the expected loading and inference counts.
    """
    # A failed model download is cached across repeated passes.
    with monkeypatch.context() as scenario_patch:
        caplog.clear()
        calls = _doctr(scenario_patch, load_error=OSError("offline model store"))
        _tesseract(scenario_patch, [("OFFLINE", 90, (20, 20, 100, 40))])
        engines = build_engines(["doctr", "tesseract"])
        fine, _ = _read(gray, engines)
        coarse, _ = _read(gray, engines)
        merged, diag = merge_passes(fine, coarse)
        assert calls["loads"] == [{"pretrained": True}]
        assert calls["images"] == []
        assert [t.text for t in merged] == ["OFFLINE"]
        assert diag["conflicts"] == 0
        assert "docTR has been disabled" in caplog.text

        # Adding another survivor must not re-trigger the failed model download.
        constructors = _paddle(scenario_patch, SimpleNamespace(
            predict=lambda image, **kwargs: _paddle_v3([
                ("offline", 0.89, (20, 20, 100, 40)),
                ("SECOND REGION", 0.92, (120, 65, 180, 80))])))
        engines.extend(build_engines(["paddle"]))
        fine, fine_diag = _read(gray, engines)
        coarse, coarse_diag = _read(gray, engines)
        assert [token.text for token in fine] == ["OFFLINE", "SECOND REGION"]
        assert fine[0].agreement == pytest.approx(0.667)
        assert {candidate.engine for candidate in fine[0].candidates} == {
            "paddle", "tesseract"}
        assert fine_diag["tokens_per_engine"] == {
            "doctr": 0, "tesseract": 1, "paddle": 2}
        assert coarse_diag == fine_diag
        merged, merge_diag = merge_passes(fine, coarse)
        assert [token.text for token in merged] == ["OFFLINE", "SECOND REGION"]
        assert merge_diag["conflicts"] == merge_diag["recovered_from_coarse"] == 0
        assert calls["loads"] == [{"pretrained": True}]
        assert calls["images"] == []
        assert len(constructors) == 1

    # Inference exceptions are isolated while the loaded model is reused.
    with monkeypatch.context() as scenario_patch:
        caplog.clear()
        calls = _doctr(scenario_patch, inference_error=RuntimeError("tensor device failure"))
        engines = build_engines(["doctr"])
        tokens, diag = _read(gray, engines)
        assert tokens == []
        assert len(calls["images"]) == 1
        assert diag["tokens_per_engine"] == {"doctr": 0}
        assert diag["token_count"] == diag["disagreements"] == 0
        assert "docTR text recognition failed" in caplog.text

        # Other adapters must still align and fuse after the same docTR model fails.
        _tesseract(scenario_patch, [("SURVIVOR", 95, (20, 20, 100, 40))])
        constructors = _paddle(scenario_patch, SimpleNamespace(
            predict=lambda image, **kwargs: _paddle_v3([
                ("survivor", 0.93, (20, 20, 100, 40)),
                ("SECOND REGION", 0.91, (120, 65, 180, 80))])))
        engines.extend(build_engines(["paddle", "tesseract"]))
        surviving, survivor_diag = _read(gray, engines)
        assert [token.text for token in surviving] == ["SURVIVOR", "SECOND REGION"]
        assert surviving[0].agreement == pytest.approx(0.667)
        assert {candidate.engine for candidate in surviving[0].candidates} == {
            "paddle", "tesseract"}
        assert surviving[1].bbox == (120, 65, 180, 80)
        assert surviving[1].agreement == pytest.approx(0.333)
        assert survivor_diag["tokens_per_engine"] == {
            "doctr": 0, "paddle": 2, "tesseract": 1}
        repeated, repeated_diag = _read(gray, engines)
        assert repeated == surviving
        assert repeated_diag == survivor_diag
        assert len(calls["images"]) == 3
        assert calls["loads"] == [{"pretrained": True}]
        assert len(constructors) == 1
        merged, merge_diag = merge_passes(surviving, repeated)
        assert [token.text for token in merged] == ["SURVIVOR", "SECOND REGION"]
        assert merged[0].agreement == pytest.approx(0.667)
        assert {candidate.engine for candidate in merged[0].candidates} == {
            "paddle", "tesseract"}
        assert merge_diag == {"fine_only": 2, "conflicts": 0,
                              "recovered_from_coarse": 0, "merged_total": 2}


def test_engine_conflicts_use_best_confidence_and_keep_alternatives(monkeypatch, gray):
    """
    OCR-IT-009: Fuse conflicting Paddle PLAN 8 and Tesseract PLAN B candidates from the same region
    through the public OCR interface. Select the higher-confidence PLAN 8, retain both candidate
    texts, combine their coordinates, and keep a separate text region. Read another actual adapter
    pass and merge it with the first, preserving disagreement values and alternative evidence
    without duplicate output or spurious recovery.
    """
    _paddle(monkeypatch, SimpleNamespace(predict=lambda image, **kwargs:
        _paddle_v3([("PLAN 8", 0.97, (18, 18, 102, 42)),
                    ("SEPARATE", 0.8, (120, 65, 180, 80))])))
    _tesseract(monkeypatch, [("PLAN B", 92, (20, 20, 100, 40))])
    engines = build_engines(["paddle", "tesseract"])
    tokens, diag = _read(gray, engines)
    assert [t.text for t in tokens] == ["PLAN 8", "SEPARATE"]
    assert tokens[0].bbox == (18, 18, 102, 42)
    assert tokens[0].conf == pytest.approx(0.97)
    assert [c.text for c in tokens[0].candidates] == ["PLAN 8", "PLAN B"]
    assert [t.agreement for t in tokens] == [0.5, 0.5]
    assert diag["disagreements"] == 2
    # Repeated passes must retain the original cross-engine disagreement.
    repeated, repeated_diag = _read(gray, engines)
    merged, merge_diag = merge_passes(tokens, repeated)
    assert [token.text for token in merged] == ["PLAN 8", "SEPARATE"]
    assert [candidate.text for candidate in merged[0].candidates] == ["PLAN 8", "PLAN B"]
    assert [token.agreement for token in merged] == [0.5, 0.5]
    assert repeated_diag == diag
    assert merge_diag == {"fine_only": 2, "conflicts": 0,
                          "recovered_from_coarse": 0, "merged_total": 2}


def test_same_engine_overlap_cannot_inflate_consensus(monkeypatch, gray):
    """
    OCR-IT-010: Run multiple Tesseract page segmentation modes that return overlapping DRAWING and
    DRAW1NG hypotheses, together with a corroborating Paddle adapter. Real fusion must allow at most
    one candidate per engine rather than counting segmentation modes as independent votes. Read and
    merge a second actual pass, retaining one DRAWING result, full agreement, two distinct engine
    candidates, and unchanged diagnostic counts.
    """
    _tesseract(monkeypatch, {
        6: [("DRAWING", 91, (20, 20, 100, 40))],
        11: [("DRAW1NG", 99, (20, 20, 100, 40))]})
    _paddle(monkeypatch, SimpleNamespace(predict=lambda image, **kwargs:
        _paddle_v3([("DRAWING", 0.9, (20, 20, 100, 40))])))
    engines = build_engines(["tesseract", "paddle"])
    tokens, diag = _read(gray, engines)
    assert len(tokens) == 1
    assert tokens[0].text == "DRAWING"
    assert tokens[0].agreement == 1.0
    assert len(tokens[0].candidates) == 2
    assert diag["tokens_per_engine"] == {"tesseract": 2, "paddle": 1}
    # A second actual pass must not create extra votes or duplicate candidates.
    repeated, repeated_diag = _read(gray, engines)
    merged, merge_diag = merge_passes(tokens, repeated)
    assert [token.text for token in merged] == ["DRAWING"]
    assert merged[0].agreement == 1.0
    assert len(merged[0].candidates) == 2
    assert {candidate.engine for candidate in merged[0].candidates} == {
        "paddle", "tesseract"}
    assert repeated_diag == diag
    assert merge_diag == {"fine_only": 1, "conflicts": 0,
                          "recovered_from_coarse": 0, "merged_total": 1}


def test_merge_real_adapter_passes_retains_conflict_and_recovers_missing_text(monkeypatch, gray):
    """
    OCR-IT-011: Run separate fine and coarse scans in which Tesseract and Paddle each contribute to
    real cross-engine fusion. Merge those fused results to preserve the fine title and its
    coordinates, retain the conflicting coarse title with its source box and provenance, and recover
    a missing region. Verify title agreement of 0.667, recovered agreement of 0.5, both recovered
    engine origins, and exact merge diagnostics.
    """
    responses = {
        6: [("FINE TITLE", 96, (20, 20, 100, 40))],
        11: [("COARSE TITLE", 90, (20, 20, 100, 40)),
             ("RECOVERED", 88, (120, 65, 180, 80))]}
    _tesseract(monkeypatch, responses)
    # Both passes are corroborated independently before their fused tokens merge.
    paddle_responses = iter([
        _paddle_v3([("FINE TITLE", 0.94, (20, 20, 100, 40))]),
        _paddle_v3([("COARSE TITLE", 0.89, (20, 20, 100, 40)),
                    ("RECOVERED", 0.87, (120, 65, 180, 80))])])
    _paddle(monkeypatch, SimpleNamespace(predict=lambda image, **kwargs:
        next(paddle_responses)))
    engines = build_engines(["tesseract", "paddle"])
    fine, _ = _read(gray, engines, psms=(6,))
    coarse, _ = _read(gray, engines, psms=(11,))
    assert all({c.engine for c in token.candidates} == {"tesseract", "paddle"}
               for token in fine + coarse)
    assert all(token.agreement == 1.0 for token in fine + coarse)
    merged, diag = merge_passes(fine, coarse)
    assert [t.text for t in merged] == ["FINE TITLE", "RECOVERED"]
    assert merged[0].bbox == (20, 20, 100, 40)
    assert merged[0].agreement == pytest.approx(0.667)
    assert merged[0].candidates[-1].text == "COARSE TITLE"
    assert merged[0].candidates[-1].engine == "tesseract@coarse"
    assert merged[0].candidates[-1].source_bbox == (20, 20, 100, 40)
    assert merged[1].agreement == 0.5
    assert merged[1].candidates[0].engine == "tesseract@coarse"
    assert {c.engine for c in merged[1].candidates} == {
        "tesseract@coarse", "paddle@coarse"}
    assert diag == {"fine_only": 0, "conflicts": 1,
                    "recovered_from_coarse": 1, "merged_total": 2}


def test_related_partial_overlap_text_matching(monkeypatch, gray):
    """
    OCR-IT-012: Exercise three related-text variants in isolated fine and coarse scans: case-and-
    space normalization, substring containment, and fuzzy matching of ARCHIVE against ARCH1VE. Each
    scan must first fuse real Tesseract and Paddle adapter results. Merge partially overlapping
    regions while retaining fine coordinates and text; normalized agreement produces no conflict,
    while the other variants preserve a coarse alternative and yield agreement of 0.667.
    """
    scenarios = [
        ("ARCHIVE", " archive ", 1, 0),
        ("ARCHIVE", "ARCHIVES", 1, 1),
        ("ARCHIVE", "ARCH1VE", 1, 1),
    ]
    for fine_text, coarse_text, expected_count, expected_conflicts in scenarios:
        with monkeypatch.context() as scenario_patch:
            _check_partial_overlap_merge(scenario_patch, gray, fine_text, coarse_text,
                                         expected_count, expected_conflicts)


def test_unrelated_partial_overlap_text_remains_separate(monkeypatch, gray):
    """
    OCR-IT-013: Verify the negative text-matching case using ARCHIVE and unrelated LEVEL TWO in
    partially overlapping regions. Both fine and coarse scans must first fuse real Tesseract and
    Paddle adapter output through the public interface. Pass merging must retain both texts without
    declaring a matched-text conflict, preserve the fine coordinates, and label both recovered
    engine candidates as coarse while capping recovered agreement at 0.5.
    """
    with monkeypatch.context() as scenario_patch:
        _check_partial_overlap_merge(scenario_patch, gray, "ARCHIVE", "LEVEL TWO", 2, 0)


def test_coarse_line_containing_fine_word_is_not_recovered_twice(monkeypatch, gray):
    """
    OCR-IT-014: Read corroborating Tesseract and Paddle fine results for TITLE, then a coarse scan
    containing both its word box and a larger surrounding line box. Keep those coarse regions
    distinct during cross-engine fusion using the configured overlap threshold. Subsequent pass
    merging must retain one TITLE token at the fine coordinates, with two engine candidates and full
    agreement, without recovering duplicate text.
    """
    _tesseract(monkeypatch, {
        6: [("TITLE", 96, (20, 20, 100, 40))],
        11: [("TITLE", 95, (20, 20, 100, 40)),
             ("TITLE", 88, (0, 10, 190, 50))]})
    paddle_responses = iter([
        _paddle_v3([("TITLE", 0.94, (20, 20, 100, 40))]),
        _paddle_v3([("TITLE", 0.93, (20, 20, 100, 40)),
                    ("TITLE", 0.87, (0, 10, 190, 50))])])
    _paddle(monkeypatch, SimpleNamespace(predict=lambda image, **kwargs:
        next(paddle_responses)))
    engines = build_engines(["tesseract", "paddle"])
    fine, _ = _read(gray, engines, psms=(6,))
    # A high threshold keeps the word and containing line as distinct tokens.
    coarse, _ = read_all(gray, engines, upscale=1.0, iou_thresh=0.8,
                         min_conf=0.5, psms=(11,))
    assert len(coarse) == 2
    assert all({c.engine for c in token.candidates} == {"tesseract", "paddle"}
               for token in fine + coarse)
    merged, diag = merge_passes(fine, coarse)
    assert len(merged) == 1
    assert diag["recovered_from_coarse"] == 0

    assert merged[0].text == "TITLE"
    assert merged[0].bbox == (20, 20, 100, 40)
    assert merged[0].agreement == 1.0
    assert len(merged[0].candidates) == 2


def test_paddle_out_of_bounds_mapping_emits_diagnostics(monkeypatch, gray, caplog):
    """
    OCR-IT-015: Verify that an out-of-bounds Paddle box emits both coordinate diagnostics while
    fusing with a valid overlapping Tesseract result. Read a second actual segmentation pass through
    the same public OCR interface and merge the results. Retain one token, both engine candidates,
    full text agreement, and the original out-of-bounds coordinates. This scenario verifies
    diagnostic reporting and evidence preservation, not automatic clipping or coordinate correction.
    """
    _paddle(monkeypatch, SimpleNamespace(predict=lambda image, **kwargs:
        _paddle_v3([("OUTSIDE", 0.95, (20, 20, 240, 120))])))
    # A valid Tesseract box overlaps the suspect Paddle region. Fusion must
    # retain the evidence and its diagnostic rather than silently clipping it.
    _tesseract(monkeypatch, [("OUTSIDE", 90, (20, 20, 200, 100))])
    tokens, diag = _read(gray, build_engines(["paddle", "tesseract"]))
    assert tokens[0].bbox == (20, 20, 240, 120)
    assert diag["token_count"] == 1
    assert "Coordinate restoration error" in caplog.text
    assert "Coordinates out of bounds" in caplog.text

    assert tokens[0].text == "OUTSIDE"
    assert tokens[0].agreement == 1.0
    assert {c.engine for c in tokens[0].candidates} == {"paddle", "tesseract"}
    assert diag["tokens_per_engine"] == {"paddle": 1, "tesseract": 1}

    # A second segmentation pass must not hide the coordinate problem during
    # merging: the current contract retains the original box and warning.
    coarse, coarse_diag = _read(gray, build_engines(["paddle", "tesseract"]),
                                psms=(11,))
    merged, merge_diag = merge_passes(tokens, coarse)
    assert len(merged) == 1
    assert merged[0].bbox == (20, 20, 240, 120)
    assert merged[0].agreement == 1.0
    assert coarse_diag["tokens_per_engine"] == {"paddle": 1, "tesseract": 1}
    assert merge_diag == {"fine_only": 1, "conflicts": 0,
                          "recovered_from_coarse": 0, "merged_total": 1}


def test_paddle_json_converted_arrays_survive_adapter_and_fusion(monkeypatch, gray):
    """
    OCR-IT-016: Verify Paddle adapter parsing when serialized result coordinates and confidence
    values are ordinary lists converted from NumPy arrays. Fuse JSON PLAN with its case-normalized
    Tesseract equivalent through the public OCR interface, retaining both original candidate texts.
    Read and merge a second actual pass, preserving one correctly located token with the expected
    confidence, full agreement, and exactly two engine candidates.
    """
    # PaddleX's JsonMixin recursively converts ndarray fields to lists. Model
    # this public serialization boundary, not an unsupported raw ndarray dict.
    polygons = np.asarray([_poly((20, 20, 100, 40))], dtype=np.float32)
    scores = np.asarray([0.95], dtype=np.float32)
    result = [SimpleNamespace(json={"res": {
        "rec_texts": ["JSON PLAN"], "rec_scores": scores.tolist(),
        "rec_polys": polygons.tolist()}})]
    _paddle(monkeypatch, SimpleNamespace(predict=lambda image, **kwargs: result))
    _tesseract(monkeypatch, [("json plan", 90, (20, 20, 100, 40))])
    tokens, diag = _read(gray, build_engines(["paddle", "tesseract"]))
    assert [t.text for t in tokens] == ["JSON PLAN"]
    assert tokens[0].bbox == (20, 20, 100, 40)
    assert tokens[0].conf == pytest.approx(0.95)
    assert diag["engines_used"] == ["paddle", "tesseract"]

    assert tokens[0].agreement == 1.0
    assert {c.engine for c in tokens[0].candidates} == {"paddle", "tesseract"}
    assert [c.text for c in tokens[0].candidates] == ["JSON PLAN", "json plan"]
    assert diag["tokens_per_engine"] == {"paddle": 1, "tesseract": 1}

    # Parsed JSON candidates must remain usable when a second OCR pass is
    # normalized and merged with the first pass's cross-engine result.
    coarse, _ = _read(gray, build_engines(["paddle", "tesseract"]), psms=(11,))
    merged, merge_diag = merge_passes(tokens, coarse)
    assert [token.text for token in merged] == ["JSON PLAN"]
    assert merged[0].bbox == (20, 20, 100, 40)
    assert merged[0].conf == pytest.approx(0.95)
    assert merged[0].agreement == 1.0
    assert len(merged[0].candidates) == 2
    assert merge_diag == {"fine_only": 1, "conflicts": 0,
                          "recovered_from_coarse": 0, "merged_total": 1}


def test_paddle_multiple_pages_and_confidence_boundary(monkeypatch, gray):
    """
    OCR-IT-017: Exercise empty Paddle pages, blank text, and the shared confidence threshold across
    Paddle and Tesseract adapters. Real normalization and fusion must retain confidence 0.5 and
    reject 0.499 while producing only TITLE and SHEET with agreeing engine candidates. Read another
    actual pass and merge it with the first, confirming that exact-boundary text survives and
    rejected text does not return as coarse-only recovery.
    """
    result = [None, SimpleNamespace(json={"rec_texts": []}),
              SimpleNamespace(json={
                  "rec_texts": ["TITLE", "SHEET", "BELOW", " "],
                  "rec_scores": [0.95, 0.5, 0.499, 0.99],
                  "rec_polys": [_poly((20, 20, 100, 40)),
                                _poly((120, 65, 180, 80)),
                                _poly((0, 85, 50, 95)),
                                _poly((50, 85, 100, 95))]})]
    _paddle(monkeypatch, SimpleNamespace(predict=lambda image, **kwargs: result))
    # The two adapters normalize different SDK confidence scales before the
    # shared fusion threshold, including values exactly at and below 0.5.
    _tesseract(monkeypatch, [
        ("TITLE", 90, (20, 20, 100, 40)),
        ("SHEET", 50, (120, 65, 180, 80)),
        ("BELOW", 49.9, (0, 85, 50, 95)),
        (" ", 99, (50, 85, 100, 95))])
    tokens, diag = _read(gray, build_engines(["paddle", "tesseract"]))
    assert [t.text for t in tokens] == ["TITLE", "SHEET"]
    assert tokens[1].conf == 0.5
    assert diag["tokens_per_engine"] == {"paddle": 3, "tesseract": 3}
    assert diag["token_count"] == 2

    assert all(token.agreement == 1.0 for token in tokens)
    assert all({c.engine for c in token.candidates} == {"paddle", "tesseract"}
               for token in tokens)
    assert [c.conf for c in tokens[1].candidates] == [0.5, 0.5]

    # The below-threshold item must not return as coarse-only text during
    # pass merging; the exact-boundary item must survive both passes.
    coarse, _ = _read(gray, build_engines(["paddle", "tesseract"]), psms=(11,))
    merged, merge_diag = merge_passes(tokens, coarse)
    assert [token.text for token in merged] == ["TITLE", "SHEET"]
    assert merged[1].conf == 0.5
    assert all(token.agreement == 1.0 for token in merged)
    assert merge_diag == {"fine_only": 2, "conflicts": 0,
                          "recovered_from_coarse": 0, "merged_total": 2}


def test_public_ocr_package_connects_adapters_raw_tokens_and_fusion(monkeypatch, gray):
    """
    OCR-IT-018: Verify that interfaces exported through __init__.py construct real Engine instances
    and RawToken results. For both fine and coarse scans, compare explicit engine.read followed by
    fuse against the public read_all orchestration, including low-confidence filtering. Merge the
    resulting passes and preserve expected texts, coordinate unions, confidence values, agreement,
    candidate origins, and diagnostic counts without duplicates, conflicts, or additional coarse-
    only recovery.
    """
    _tesseract(monkeypatch, [
        ("reference", 91, (40, 40, 200, 80)),
        ("SHEET", 88, (240, 130, 360, 160)),
        ("LOW", 20, (0, 170, 100, 190))])
    _paddle(monkeypatch, SimpleNamespace(predict=lambda image, **kwargs:
        _paddle_v3([("REFERENCE", 0.96, (40, 40, 204, 80))])))
    engines = build_engines(["tesseract", "paddle"], tess_psms=(6,))
    assert all(isinstance(engine, Engine) for engine in engines)
    raw_tokens = [raw for engine in engines
                  for raw in engine.read(gray, upscale=2)]
    assert len(raw_tokens) == 4
    assert all(isinstance(raw, RawToken) for raw in raw_tokens)

    tokens = fuse(raw_tokens, iou_thresh=0.4, min_conf=0.5,
                  n_engines=len(engines))
    assert [token.text for token in tokens] == ["REFERENCE", "SHEET"]
    assert tokens[0].bbox == (20, 20, 102, 40)
    assert tokens[0].conf == pytest.approx(0.96)
    assert tokens[0].agreement == 1.0
    assert {candidate.engine for candidate in tokens[0].candidates} == {
        "tesseract", "paddle"}
    assert tokens[1].bbox == (120, 65, 180, 80)
    assert tokens[1].agreement == 0.5

    orchestrated_tokens, diag = read_all(
        gray, engines, upscale=2, iou_thresh=0.4, min_conf=0.5)
    assert orchestrated_tokens == tokens
    assert diag["tokens_per_engine"] == {"tesseract": 3, "paddle": 1}
    assert diag["token_count"] == 2

    # Complete the caller's fine/coarse flow through public engine/read APIs,
    # then verify that the shared merge preserves both public fusion results.
    coarse_raws = [raw for engine in engines
                   for raw in engine.read(gray, upscale=2, psms=(11,))]
    coarse_expected = fuse(coarse_raws, iou_thresh=0.4, min_conf=0.5,
                           n_engines=len(engines))
    coarse, coarse_diag = read_all(
        gray, engines, upscale=2, iou_thresh=0.4, min_conf=0.5, psms=(11,))
    assert coarse == coarse_expected == tokens
    assert coarse_diag["tokens_per_engine"] == {"tesseract": 3, "paddle": 1}
    merged, merge_diag = merge_passes(orchestrated_tokens, coarse)
    assert merged == tokens
    assert merge_diag == {"fine_only": 2, "conflicts": 0,
                          "recovered_from_coarse": 0, "merged_total": 2}
