import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from uma.llm import (
    LlmClient,
    _evidence_text,
    assign_fields,
    build_prompt,
    classify_correction,
    parse_json,
    record_evidence_regions,
    resolve_api_key,
    resolve_evidence_regions,
    verify_evidence,
)
from uma.config import LlmCfg
from uma.models import FieldValue, Token


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def make_token(
    tid,
    text,
    bbox=(10, 10, 50, 30),
    conf=0.9,
    agreement=0.9,
):
    token = Token(
        text=text,
        bbox=bbox,
        conf=conf,
    )

    token.tid = tid
    token.agreement = agreement

    return token


# ---------------------------------------------------------
# resolve_api_key
# ---------------------------------------------------------

def test_resolve_api_key():
    cfg = LlmCfg()
    cfg.api_key_env = "TEST_KEY"

    assert resolve_api_key(cfg) == "TEST_KEY"


# ---------------------------------------------------------
# classify_correction
# ---------------------------------------------------------

def test_classify_correction():
    # No corrected value
    assert classify_correction("ABC", None, None) is None

    # Same after normalization
    assert classify_correction(
        "Site Plan",
        "SITE PLAN",
        None,
    ) is None

    # Changed without evidence basis
    assert classify_correction(
        "S1TE",
        "SITE",
        None,
    ) == "normalize"

    # Changed with correction evidence
    assert classify_correction(
        "S1TE",
        "SITE",
        "OCR confused 1 and I",
    ) == "fix"


# ---------------------------------------------------------
# build_prompt
# ---------------------------------------------------------

def test_build_prompt():
    locked_value = FieldValue(
        name="architect",
        tokens=[],
        verbatim="BATES SMART",
    )

    prompt = build_prompt(
        grid="GRID CONTENT",
        index="[1] SITE PLAN",
        pending=["drawing_title", "drawing_number"],
        locked={"architect": locked_value},
        rejected={"date": "low confidence"},
    )

    assert "GRID CONTENT" in prompt
    assert "[1] SITE PLAN" in prompt
    assert "drawing_title" in prompt
    assert "drawing_number" in prompt
    assert "BATES SMART" in prompt
    assert "low confidence" in prompt


# ---------------------------------------------------------
# parse_json
# ---------------------------------------------------------

def test_parse_json():
    response = """
    ```json
    {
        "drawing_title": "SITE PLAN",
        "drawing_number": "603/A1"
    }
    ```
    """

    result = parse_json(response)

    assert result["drawing_title"] == "SITE PLAN"
    assert result["drawing_number"] == "603/A1"


def test_parse_json_with_extra_text():
    response = """
    Model result:
    {"drawing_title": "SITE PLAN"}
    End.
    """

    result = parse_json(response)

    assert result == {
        "drawing_title": "SITE PLAN"
    }


def test_parse_json_invalid():
    with pytest.raises(json.JSONDecodeError):
        parse_json("this is not json")


# ---------------------------------------------------------
# _evidence_text
# ---------------------------------------------------------

def test_evidence_text():
    assert _evidence_text("Site Plan") == "SITEPLAN"
    assert _evidence_text(" 603 / A1 ") == "603/A1"
    assert _evidence_text(None) == ""


# ---------------------------------------------------------
# resolve_evidence_regions
# ---------------------------------------------------------

def test_resolve_evidence_regions_success():
    token1 = make_token(
        1,
        "SITE",
        (10, 10, 50, 30),
    )

    token2 = make_token(
        2,
        "PLAN",
        (60, 10, 100, 30),
    )

    fv = FieldValue(
        name="drawing_title",
        tokens=[1, 2],
        verbatim="SITE PLAN",
    )

    result = resolve_evidence_regions(
        fv,
        {
            1: token1,
            2: token2,
        },
    )

    assert result is not None
    assert len(result) == 2

    assert result[0]["tid"] == 1
    assert result[1]["tid"] == 2

    assert result[0]["text"] == "SITE"
    assert result[1]["text"] == "PLAN"


def test_resolve_evidence_regions_missing_token():
    fv = FieldValue(
        name="drawing_title",
        tokens=[999],
        verbatim="SITE PLAN",
    )

    result = resolve_evidence_regions(
        fv,
        {},
    )

    assert result is None


def test_resolve_evidence_regions_no_tokens():
    fv = FieldValue(
        name="drawing_title",
        tokens=[],
        verbatim=None,
    )

    assert resolve_evidence_regions(fv, {}) is None


# ---------------------------------------------------------
# record_evidence_regions
# ---------------------------------------------------------

def test_record_evidence_regions():
    token1 = make_token(
        1,
        "SITE",
        (10, 20, 50, 40),
    )

    token2 = make_token(
        2,
        "PLAN",
        (60, 20, 110, 40),
    )

    fv = FieldValue(
        name="drawing_title",
        tokens=[1, 2],
        verbatim="SITE PLAN",
    )

    result = record_evidence_regions(
        fv,
        {
            1: token1,
            2: token2,
        },
    )

    assert result is True

    assert "evidence_regions" in fv.signals
    assert "evidence_bbox" in fv.signals

    assert fv.signals["evidence_bbox"] == {
        "x0": 10,
        "y0": 20,
        "x1": 110,
        "y1": 40,
    }


# ---------------------------------------------------------
# verify_evidence
# ---------------------------------------------------------

def test_verify_evidence_no_tokens():
    fv = FieldValue(
        name="drawing_title",
        tokens=[],
        verbatim=None,
    )

    ok, message, extra = verify_evidence(
        fv,
        {},
    )

    assert ok is True
    assert message == ""
    assert extra == []


def test_verify_evidence_missing_token():
    fv = FieldValue(
        name="drawing_title",
        tokens=[100],
        verbatim="SITE",
    )

    ok, message, extra = verify_evidence(
        fv,
        {},
    )

    assert ok is False
    assert "nonexistent token" in message
    assert extra == []


def test_verify_evidence_success():
    token1 = make_token(
        1,
        "SITE",
        (10, 10, 50, 30),
    )

    token2 = make_token(
        2,
        "PLAN",
        (60, 10, 100, 30),
    )

    fv = FieldValue(
        name="drawing_title",
        tokens=[1, 2],
        verbatim="SITE PLAN",
    )

    ok, message, extra = verify_evidence(
        fv,
        {
            1: token1,
            2: token2,
        },
    )

    assert ok is True
    assert message == ""
    assert extra == []


def test_verify_evidence_fabricated_text():
    token = make_token(
        1,
        "SITE",
    )

    fv = FieldValue(
        name="drawing_title",
        tokens=[1],
        verbatim="COMPLETELY DIFFERENT",
    )

    ok, message, extra = verify_evidence(
        fv,
        {1: token},
    )

    assert ok is False
    assert "failed" in message.lower()
    assert extra == []


# ---------------------------------------------------------
# LlmClient.available
# ---------------------------------------------------------

def test_llm_client_disabled():
    cfg = LlmCfg()
    cfg.enabled = False

    client = LlmClient(cfg)

    assert client.available() is False


# ---------------------------------------------------------
# LlmClient.complete without real API
# ---------------------------------------------------------

def test_llm_complete_without_retry():
    cfg = LlmCfg()
    cfg.max_retries = 0

    client = LlmClient(cfg)

    client._once = Mock(
        return_value=(
            '{"drawing_title":"SITE PLAN"}',
            {"total_tokens": 10},
        )
    )

    text, usage = client.complete(
        "system",
        "user",
    )

    assert text == '{"drawing_title":"SITE PLAN"}'
    assert usage["total_tokens"] == 10

    client._once.assert_called_once_with(
        "system",
        "user",
    )


# ---------------------------------------------------------
# _once using fake API response
# ---------------------------------------------------------

def test_llm_once():
    cfg = LlmCfg()

    cfg.model = "test-model"
    cfg.temperature = 0
    cfg.max_tokens = 100

    client = LlmClient(cfg)

    message = SimpleNamespace(
        content='{"drawing_title":"SITE PLAN"}',
        reasoning_content="",
    )

    choice = SimpleNamespace(
        message=message,
        finish_reason="stop",
    )

    usage = SimpleNamespace(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        completion_tokens_details=None,
    )

    response = SimpleNamespace(
        choices=[choice],
        usage=usage,
        system_fingerprint="test-fingerprint",
    )

    fake_create = Mock(
        return_value=response
    )

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=fake_create
            )
        )
    )

    client._client = fake_client

    text, info = client._once(
        "SYSTEM",
        "USER",
    )

    assert text == '{"drawing_title":"SITE PLAN"}'

    assert info["prompt_tokens"] == 10
    assert info["completion_tokens"] == 5
    assert info["total_tokens"] == 15

    assert info["finish_reason"] == "stop"
    assert info["system_fingerprint"] == "test-fingerprint"

    fake_create.assert_called_once()


# ---------------------------------------------------------
# assign_fields - skipped
# ---------------------------------------------------------

def test_assign_fields_no_pending():
    cfg = LlmCfg()

    client = LlmClient(cfg)

    result, diag = assign_fields(
        grid="",
        index="",
        pending=[],
        locked={},
        flat=[],
        client=client,
    )

    assert result == {}
    assert diag["called"] is False
    assert diag["skipped"] == "no_pending"


# ---------------------------------------------------------
# assign_fields - successful fake LLM call
# ---------------------------------------------------------

def test_assign_fields_success():
    token = make_token(
        1,
        "SITE PLAN",
        (10, 10, 100, 30),
        conf=0.95,
        agreement=0.90,
    )

    client = Mock()

    client.available.return_value = True

    client.complete.return_value = (
        """
        {
            "drawing_title": {
                "tokens": [1],
                "verbatim": "SITE PLAN",
                "corrected": "Site Plan",
                "confidence": "high"
            }
        }
        """,
        {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "elapsed_sec": 0.1,
        },
    )

    result, diag = assign_fields(
        grid="SITE PLAN",
        index="[1] SITE PLAN",
        pending=["drawing_title"],
        locked={},
        flat=[token],
        client=client,
        verbose=False,
    )

    assert diag["called"] is True

    assert "drawing_title" in result

    fv = result["drawing_title"]

    assert fv.verbatim == "SITE PLAN"
    assert fv.tokens == [1]

    assert fv.signals["evidence_ok"] is True
    assert fv.signals["llm_confidence"] == "high"

    client.complete.assert_called_once()


# ---------------------------------------------------------
# assign_fields - API failure
# ---------------------------------------------------------

def test_assign_fields_api_error():
    client = Mock()

    client.available.return_value = True
    client.complete.side_effect = RuntimeError(
        "API unavailable"
    )

    result, diag = assign_fields(
        grid="GRID",
        index="INDEX",
        pending=["drawing_title"],
        locked={},
        flat=[],
        client=client,
        verbose=False,
    )

    assert result == {}
    assert "error" in diag
    assert "API unavailable" in diag["error"]


# ---------------------------------------------------------
# assign_fields - invalid JSON
# ---------------------------------------------------------

def test_assign_fields_invalid_json():
    client = Mock()

    client.available.return_value = True

    client.complete.return_value = (
        "INVALID JSON",
        {
            "elapsed_sec": 0.1,
        },
    )

    result, diag = assign_fields(
        grid="GRID",
        index="INDEX",
        pending=["drawing_title"],
        locked={},
        flat=[],
        client=client,
        verbose=False,
    )

    assert result == {}
    assert "parse_error" in diag