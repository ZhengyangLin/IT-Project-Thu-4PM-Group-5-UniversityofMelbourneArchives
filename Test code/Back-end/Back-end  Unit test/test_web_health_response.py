"""Unit tests for health_response.py (UMA web backend).
Run individually with: python -m pytest tests/test_web_health_response.py -v
"""
from __future__ import annotations
import pytest
from pydantic import ValidationError
from backend_test_helpers import module

@pytest.fixture(scope='module')
def health(): return module('schemas.health_response')


def test_health_response_required_fields(health):
    value = health.HealthResponse(status='ok',service='uma-ocr',version='0.1.0')
    assert value.model_dump() == dict(status='ok',service='uma-ocr',version='0.1.0')
    with pytest.raises(ValidationError): health.HealthResponse(status='ok')

