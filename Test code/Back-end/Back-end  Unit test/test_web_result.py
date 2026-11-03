"""Unit tests for result.py (UMA web backend).
Run individually with: python -m pytest tests/test_web_result.py -v
"""
from __future__ import annotations
import time
import pytest
from backend_test_helpers import module

@pytest.fixture(scope='module')
def results(): return module('schemas.result')


def test_result_default_and_timestamp(results):
    before = int(time.time()*1000)
    r = results.Result()
    after = int(time.time()*1000)
    assert r.success and r.code == 200 and r.message == '' and r.result is None
    assert before <= r.timestamp <= after


def test_result_ok(results):
    r = results.Result[dict].ok({'x':1})
    assert (r.success,r.code,r.message,r.result) == (True,200,'Operation successful.',{'x':1})
    assert results.Result[str].ok('value',message='done').message == 'done'


def test_result_error_and_customization(results):
    r = results.Result[str].error('bad input', 422, result='details')
    assert (r.success,r.code,r.message,r.result) == (False,422,'bad input','details')
    assert results.Result.error().code == 500


def test_result_noauth(results):
    r = results.Result.noauth()
    assert not r.success and r.code == 403 and r.message == 'Access denied'
    assert results.Result.noauth('Not allowed').message == 'Not allowed'

