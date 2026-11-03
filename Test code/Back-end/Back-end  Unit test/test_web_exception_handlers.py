"""Unit tests for exception_handlers.py (UMA web backend).
Run individually with: python -m pytest tests/test_web_exception_handlers.py -v
"""
from __future__ import annotations
import logging
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from backend_test_helpers import module

@pytest.fixture
def exceptions(): return module('exception_handlers')


def test_json_helper(exceptions):
    res=exceptions._json(module('schemas.result').Result.error('no',401),401,headers={'X-Debug':'a'})
    assert res.status_code==401 and res.headers['x-debug']=='a'
    assert b'"success":false' in res.body


def test_http_error_handler(exceptions):
    app=FastAPI();exceptions.register_exception_handlers(app)
    @app.get('/fail')
    def fail(): raise HTTPException(status_code=418,detail='teapot',headers={'X-Extra':'yes'})
    with TestClient(app) as client:
        r=client.get('/fail')
    assert r.status_code==418 and r.json()['code']==418 and r.json()['message']=='teapot'
    assert r.headers['x-extra']=='yes'


def test_validation_error_handler(exceptions):
    app=FastAPI();exceptions.register_exception_handlers(app)
    @app.get('/number')
    def number(number:int): return {'number':number}
    with TestClient(app) as client:
        r=client.get('/number?number=abc')
    assert r.status_code==422 and r.json()['code']==422
    assert r.json()['message']=='Request validation failed'
    assert len(r.json()['result'])>=1


def test_unexpected_error_handler(exceptions,caplog):
    app=FastAPI();exceptions.register_exception_handlers(app)
    @app.get('/explode')
    def explode(): raise RuntimeError('do not leak this message')
    with caplog.at_level(logging.ERROR),TestClient(app,raise_server_exceptions=False) as client:
        r=client.get('/explode')
    assert r.status_code==500 and r.json()['message']=='Internal server error'
    assert 'do not leak' not in r.text
    assert any('Unhandled API exception' in record.message for record in caplog.records)

