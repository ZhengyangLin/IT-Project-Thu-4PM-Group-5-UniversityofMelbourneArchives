"""Unit tests for Unimelb_ocr_worker.py (UMA web backend).
Run individually with: python -m pytest tests/test_web_Unimelb_ocr_worker.py -v
"""
from __future__ import annotations
import importlib
import logging
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from backend_test_helpers import module

@pytest.fixture
def worker(): return module('workers.Unimelb_ocr_worker')


@pytest.fixture
def job(worker): return worker.OcrImageJob(image_path='/tmp/a.png',node_id='N',item_output_dir='/tmp/out')


def test_worker_dataclasses(worker,job):
    assert job.node_id=='N' and job.image_path.endswith('a.png')
    a=worker.OcrImageOutcome(ok=True); b=worker.OcrImageOutcome(ok=True)
    a.result['x']=1
    assert b.result=={} and b.error is None and b.summary_row is None


def test_worker_not_initialised(worker,job,monkeypatch):
    monkeypatch.setattr(worker,'_PIPELINE',None)
    r=worker.process_ocr_image(job)
    assert not r.ok and 'not initialized' in r.error


@pytest.mark.parametrize('failure',[None,'bad file'])
def test_worker_process_success_or_diagnostic_failure(worker,job,monkeypatch,failure):
    pipeline=importlib.import_module('uma.pipeline')
    record=SimpleNamespace(diagnostics={'processing_error':failure,'fatal':None})
    mock_pipeline=SimpleNamespace(process=Mock(return_value=record))
    monkeypatch.setattr(worker,'_PIPELINE',mock_pipeline)
    write=Mock();monkeypatch.setattr(pipeline,'write_outputs',write)
    monkeypatch.setattr(pipeline,'record_to_row',lambda rec:{'n':1})
    monkeypatch.setattr(worker,'extract_ocr_result',lambda rec:{'date':'2026'})
    result=worker.process_ocr_image(job)
    assert result.ok is (failure is None)
    assert result.error==failure
    assert result.result=={'date':'2026'}
    assert result.summary_row=={'n':1}
    mock_pipeline.process.assert_called_once_with(job.image_path,job.node_id,out_dir=job.item_output_dir)
    write.assert_called_once_with([record],job.item_output_dir)


def test_worker_process_exception(worker,job,monkeypatch,caplog):
    monkeypatch.setattr(worker,'_PIPELINE',SimpleNamespace(process=Mock(side_effect=ValueError('bad image'))))
    with caplog.at_level(logging.ERROR): r=worker.process_ocr_image(job)
    assert not r.ok and r.error=='ValueError: bad image'
    assert 'OCR worker failed' in caplog.text


def test_worker_initialize_without_logging(worker,monkeypatch):
    config=importlib.import_module('uma.config')
    pipeline=importlib.import_module('uma.pipeline')
    cfg=SimpleNamespace(ocr=SimpleNamespace(engines=[]))
    monkeypatch.setattr(config.Config,'load',staticmethod(lambda path:cfg))
    captured={}
    def fake_pipeline(config,verbose_prompt):
        captured.update(config=config,verbose_prompt=verbose_prompt)
        return 'fake pipeline'
    monkeypatch.setattr(pipeline,'Pipeline',fake_pipeline)
    worker.initialize_ocr_worker('/test/config.yml',('paddle','other'),False)
    assert cfg.ocr.engines==['paddle','other']
    assert captured['verbose_prompt'] is False
    assert worker._PIPELINE=='fake pipeline'


def test_worker_initialize_with_logging(worker,monkeypatch,tmp_path):
    config=importlib.import_module('uma.config')
    pipeline=importlib.import_module('uma.pipeline')
    logs=importlib.import_module('uma.logging_utils')
    config_obj=SimpleNamespace(ocr=SimpleNamespace(engines=[]))
    monkeypatch.setattr(config.Config,'load',staticmethod(lambda path:config_obj))
    monkeypatch.setattr(pipeline,'Pipeline',lambda *args,**kwargs:'pipeline')
    mocked=Mock();monkeypatch.setattr(logs,'setup_logging',mocked)
    worker.initialize_ocr_worker('config',('paddle',),True,str(tmp_path))
    mocked.assert_called_once_with(str(tmp_path))
    assert worker._PIPELINE=='pipeline'


def test_worker_write_batch_summary(worker,monkeypatch):
    pipeline=importlib.import_module('uma.pipeline')
    called=Mock(return_value='summary.xlsx')
    monkeypatch.setattr(pipeline,'write_summary_rows',called)
    assert worker.write_batch_summary('/tmp/out',[{'image':'x'}])=='summary.xlsx'
    called.assert_called_once_with([{'image':'x'}],'/tmp/out')

