"""Unit tests for Unimelb_response.py (UMA web backend).
Run individually with: python -m pytest tests/test_web_Unimelb_response.py -v
"""
from __future__ import annotations
import pytest
from pydantic import ValidationError
from backend_test_helpers import module

@pytest.fixture(scope='module')
def response(): return module('schemas.Unimelb_response')


@pytest.mark.parametrize('confidence',[0,0.5,1,None])
def test_ocr_confidence_valid(response,confidence):
    assert response.UnimelbOCRFieldResult(confidence=confidence).confidence == confidence


@pytest.mark.parametrize('confidence',[-0.01,1.01,10])
def test_ocr_confidence_invalid(response,confidence):
    with pytest.raises(ValidationError): response.UnimelbOCRFieldResult(confidence=confidence)


def test_ocr_field_default_lists_independent(response):
    a = response.UnimelbOCRFieldResult()
    b = response.UnimelbOCRFieldResult()
    a.tokens.append(1)
    a.notes.append('review')
    assert b.tokens == [] and b.notes == []


def test_image_select_item(response):
    item = response.UnimelbImageSelectItem(path='F/N/a.png',image_key='N/a.png',folder='F',node_id='N',image_name='a.png',image_url='',status=2,result={'date':{'verbatim':'2026','confidence':0.8}})
    assert item.result['date'].verbatim == '2026'
    assert item.result['date'].confidence == 0.8
    assert item.manual_reviewed == 0


@pytest.mark.parametrize('status',[-1,4])
def test_image_status_rejects_invalid(response,status):
    with pytest.raises(ValidationError): response.UnimelbImageSelectItem(path='p',image_key='k',folder='f',node_id='n',image_name='i',image_url='',status=status)


@pytest.mark.parametrize('reviewed',[-1,2])
def test_image_reviewed_flag_rejects_invalid(response,reviewed):
    with pytest.raises(ValidationError): response.UnimelbImageSelectItem(path='p',image_key='k',folder='f',node_id='n',image_name='i',image_url='',status=0,manual_reviewed=reviewed)


def test_image_select_response(response):
    r=response.UnimelbImageSelectResponse(task_uuid='task',folders={'F':2},total=2,status_counts={'0':1,'2':1},images=[])
    assert r.folders['F'] == 2 and r.total == 2 and r.images == []


@pytest.mark.parametrize('status',[0,1,2,3])
def test_task_created_and_rerun_response(response,status):
    c=response.UnimelbTaskCreatedResponse(task_uuid='x',status=status,total=2,started=True)
    r=response.UnimelbTaskRerunResponse(task_uuid='x',status=status,selected_total=1,image_keys=['N/a'],started=False)
    assert c.status == r.status == status
    assert r.image_keys == ['N/a']


@pytest.mark.parametrize('status',[-1,4])
def test_task_responses_reject_status(response,status):
    with pytest.raises(ValidationError): response.UnimelbTaskCreatedResponse(task_uuid='x',status=status,total=1,started=True)
    with pytest.raises(ValidationError): response.UnimelbTaskRerunResponse(task_uuid='x',status=status,selected_total=1,image_keys=[],started=True)


@pytest.mark.parametrize('reviewed',[0,1])
def test_update_response_review_flag(response,reviewed):
    r=response.UnimelbImageResultUpdateResponse(image_key='N/a',manual_reviewed=reviewed,updated_at=100,result={'scale':'1:100'})
    assert r.manual_reviewed == reviewed and r.result['scale'] == '1:100'


@pytest.mark.parametrize('reviewed',[-1,2])
def test_update_response_review_flag_invalid(response,reviewed):
    with pytest.raises(ValidationError): response.UnimelbImageResultUpdateResponse(image_key='N/a',manual_reviewed=reviewed,updated_at=100,result={})

