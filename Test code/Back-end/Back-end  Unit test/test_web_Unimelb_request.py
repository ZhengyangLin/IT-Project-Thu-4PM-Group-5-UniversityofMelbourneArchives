"""Unit tests for Unimelb_request.py (UMA web backend).
Run individually with: python -m pytest tests/test_web_Unimelb_request.py -v
"""
from __future__ import annotations
import pytest
from pydantic import ValidationError
from backend_test_helpers import module, fixture_values

@pytest.fixture(scope='module')
def requests_module(): return module('schemas.Unimelb_request')


@pytest.mark.parametrize('force',[False,True])
def test_create_request_force(requests_module,force):
    assert requests_module.UnimelbTaskCreateRequest(force=force).force is force


def test_create_request_default(requests_module):
    assert requests_module.UnimelbTaskCreateRequest().force is False


def test_rerun_request_list_and_required(requests_module):
    assert requests_module.UnimelbTaskRerunRequest(image_keys=['n/a.png','n/b.png']).image_keys == ['n/a.png','n/b.png']
    with pytest.raises(ValidationError): requests_module.UnimelbTaskRerunRequest()
    with pytest.raises(ValidationError): requests_module.UnimelbTaskRerunRequest(image_keys=123)


def test_result_values_all_fields_required_even_if_nullable(requests_module):
    values = requests_module.UnimelbOCRResultValues(**dict.fromkeys(fixture_values(),None))
    assert all(value is None for value in values.model_dump().values())
    with pytest.raises(ValidationError): requests_module.UnimelbOCRResultValues(drawing_number='A-001')
    with pytest.raises(ValidationError): requests_module.UnimelbOCRResultValues(**fixture_values(), extra='denied')


def test_update_request_nested_model_and_extras(requests_module):
    payload = requests_module.UnimelbImageResultUpdateRequest(image_key='N1/img.png',result=fixture_values())
    assert payload.result.drawing_number == 'A-001'
    with pytest.raises(ValidationError): requests_module.UnimelbImageResultUpdateRequest(image_key='x',result=fixture_values(),unused='no')
    with pytest.raises(ValidationError): requests_module.UnimelbImageResultUpdateRequest(image_key='x',result={'date': 'today'})


@pytest.mark.parametrize('status',[0,1,2,3,None])
def test_export_status_valid(requests_module,status):
    assert requests_module.UnimelbResultExportRequest(status=status).status == status


@pytest.mark.parametrize('status',[-1,4,100,'not-a-status'])
def test_export_status_invalid(requests_module,status):
    with pytest.raises(ValidationError): requests_module.UnimelbResultExportRequest(status=status)


@pytest.mark.parametrize('field',["status","folder"])
@pytest.mark.parametrize('value',['','   ','\t\n'])
def test_export_normalizes_blank_filters(requests_module,field,value):
    assert getattr(requests_module.UnimelbResultExportRequest(**{field:value}),field) is None


def test_export_retains_nonblank_filters(requests_module):
    result = requests_module.UnimelbResultExportRequest(image_keys=['N1/a.png'],folder='FOLDER',status='2')
    assert result.image_keys == ['N1/a.png']
    assert result.folder == 'FOLDER'
    assert result.status == 2


def test_export_defaults(requests_module):
    assert requests_module.UnimelbResultExportRequest().model_dump() == dict(image_keys=None,status=None,folder=None)

