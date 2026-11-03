"""Unit tests for Unimelb_api.py (UMA web backend).
Run individually with: python -m pytest tests/test_web_Unimelb_api.py -v
"""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
import pytest
from fastapi import HTTPException
from backend_test_helpers import module, fixture_values

@pytest.fixture
def deps():
    api_mod=module('api.Unimelb_api')
    schemas=module('schemas')
    service=SimpleNamespace(
        get_image_select=AsyncMock(), resolve_image=Mock(),
        create_ocr_task=AsyncMock(),rerun_ocr_images=AsyncMock(),
        update_ocr_image_result=AsyncMock(),export_ocr_results=AsyncMock(),
    )
    controller=api_mod.UnimelbAPI(service)
    return SimpleNamespace(api_mod=api_mod,schemas=schemas,service=service,controller=controller)


def test_api_registers_expected_routes(deps):
    methods={(route.path,method) for route in deps.controller.router.routes for method in route.methods}
    assert {('/Unimelb/ImageSelect','GET'),('/Unimelb/images/{folder}/{node_id}/{image_name}','GET'),('/Unimelb/ocr-tasks','POST'),('/Unimelb/ocr-tasks/rerun','POST'),('/Unimelb/ocr-results/manual-review','POST'),('/Unimelb/ocr-results/export','POST')} <= methods


@pytest.mark.asyncio
async def test_image_select_generates_urls(deps):
    image=deps.schemas.UnimelbImageSelectItem(path='F/N/x.png',image_key='N/x.png',folder='F',node_id='N',image_name='x.png',image_url='',status=0)
    deps.service.get_image_select.return_value=deps.schemas.UnimelbImageSelectResponse(task_uuid='id',folders={'F':1},total=1,status_counts={'0':1},images=[image])
    req=SimpleNamespace(url_for=lambda name,**kw:'http://localhost/'+kw['folder']+'/'+kw['node_id']+'/'+kw['image_name'])
    result=await deps.controller.image_select(req)
    assert result.success and result.result.images[0].image_url=='http://localhost/F/N/x.png'
    deps.service.get_image_select.assert_awaited_once()


@pytest.mark.asyncio
async def test_image_select_missing_file_404(deps):
    deps.service.get_image_select.side_effect=FileNotFoundError('missing images')
    with pytest.raises(HTTPException) as exc:
        await deps.controller.image_select(SimpleNamespace())
    assert exc.value.status_code==404


@pytest.mark.asyncio
async def test_get_image_file_response(deps,tmp_path):
    file=tmp_path/'test.png';file.write_bytes(b'abc')
    deps.service.resolve_image.return_value=file
    result=await deps.controller.get_image('F','N','test.png')
    assert Path(result.path)==file
    assert result.headers['cache-control']=='public, max-age=3600'
    deps.service.resolve_image.assert_called_once_with('F','N','test.png')


@pytest.mark.asyncio
async def test_get_image_missing_404(deps):
    deps.service.resolve_image.return_value=None
    with pytest.raises(HTTPException) as exc: await deps.controller.get_image('F','N','bad.png')
    assert exc.value.status_code==404


@pytest.mark.parametrize('started',[True,False])
@pytest.mark.asyncio
async def test_create_task_success_messages(deps,started):
    deps.service.create_ocr_task.return_value=deps.schemas.UnimelbTaskCreatedResponse(task_uuid='id',status=1,total=1,started=started)
    result=await deps.controller.create_ocr_task(deps.schemas.UnimelbTaskCreateRequest(force=True))
    assert result.success and result.result.started is started
    assert result.message==('OCR task started.' if started else 'The shared task already exists.')
    deps.service.create_ocr_task.assert_awaited_once_with(True)


@pytest.mark.asyncio
async def test_create_task_missing_404(deps):
    deps.service.create_ocr_task.side_effect=FileNotFoundError('missing')
    with pytest.raises(HTTPException) as exc: await deps.controller.create_ocr_task(deps.schemas.UnimelbTaskCreateRequest())
    assert exc.value.status_code==404


@pytest.mark.asyncio
async def test_create_task_running_returns_409_result(deps):
    deps.service.create_ocr_task.side_effect=deps.api_mod.UnimelbTaskRunningError('busy')
    result=await deps.controller.create_ocr_task(deps.schemas.UnimelbTaskCreateRequest())
    assert not result.success and result.code==409 and 'busy' in result.message


@pytest.mark.parametrize('elapsed,expected_retry',[(0.0,60),(10.1,50),(59.9,1)])
def test_create_task_rate_limit(deps,monkeypatch,elapsed,expected_retry):
    ctrl=deps.controller
    monkeypatch.setattr(ctrl,'_create_task_call_time',lambda:100.0)
    ctrl._register_create_task_submission_locked()
    assert ctrl._last_create_task_call_time==100.0
    monkeypatch.setattr(ctrl,'_create_task_call_time',lambda:100.0+elapsed)
    with pytest.raises(HTTPException) as err: ctrl._register_create_task_submission_locked()
    assert err.value.status_code==429
    assert err.value.headers['Retry-After']==str(expected_retry)
    assert ctrl._last_create_task_call_time==100.0


def test_create_task_after_cooldown(deps,monkeypatch):
    times=iter([10.,70.,150.]);monkeypatch.setattr(deps.controller,'_create_task_call_time',lambda:next(times))
    for _ in range(3): deps.controller._register_create_task_submission_locked()
    assert deps.controller._last_create_task_call_time==150.


def test_create_task_clock_monotonic(deps,monkeypatch):
    monkeypatch.setattr(deps.api_mod.time,'monotonic',lambda:12.75)
    assert deps.controller._create_task_call_time()==12.75


@pytest.mark.asyncio
async def test_rerun_success(deps):
    deps.service.rerun_ocr_images.return_value=deps.schemas.UnimelbTaskRerunResponse(task_uuid='id',status=1,selected_total=1,image_keys=['N/a.png'],started=True)
    r=await deps.controller.rerun_ocr_images(deps.schemas.UnimelbTaskRerunRequest(image_keys=['N/a.png']))
    assert r.success and r.result.selected_total==1
    deps.service.rerun_ocr_images.assert_awaited_once_with(['N/a.png'])


@pytest.mark.parametrize('error,expected',[(FileNotFoundError,404),('running',409)])
@pytest.mark.asyncio
async def test_rerun_errors(deps,error,expected):
    cls=deps.api_mod.UnimelbTaskRunningError if error=='running' else error
    deps.service.rerun_ocr_images.side_effect=cls('error')
    with pytest.raises(HTTPException) as e: await deps.controller.rerun_ocr_images(deps.schemas.UnimelbTaskRerunRequest(image_keys=[]))
    assert e.value.status_code==expected


@pytest.mark.asyncio
async def test_manual_update_success(deps):
    deps.service.update_ocr_image_result.return_value=deps.schemas.UnimelbImageResultUpdateResponse(image_key='N/a',manual_reviewed=1,updated_at=1,result={'date':'2026-01-01'})
    data=deps.schemas.UnimelbImageResultUpdateRequest(image_key='N/a',result=fixture_values())
    r=await deps.controller.update_ocr_image_result(data)
    assert r.success and r.result.manual_reviewed==1
    deps.service.update_ocr_image_result.assert_awaited_once_with('N/a',fixture_values())


@pytest.mark.parametrize('error,expected',[(FileNotFoundError,404),('running',409)])
@pytest.mark.asyncio
async def test_manual_update_errors(deps,error,expected):
    cls=deps.api_mod.UnimelbTaskRunningError if error=='running' else error
    deps.service.update_ocr_image_result.side_effect=cls('problem')
    with pytest.raises(HTTPException) as exc: await deps.controller.update_ocr_image_result(deps.schemas.UnimelbImageResultUpdateRequest(image_key='N/a',result=fixture_values()))
    assert exc.value.status_code==expected


@pytest.mark.asyncio
async def test_export_content_and_headers(deps):
    deps.service.export_ocr_results.return_value=b'xlsx bytes'
    r=await deps.controller.export_ocr_results(deps.schemas.UnimelbResultExportRequest(image_keys=['N/a'],status=2,folder='ABC'))
    assert r.body==b'xlsx bytes'
    assert r.media_type=='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    assert 'unimelb_image_status.xlsx' in r.headers['content-disposition']
    assert r.headers['cache-control']=='no-store'
    deps.service.export_ocr_results.assert_awaited_once_with(['N/a'],status=2,folder='ABC')


@pytest.mark.parametrize('error,expected',[(FileNotFoundError,404),('running',409)])
@pytest.mark.asyncio
async def test_export_errors(deps,error,expected):
    cls=deps.api_mod.UnimelbTaskRunningError if error=='running' else error
    deps.service.export_ocr_results.side_effect=cls('error')
    with pytest.raises(HTTPException) as exc: await deps.controller.export_ocr_results(deps.schemas.UnimelbResultExportRequest())
    assert exc.value.status_code==expected

