"""Unit tests for Unimelb_service.py (UMA web backend).
Run individually with: python -m pytest tests/test_web_Unimelb_service.py -v
"""
from __future__ import annotations
import pytest
from backend_test_helpers import module

def test_service_base_abstract_methods():
    base=module('services.Unimelb_service')
    with pytest.raises(TypeError): base.UnimelbService()
    class Impl(base.UnimelbService):
        async def get_image_select(self): return await super().get_image_select()
        def resolve_image(self,*a): return super().resolve_image(*a)
        async def create_ocr_task(self,force): return await super().create_ocr_task(force)
        async def rerun_ocr_images(self,keys): return await super().rerun_ocr_images(keys)
        async def update_ocr_image_result(self,key,result): return await super().update_ocr_image_result(key,result)
        async def export_ocr_results(self,keys,status=None,folder=None): return await super().export_ocr_results(keys,status,folder)
    item=Impl()
    assert isinstance(base.UnimelbTaskRunningError('busy'),RuntimeError)
    with pytest.raises(NotImplementedError): item.resolve_image('F','N','a')


@pytest.mark.asyncio
async def test_service_base_default_lifecycle():
    base=module('services.Unimelb_service')
    # Give the abstract methods concrete implementations, then exercise harmless defaults.
    class Concrete(base.UnimelbService):
        async def get_image_select(self): return None
        def resolve_image(self,*args): return None
        async def create_ocr_task(self,force): return None
        async def rerun_ocr_images(self,keys): return None
        async def update_ocr_image_result(self,key,result): return None
        async def export_ocr_results(self,keys,status=None,folder=None): return b''
    instance=Concrete()
    assert await instance.start() is None
    assert await instance.close() is None


@pytest.mark.asyncio
async def test_abstract_service_super_methods_raise():
    base=module('services.Unimelb_service')
    class Concrete(base.UnimelbService):
        async def get_image_select(self): return await super().get_image_select()
        def resolve_image(self,*args): return super().resolve_image(*args)
        async def create_ocr_task(self,force): return await super().create_ocr_task(force)
        async def rerun_ocr_images(self,keys): return await super().rerun_ocr_images(keys)
        async def update_ocr_image_result(self,key,result): return await super().update_ocr_image_result(key,result)
        async def export_ocr_results(self,keys,status=None,folder=None): return await super().export_ocr_results(keys,status,folder)
    item=Concrete()
    with pytest.raises(NotImplementedError):await item.get_image_select()
    with pytest.raises(NotImplementedError):item.resolve_image('a','b','c')
    with pytest.raises(NotImplementedError):await item.create_ocr_task(False)
    with pytest.raises(NotImplementedError):await item.rerun_ocr_images([])
    with pytest.raises(NotImplementedError):await item.update_ocr_image_result('a',{})
    with pytest.raises(NotImplementedError):await item.export_ocr_results(None)

