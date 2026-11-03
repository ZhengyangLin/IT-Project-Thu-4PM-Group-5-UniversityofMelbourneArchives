"""Unit tests for base_api.py (UMA web backend).
Run individually with: python -m pytest tests/test_web_base_api.py -v
"""
from __future__ import annotations
import pytest
from backend_test_helpers import module

def test_base_api_is_abstract_and_subclass_registers_routes():
    base=module('api.base_api').BaseAPI
    with pytest.raises(TypeError): base()
    class Concrete(base):
        prefix='/test'
        tags=['test']
        def register_routes(self):
            self.router.add_api_route('/ping',lambda:{'ok':True},methods=['GET'])
    instance=Concrete()
    assert instance.router.prefix=='/test'
    assert len(instance.router.routes)==1
    with pytest.raises(NotImplementedError):
        super(Concrete,instance).register_routes()

