"""Resolve the project's real backend package instead of assuming 'uma.backend'.

Place this file and the tests in PROJECT_ROOT/tests. An optional
UMA_BACKEND_PACKAGE env var overrides autodiscovery (example: uma.server).
"""
from __future__ import annotations
import importlib
import os
from pathlib import Path
from functools import lru_cache

PROJECT_ROOT = Path(__file__).resolve().parents[1]

@lru_cache(maxsize=1)
def package_name() -> str:
    override = os.getenv('UMA_BACKEND_PACKAGE')
    if override:
        return override
    for path in PROJECT_ROOT.rglob('app.py'):
        if any(p.startswith('.') or p in {'venv', 'node_modules', '__pycache__'} for p in path.relative_to(PROJECT_ROOT).parts[:-1]):
            continue
        if path.is_file() and 'UMA OCR API' in path.read_text(encoding='utf-8', errors='ignore') and (path.parent / 'schemas').is_dir():
            return '.'.join(path.parent.relative_to(PROJECT_ROOT).parts)
    raise RuntimeError('Cannot locate UMA backend app.py under project root; set UMA_BACKEND_PACKAGE to its dotted package name.')

def module(suffix: str):
    return importlib.import_module(f'{package_name()}.{suffix}')

def fixture_values():
    return dict(drawing_number='A-001', project_name='Test Project', drawing_title='Ground Floor Plan', architect='Test Architect', draughtsperson='Designer', date='2026-01-01', scale='1:100', drawing_type='plan')
