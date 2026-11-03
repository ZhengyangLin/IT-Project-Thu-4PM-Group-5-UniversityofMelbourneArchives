
from __future__ import annotations

import asyncio
import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient


def _package_name():
    from importlib.util import find_spec

    for name in ("uma.web", "uma.backend"):
        if find_spec(name) is not None:
            return name

    raise ImportError("Expected uma.web or uma.backend")


@pytest.fixture
def app_module(monkeypatch, tmp_path):
    package = _package_name()

    impl = importlib.import_module(
        f"{package}.services.impl.Unimelb_service_impl"
    )

    sample_dir = tmp_path / "images"
    sample_dir.mkdir()

    monkeypatch.setattr(
        impl, "PLAN_SAMPLE_DIR", sample_dir
    )
    monkeypatch.setattr(
        impl, "IMAGE_STATUS_FILE", tmp_path / "status.json"
    )

    return importlib.import_module(f"{package}.app")


@pytest.fixture
def fake_service():
    return SimpleNamespace(
        start=AsyncMock(),
        close=AsyncMock(),
        get_image_select=AsyncMock(),
        create_ocr_task=AsyncMock(),
        rerun_ocr_images=AsyncMock(),
        update_ocr_image_result=AsyncMock(),
        export_ocr_results=AsyncMock(),
        resolve_image=Mock(return_value=None),
    )


def _cors_middleware(app):
    matches = [
        m for m in app.user_middleware
        if m.cls is CORSMiddleware
    ]

    assert len(matches) == 1
    return matches[0]


def test_app_module_version_and_global_app(app_module):
    assert app_module.API_VERSION == "0.1.0"
    assert app_module.app.title == "UMA OCR API"
    assert isinstance(app_module.app, FastAPI)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param(
            None, ("*",), id="unset"
        ),
        pytest.param(
            "  ", ("*",), id="blank"
        ),
        pytest.param(
            ", ,", ("*",), id="empty_entries"
        ),
        pytest.param(
            "https://a.example",
            ("https://a.example",),
            id="one_origin"
        ),
        pytest.param(
            " https://a.example,https://b.example, https://a.example,",
            ("https://a.example", "https://b.example"),
            id="trim_deduplicate"
        ),
        pytest.param(
            "*,https://a.example",
            ("*", "https://a.example"),
            id="wildcard"
        ),
    ],
)
def test_cors_origins_from_environment(
    app_module, monkeypatch, raw, expected
):
    if raw is None:
        monkeypatch.delenv(
            "UMA_CORS_ORIGINS", raising=False
        )
    else:
        monkeypatch.setenv(
            "UMA_CORS_ORIGINS", raw
        )

    assert app_module._cors_origins_from_env() == expected


@pytest.mark.parametrize(
    ("supplied", "environment", "expected", "credentials"),
    [
        pytest.param(
            None, "*", ["*"], False,
            id="default_wildcard"
        ),
        pytest.param(
            None,
            "https://from-env.example",
            ["https://from-env.example"],
            True,
            id="from_environment"
        ),
        pytest.param(
            [],
            "https://ignored.example",
            ["*"],
            False,
            id="empty_falls_back"
        ),
        pytest.param(
            ["  ", ""],
            "https://ignored.example",
            ["*"],
            False,
            id="blank_falls_back"
        ),
        pytest.param(
            ["*", "https://a.example"],
            "",
            ["*"],
            False,
            id="wildcard_precedence"
        ),
        pytest.param(
            [
                " https://a.example ",
                "https://a.example",
                "https://b.example"
            ],
            "*",
            [
                "https://a.example",
                "https://b.example"
            ],
            True,
            id="explicit_trim_deduplicate"
        ),
    ],
)
def test_create_app_cors_configuration(
    app_module,
    fake_service,
    monkeypatch,
    supplied,
    environment,
    expected,
    credentials,
):
    monkeypatch.setenv(
        "UMA_CORS_ORIGINS", environment
    )

    app = app_module.create_app(
        cors_origins=supplied,
        unimel_service=fake_service
    )

    cors = _cors_middleware(app)

    assert cors.kwargs["allow_origins"] == expected
    assert cors.kwargs["allow_credentials"] is credentials
    assert cors.kwargs["allow_methods"] == ["*"]
    assert cors.kwargs["allow_headers"] == ["*"]
    assert cors.kwargs["max_age"] == 600

    assert "Content-Disposition" in (
        cors.kwargs["expose_headers"]
    )

    assert app.state.unimel_service is fake_service


def test_default_service_is_constructed_when_not_supplied(
    app_module, monkeypatch, fake_service
):
    factory = Mock(return_value=fake_service)

    monkeypatch.setattr(
        app_module, "UnimelbServiceImpl", factory
    )

    app = app_module.create_app(
        cors_origins=["*"],
        unimel_service=None
    )

    factory.assert_called_once_with()

    assert app.state.unimel_service is fake_service


def test_create_app_metadata(app_module, fake_service):
    app = app_module.create_app(
        unimel_service=fake_service
    )

    assert app.title == "UMA OCR API"
    assert app.version == app_module.API_VERSION
    assert "metadata extraction" in app.description
    assert app.state.unimel_service is fake_service


def test_health_endpoint_and_normal_lifecycle(
    app_module, fake_service
):
    app = app_module.create_app(
        unimel_service=fake_service
    )

    with TestClient(app) as client:
        response = client.get("/health")

        assert response.status_code == 200

        body = response.json()

        assert body["success"] is True
        assert body["code"] == 200

        assert body["result"] == {
            "status": "ok",
            "service": "uma-ocr",
            "version": app_module.API_VERSION,
        }

        assert isinstance(body["timestamp"], int)

        fake_service.start.assert_awaited_once_with()
        fake_service.close.assert_not_awaited()

    fake_service.close.assert_awaited_once_with()


def test_lifespan_closes_service_when_error_occurs(
    app_module, fake_service
):
    app = app_module.create_app(
        unimel_service=fake_service
    )

    async def run_lifespan():
        async with app.router.lifespan_context(app):
            raise RuntimeError(
                "simulated application failure"
            )

    with pytest.raises(
        RuntimeError,
        match="simulated application failure"
    ):
        asyncio.run(run_lifespan())

    fake_service.start.assert_awaited_once_with()
    fake_service.close.assert_awaited_once_with()


def test_create_app_mounts_unimelb_router(
    app_module, fake_service, monkeypatch
):
    router = APIRouter(prefix="/Unimelb")

    async def sample_endpoint():
        return {"ok": True}

    router.add_api_route(
        "/sample",
        sample_endpoint,
        methods=["GET"]
    )

    api_factory = Mock(
        return_value=SimpleNamespace(router=router)
    )

    monkeypatch.setattr(
        app_module, "UnimelbAPI", api_factory
    )

    included_routers = []

    original_include_router = (
        app_module.FastAPI.include_router
    )

    def recording_include_router(
        self, included_router, *args, **kwargs
    ):
        included_routers.append(included_router)

        return original_include_router(
            self, included_router, *args, **kwargs
        )

    monkeypatch.setattr(
        app_module.FastAPI,
        "include_router",
        recording_include_router
    )

    app = app_module.create_app(
        unimel_service=fake_service
    )

    api_factory.assert_called_once_with(fake_service)

    assert included_routers == [router]
    assert app.state.unimel_service is fake_service

    assert any(
        route.path == "/health"
        for route in app.router.routes
    )


def test_health_route_rejects_unregistered_path(
    app_module, fake_service
):
    app = app_module.create_app(
        unimel_service=fake_service
    )

    with TestClient(app) as client:
        response = client.get(
            "/route-that-does-not-exist"
        )

    assert response.status_code == 404


def test_cors_request_allows_configured_origin(
    app_module, fake_service
):
    app = app_module.create_app(
        cors_origins=["https://allowed.example"],
        unimel_service=fake_service
    )

    with TestClient(app) as client:
        response = client.get(
            "/health",
            headers={
                "Origin": "https://allowed.example"
            }
        )

    assert response.status_code == 200

    assert response.headers.get(
        "access-control-allow-origin"
    ) == "https://allowed.example"


def test_cors_request_does_not_allow_unknown_origin(
    app_module, fake_service
):
    app = app_module.create_app(
        cors_origins=["https://allowed.example"],
        unimel_service=fake_service
    )

    with TestClient(app) as client:
        response = client.get(
            "/health",
            headers={
                "Origin": "https://denied.example"
            }
        )

    assert response.status_code == 200

    assert (
        "access-control-allow-origin"
        not in response.headers
    )
