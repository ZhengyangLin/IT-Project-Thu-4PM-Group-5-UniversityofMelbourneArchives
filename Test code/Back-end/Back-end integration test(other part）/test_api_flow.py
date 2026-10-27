import pytest
from fastapi.testclient import TestClient

from uma.web.api.Unimelb_api import UnimelbAPI
from uma.web.services import UnimelbTaskRunningError
from conftest import REVIEW_VALUES
from uma.web.app import create_app


def test_http_routes_health_images_tasks_review_and_download(web_client, web_service):
    assert web_service.started
    health = web_client.get("/health").json()
    assert health["success"] and health["result"]["status"] == "ok"
    assert web_client.get("/openapi.json").status_code == 200
    listing = web_client.get("/Unimelb/ImageSelect").json()["result"]
    image = listing["images"][0]
    response = web_client.get(image["image_url"])
    assert response.content == web_service.image.read_bytes()
    assert response.headers["cache-control"] == "public, max-age=3600"
    assert web_client.get("/Unimelb/images/YFA/123/missing.png").status_code == 404
    started = web_client.post("/Unimelb/ocr-tasks", json={"force": False}).json()
    assert started["result"]["started"] is False
    assert "already exists" in started["message"]
    rerun = web_client.post("/Unimelb/ocr-tasks/rerun", json={"image_keys": [image["image_key"]]})
    assert rerun.json()["result"]["selected_total"] == 1
    saved = web_client.post("/Unimelb/ocr-results/manual-review", json={
        "image_key": image["image_key"], "result": REVIEW_VALUES,
    })
    assert saved.json()["result"]["result"]["drawing_number"] == "003/A2"
    assert web_service.calls[-1] == ("review", image["image_key"], REVIEW_VALUES)
    assert web_client.get("/Unimelb/ImageSelect").json()["result"]["images"][0]["manual_reviewed"] == 1
    exported = web_client.post("/Unimelb/ocr-results/export", json={
        "image_keys": [image["image_key"]], "status": 2, "folder": "YFA",
    })
    assert exported.status_code == 200
    assert exported.content == b"workbook-content"
    assert exported.headers["content-type"] == (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    assert "attachment" in exported.headers["content-disposition"]
    assert exported.headers["cache-control"] == "no-store"
    assert web_service.calls[-1] == ("export", [image["image_key"]], 2, "YFA")
    all_records = web_client.post("/Unimelb/ocr-results/export", json={})
    assert all_records.status_code == 200
    assert all_records.content == b"workbook-content"
    assert web_service.calls[-1] == ("export", None, None, None)


@pytest.mark.parametrize("method,url,payload,error,code", [
    ("list", "/Unimelb/ImageSelect", None, FileNotFoundError("missing"), 404),
    ("review", "/Unimelb/ocr-results/manual-review",
     {"image_key": "123/drawing.png", "result": REVIEW_VALUES}, FileNotFoundError("missing"), 404),
    ("export", "/Unimelb/ocr-results/export", {}, FileNotFoundError("missing"), 404),
    # One unexpected error exercises the shared 500 handler for all routes.
    ("list", "/Unimelb/ImageSelect", None, RuntimeError("unexpected"), 500),
])
def test_http_errors_have_consistent_json(web_client, web_service, method, url, payload, error, code):
    web_service.errors[method] = error
    response = web_client.get(url) if payload is None else web_client.post(url, json=payload)
    assert response.status_code == code
    data = response.json()
    assert data["success"] is False and data["code"] == code
    assert data["message"] == ("Internal server error" if code == 500 else "missing")
    assert isinstance(data["timestamp"], int)


@pytest.mark.parametrize("method,url,payload", [
    ("create", "/Unimelb/ocr-tasks", {"force": True}),
    ("rerun", "/Unimelb/ocr-tasks/rerun", {"image_keys": ["123/drawing.png"]}),
    ("review", "/Unimelb/ocr-results/manual-review", {"image_key": "123/drawing.png", "result": REVIEW_VALUES}),
])
def test_running_task_conflicts_follow_current_http_contract(web_client, web_service, method, url, payload):
    web_service.errors[method] = UnimelbTaskRunningError("busy")
    response = web_client.post(url, json=payload)
    # The create endpoint currently wraps a conflict in HTTP 200; other routes use HTTP 409.
    assert response.status_code == (200 if method == "create" else 409)
    assert response.json()["code"] == 409
    assert response.json()["success"] is False


def test_task_cooldown_preserves_retry_header_then_allows_retry(web_service, monkeypatch):
    clock = iter([100.0, 110.0, 161.0])
    monkeypatch.setattr(UnimelbAPI, "_create_task_call_time", staticmethod(lambda: next(clock)))
    with TestClient(create_app(unimel_service=web_service)) as client:
        assert client.post("/Unimelb/ocr-tasks", json={"force": True}).json()["result"]["started"]
        limited = client.post("/Unimelb/ocr-tasks", json={})
        assert limited.status_code == 429
        assert limited.headers["retry-after"] == "50"
        assert limited.json()["code"] == 429
        assert client.post("/Unimelb/ocr-tasks", json={}).status_code == 200
    assert web_service.closed


@pytest.mark.parametrize("origins", [None, [" ", "http://client.test", "http://client.test"]])
def test_cors_configuration_and_service_lifespan(web_service, monkeypatch, origins):
    monkeypatch.setenv("UMA_CORS_ORIGINS", "http://client.test, http://client.test, ")
    app = create_app(cors_origins=origins, unimel_service=web_service)
    with TestClient(app) as client:
        response = client.options("/health", headers={
            "Origin": "http://client.test", "Access-Control-Request-Method": "GET",
        })
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] in {"*", "http://client.test"}
        assert web_service.started
    assert web_service.closed


def test_web_public_exports_and_default_app_factory(web_service, monkeypatch):
    import importlib
    import uma.web as web
    module = importlib.import_module("uma.web.app")
    monkeypatch.setenv("UMA_CORS_ORIGINS", " , ")
    monkeypatch.setattr(module, "UnimelbServiceImpl", lambda: web_service)
    with TestClient(module.create_app()) as client:
        assert client.get("/health").status_code == 200
    assert web.__getattr__("create_app") is module.create_app
    assert web.__getattr__("app") is module.app
    assert web.__getattr__("UnimelbAPI") is UnimelbAPI
    assert web.__getattr__("BaseAPI").__name__ == "BaseAPI"
    with pytest.raises(AttributeError):
        web.__getattr__("missing_export")
