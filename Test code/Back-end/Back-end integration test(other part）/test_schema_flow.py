"""Schema contracts exercised through real FastAPI requests and responses."""
import pytest
from fastapi import HTTPException

from conftest import REVIEW_VALUES
from uma.web.schemas import Result, UnimelbImageSelectResponse


def test_export_filter_normalization_reaches_service(web_client, web_service):
    response = web_client.post("/Unimelb/ocr-results/export", json={"status": " ", "folder": " "})
    assert response.status_code == 200
    assert web_service.calls[-1] == ("export", None, None, None)


@pytest.mark.parametrize("url,payload", [
    ("/Unimelb/ocr-results/export", {"status": -1}),
    ("/Unimelb/ocr-results/export", {"status": 4}),
    ("/Unimelb/ocr-tasks/rerun", {"image_keys": "not-a-list"}),
    ("/Unimelb/ocr-results/manual-review", {"image_key": "123/drawing.png", "result": {}}),
    ("/Unimelb/ocr-results/manual-review", {"image_key": "123/drawing.png", "result": REVIEW_VALUES, "extra": True}),
    ("/Unimelb/ocr-results/manual-review", {"image_key": "123/drawing.png", "result": {**REVIEW_VALUES, "extra": True}}),
])
def test_bad_requests_are_rejected_before_service_calls(web_client, web_service, url, payload):
    response = web_client.post(url, json=payload)
    assert response.status_code == 422
    body = response.json()
    assert body["code"] == 422 and body["success"] is False
    assert body["result"] and body["result"][0]["loc"]
    assert web_service.calls == []


def test_image_response_roundtrip_preserves_typed_fields(web_client):
    body = web_client.get("/Unimelb/ImageSelect").json()
    parsed = Result[UnimelbImageSelectResponse].model_validate(body)
    field = parsed.result.images[0].result["drawing_number"]
    assert field.value == "003/A2" and field.confidence == .9
    assert field.tokens == [] and field.notes == []
    assert parsed.model_dump(mode="json") == body


@pytest.mark.parametrize("values", [REVIEW_VALUES, dict.fromkeys(REVIEW_VALUES)], ids=["populated", "null"])
def test_review_values_and_health_response_survive_http(web_client, web_service, values):
    response = web_client.post("/Unimelb/ocr-results/manual-review", json={
        "image_key": "123/drawing.png", "result": values,
    })
    assert response.status_code == 200
    assert response.json()["result"]["result"] == values
    assert web_service.calls[-1][2] == values
    health = web_client.get("/health").json()
    assert health["result"] == {"status": "ok", "service": "uma-ocr", "version": "0.1.0"}


def test_denied_result_preserves_envelope_and_http_headers(web_client):
    @web_client.app.get("/test-denied")
    def denied():
        result = Result.noauth("Review permission required")
        raise HTTPException(status_code=result.code, detail=result.message, headers={"X-Reason": "permission"})

    response = web_client.get("/test-denied")
    assert response.status_code == 403
    assert response.headers["x-reason"] == "permission"
    assert response.json()["message"] == "Review permission required"
