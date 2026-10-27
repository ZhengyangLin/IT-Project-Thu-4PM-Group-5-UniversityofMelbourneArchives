"""Isolated web fixtures: real HTTP/models/files, no OCR or model inference."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from uma.models import FieldValue, SheetRecord
from uma.unimelb_result_codec import extract_ocr_result
from uma.pipeline import record_to_row
from uma.web.schemas import (
    UnimelbImageSelectItem, UnimelbImageSelectResponse,
    UnimelbImageResultUpdateResponse, UnimelbTaskCreatedResponse,
    UnimelbTaskRerunResponse,
)
from uma.web.services import UnimelbService
from uma.web.services.impl import Unimelb_service_impl as service_module
from uma.web.workers.Unimelb_ocr_worker import OcrImageOutcome

# Importing app creates a default service. Isolate that one construction too.
with TemporaryDirectory(prefix="uma-web-import-") as sandbox:
    with patch.object(service_module, "IMAGE_STATUS_FILE", Path(sandbox) / "status.json"), \
            patch.object(service_module, "PLAN_SAMPLE_DIR", Path(sandbox) / "images"):
        from uma.web.app import create_app


REVIEW_VALUES = {
    "drawing_number": "003/A2", "project_name": "STATE GOVERNMENT OFFICES",
    "drawing_title": "MULLION DETAILS", "architect": "YUNCKEN FREEMAN",
    "draughtsperson": None, "date": "4.11.63", "scale": "1:4",
    "drawing_type": "Detail",
}


class WebServiceStub(UnimelbService):
    """Substitute only the service boundary when testing HTTP contracts."""
    def __init__(self, image):
        self.image = image
        self.errors = {}
        self.calls = []
        self.started = self.closed = False
        self.reviewed = False

    def check(self, name):
        if name in self.errors:
            raise self.errors[name]

    async def start(self):
        await super().start()
        self.started = True

    async def close(self):
        await super().close()
        self.closed = True

    async def get_image_select(self):
        self.check("list")
        return UnimelbImageSelectResponse(
            task_uuid="test-task", folders={"YFA": 1}, total=1,
            status_counts={"0": 0, "1": 0, "2": 1, "3": 0},
            images=[UnimelbImageSelectItem(
                path="YFA/123/drawing.png", image_key="123/drawing.png",
                folder="YFA", node_id="123", image_name="drawing.png",
                image_url="", status=2, manual_reviewed=int(self.reviewed),
                result={"drawing_number": {"value": "003/A2", "confidence": .9}},
            )],
        )

    def resolve_image(self, folder, node_id, image_name):
        return self.image if (folder, node_id, image_name) == ("YFA", "123", "drawing.png") else None

    async def create_ocr_task(self, force):
        self.check("create")
        self.calls.append(("create", force))
        return UnimelbTaskCreatedResponse(task_uuid="test-task", status=1, total=1, started=force)

    async def rerun_ocr_images(self, image_keys):
        self.check("rerun")
        self.calls.append(("rerun", image_keys))
        return UnimelbTaskRerunResponse(
            task_uuid="test-task", status=1, selected_total=len(image_keys),
            image_keys=image_keys, started=True,
        )

    async def update_ocr_image_result(self, image_key, result):
        self.check("review")
        self.calls.append(("review", image_key, result))
        self.reviewed = True
        return UnimelbImageResultUpdateResponse(
            image_key=image_key, manual_reviewed=1, updated_at=123, result=result,
        )

    async def export_ocr_results(self, image_keys, status=None, folder=None):
        self.check("export")
        self.calls.append(("export", image_keys, status, folder))
        return b"workbook-content"


@pytest.fixture
def web_service(tmp_path):
    image = tmp_path / "drawing.png"
    image.write_bytes(b"test-image-bytes")
    return WebServiceStub(image)


@pytest.fixture
def web_client(web_service):
    with TestClient(create_app(unimel_service=web_service), raise_server_exceptions=False) as client:
        yield client


class LocalWorkerPool(ThreadPoolExecutor):
    """Use local threads for dispatch; never run a process/OCR initializer."""
    def __init__(self, max_workers, **kwargs):
        super().__init__(max_workers=max_workers)
        self._processes = {}


@pytest.fixture
def service_env(tmp_path, monkeypatch):
    root = tmp_path / "project"
    images_root = root / "plan_sample"
    paths = []
    for folder, node in [("YFA", "100"), ("YFA", "101"), ("OTHER", "102")]:
        path = images_root / folder / node / "drawing.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test-image-bytes")
        paths.append(path)
    excluded = images_root / "bates_smart_mccutcheon" / "999" / "ignored.jpg"
    excluded.parent.mkdir(parents=True)
    excluded.write_bytes(b"excluded")
    (images_root / "notes.txt").write_text("not an image directory")
    (images_root / "YFA" / "100" / "notes.txt").write_text("not an image")
    config = root / "config.yaml"
    config.write_text("conf:\n  t_high: 0.75\n")
    for name, value in {
        "PROJECT_DIR": root, "PLAN_SAMPLE_DIR": images_root,
        "UNIMELB_OCR_OUTPUT_ROOT": root / "output", "UNIMELB_CONFIG_PATH": config,
        "UNIMELB_LOG_DIR": root / "output/logs",
        "IMAGE_STATUS_FILE": root / "output/status.json",
    }.items():
        monkeypatch.setattr(service_module, name, value)
    monkeypatch.setattr(service_module, "ProcessPoolExecutor", LocalWorkerPool)

    def inference_result(job):
        # Controlled inference output; downstream codec and summary functions are real.
        record = SheetRecord(node_id=job.node_id, path=job.image_path, fields={
            "drawing_number": FieldValue(name="drawing_number", verbatim="003/A1", confidence=.95),
        })
        return OcrImageOutcome(ok=True, result=extract_ocr_result(record), summary_row=record_to_row(record))

    monkeypatch.setattr(service_module, "process_ocr_image", inference_result)
    services = []

    def make_service():
        service = service_module.UnimelbServiceImpl(status_file=root / "output/status.json")
        services.append(service)
        return service

    yield root, paths, make_service
    for service in services:
        # Tests close their own async sessions; clean up pools if an assertion fails.
        executor = service._ocr_executor
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)


async def drain(service):
    while service._background_tasks:
        await asyncio.wait_for(asyncio.gather(*tuple(service._background_tasks)), timeout=5)
