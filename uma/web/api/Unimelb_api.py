from __future__ import annotations

import math
from threading import Lock
import time

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse

from .base_api import BaseAPI
from ..schemas import (
    UnimelbImageSelectResponse,
    UnimelbTaskCreateRequest,
    UnimelbTaskCreatedResponse,
    Result,
)
from ..services import (
    UnimelbService,
    UnimelbTaskRunningError,
)

# Minimum time between OCR task submissions
CREATE_OCR_TASK_COOLDOWN_SECONDS = 60


# API routes for Unimelb OCR
class UnimelbAPI(BaseAPI):
    prefix = "/Unimelb"
    tags = ["Unimelb"]
    create_task_cooldown_seconds = CREATE_OCR_TASK_COOLDOWN_SECONDS

    def __init__(self, service: UnimelbService) -> None:
        # Store the service
        self.service = service

        # Lock for task submission rate limiting
        self._create_task_rate_limit_lock = Lock()
        self._last_create_task_call_time: float | None = None
        super().__init__()

    # Register all API routes
    def register_routes(self) -> None:
        # Get image list
        self.router.add_api_route(
            "/ImageSelect",
            self.image_select,
            methods=["GET"],
            response_model=Result[UnimelbImageSelectResponse],
            summary="Query image list",
        )

        # Get a specific image
        self.router.add_api_route(
            "/images/{folder}/{node_id}/{image_name}",
            self.get_image,
            methods=["GET"],
            response_class=FileResponse,
            name="unimelb-sample-image",
            summary="Display the image to be recognized.",
        )

        # Create an OCR task
        self.router.add_api_route(
            "/ocr-tasks",
            self.create_ocr_task,
            methods=["POST"],
            response_model=Result[UnimelbTaskCreatedResponse],
            summary="Create an OCR recognition task",
        )

    # Return the image list
    async def image_select(
            self, request: Request,
    ) -> Result[UnimelbImageSelectResponse]:
        try:
            data = await self.service.get_image_select()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        # Generate URL for each image
        for item in data.images:
            item.image_url = str(request.url_for(
                "Unimelb-sample-image",
                folder=item.folder,
                node_id=item.node_id,
                image_name=item.image_name,
            ))
        return Result[UnimelbImageSelectResponse].ok(data)

    # Return a specific image file
    async def get_image(
            self, folder: str, node_id: str, image_name: str,
    ) -> FileResponse:
        image = self.service.resolve_image(folder, node_id, image_name)
        if image is None:
            raise HTTPException(status_code=404, detail="Image not found")
        return FileResponse(
            image,
            headers={"Cache-Control": "public, max-age=3600"},
        )

    # Create a new OCR task
    async def create_ocr_task(
            self, request: UnimelbTaskCreateRequest,
    ) -> Result[UnimelbTaskCreatedResponse]:
        # Check submission cooldown
        with self._create_task_rate_limit_lock:
            self._register_create_task_submission_locked()

        try:
            data = await self.service.create_ocr_task(request.force)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except UnimelbTaskRunningError as exc:
            return Result[UnimelbTaskCreatedResponse].error(
                message=str(exc),
                code=409,
            )
        return Result[UnimelbTaskCreatedResponse].ok(
            data,
            message="OCR task started." if data.started else "The shared task already exists.",
        )

    # Check OCR task submission rate
    def _register_create_task_submission_locked(self) -> None:
        now = self._create_task_call_time()
        last_call_time = self._last_create_task_call_time
        cooldown_seconds = self.create_task_cooldown_seconds

        # Reject requests during cooldown
        if (
                last_call_time is not None
                and now - last_call_time < cooldown_seconds
        ):
            retry_after_seconds = max(
                1,
                math.ceil(cooldown_seconds - (now - last_call_time)),
            )
            raise HTTPException(
                status_code=429,
                detail=(
                    "OCR task submissions are too frequent: another "
                    "submission is not allowed within one minute of the "
                    "first submission. "
                    f"Try again in {retry_after_seconds} seconds"
                ),
                headers={"Retry-After": str(retry_after_seconds)},
            )

        # Save the latest submission time
        self._last_create_task_call_time = now

    # Get monotonic time for cooldown calculation
    @staticmethod
    def _create_task_call_time() -> float:
        return time.monotonic()