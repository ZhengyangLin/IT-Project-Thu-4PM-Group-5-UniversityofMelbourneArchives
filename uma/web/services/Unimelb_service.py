from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..schemas import (
    UnimelbImageSelectResponse,
    UnimelbTaskCreatedResponse,
    UnimelbTaskRerunResponse,
    UnimelbImageResultUpdateResponse
)


# Error raised when an OCR task is already running
class UnimelbTaskRunningError(RuntimeError):
    pass


# Base class for Unimelb services
class UnimelbService(ABC):

    # Get the image list
    @abstractmethod
    async def get_image_select(self) -> UnimelbImageSelectResponse:
        raise NotImplementedError

    # Resolve the path of an image
    @abstractmethod
    def resolve_image(
        self, folder: str, node_id: str, image_name: str,
    ) -> Path | None:
        raise NotImplementedError

    # Create an OCR task
    @abstractmethod
    async def create_ocr_task(self, force: bool) -> UnimelbTaskCreatedResponse:
        raise NotImplementedError

    # Rerun OCR for the selected images.
    @abstractmethod
    async def rerun_ocr_images(
        self, image_keys: list[str],
    ) -> UnimelbTaskRerunResponse:
        raise NotImplementedError


    # Updates and saves the OCR result for a specified image.
    @abstractmethod
    async def update_ocr_image_result(
        self,
        image_key: str,
        result: dict[str, str | None],
    ) -> UnimelbImageResultUpdateResponse:
        raise NotImplementedError

    async def close(self) -> None:
        pass

    async def start(self) -> None:
        pass