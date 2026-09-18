from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..schemas import (
    UnimelbImageSelectResponse,
    UnimelbTaskCreatedResponse,
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