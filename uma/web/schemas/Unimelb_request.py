from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


# Request model for creating an OCR task
class UnimelbTaskCreateRequest(BaseModel):

    # Whether to force the task to rerun
    force: bool = Field(
        default=False,
        description="Whether to force a rerun of a shared task that has already completed.",
    )

# Request model for rerunning OCR on selected images.
class UnimelbTaskRerunRequest(BaseModel):

    image_keys: list[str] = Field(
        description="List of unique identifiers for images to be rerun, "
                    "with each item in the format of NodeID/IMAGENAME",
    )
