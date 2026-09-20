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



# Defines the final values for the eight OCR result fields.
class UnimelbOCRResultValues(BaseModel):

    model_config = ConfigDict(extra="forbid")

    drawing_number: str | None = Field(description="Final Drawing Number Value")
    project_name: str | None = Field(description="Final Value of Project Name")
    drawing_title: str | None = Field(description="Final value of the drawing title")
    architect: str | None = Field(description="Architect's Final Value")
    draughtsperson: str | None = Field(description="Cartographer's Final Value")
    date: str | None = Field(description="Final date value")
    scale: str | None = Field(description="Final ratio value")
    drawing_type: str | None = Field(description="Final value of drawing type")


# Defines the request data for updating an image OCR result.
class UnimelbImageResultUpdateRequest(BaseModel):

    model_config = ConfigDict(extra="forbid")

    image_key: str = Field(
        description="Unique identifier of the image to be modified; format: NODEID/IMAGENAME",
    )
    result: UnimelbOCRResultValues = Field(
        description="The final values for the eight OCR fields; the fields must be fully populated, though the values themselves may be null.",
    )