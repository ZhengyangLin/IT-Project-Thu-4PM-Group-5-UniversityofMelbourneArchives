from __future__ import annotations

from pydantic import BaseModel, Field


# OCR result for a single field
class UnimelbOCRFieldResult(BaseModel):

    # Final field value
    value: str | None = Field(
        default=None,
        description="For the final field value, `corrected` takes precedence; otherwise, `verbatim` is used.",
    )

    # Original OCR text
    verbatim: str | None = Field(
        default=None,
        description="Original OCR text; no normalization or error correction performed.",
    )

    # Corrected field value
    corrected: str | None = Field(
        default=None,
        description="Normalized or corrected field values",
    )

    # Basis for correction
    correction_basis: str | None = Field(
        default=None,
        description="The basis for character error correction",
    )

    # Confidence score
    confidence: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Field confidence, ranging from 0 to 1.",
    )

    # Source of the field
    source: str | None = Field(
        default=None,
        description="Field source, e.g., rule or llm",
    )

    # Referenced OCR tokens
    tokens: list[int] = Field(
        default_factory=list,
        description="OCR token number referenced by the field",
    )

    # Bounding box of tokens
    tokens_bbox: dict[str, int] | None = Field(
        default=None,
        description="Bounding box of the merged referenced tokens: x0, y0, x1, y1",
    )

    # Reason for missing value
    reason: str | None = Field(
        default=None,
        description=(
            "Structured reason codes for cases where a field cannot be assigned a value, such as `not_present` or `ambiguous`."
        ),
    )

    # Additional notes
    notes: list[str] = Field(
        default_factory=list,
        description="Structured reason codes for cases where a field cannot be assigned a value, such as `not_present` or `ambiguous`.",
    )

    # Review status
    review_status: str | None = Field(
        default=None,
        description="Field review status, e.g., auto_accept or needs_review.",
    )


# Information for one image
class UnimelbImageSelectItem(BaseModel):
    path: str = Field(description="FOLDER/NODEID/IMAGENAME Display Path")
    image_key: str = Field(description="Unique Identifier: NODEID/IMAGENAME")
    folder: str
    node_id: str
    image_name: str
    image_url: str = Field(description="Image URL directly displayable on the web")
    status: int = Field(
        ge=0,
        le=3,
        description="Image status: 0 Pending detection, 1 Detecting, 2 Completed, 3 Failed",
    )
    task_uuid: str | None = None
    error: str | None = None
    message: str | None = Field(default=None, description="Image failure information")
    updated_at: int | None = Field(default=None, description="Update timestamp (milliseconds)")
    manual_reviewed: int = Field(
        default=0,
        ge=0,
        le=1,
        description="Has manual review been completed: 0 = Not reviewed, 1 = Reviewed",
    )
    result: dict[str, UnimelbOCRFieldResult] = Field(
        default_factory=dict,
        description="OCR results, evidence, and review information grouped by field name",
    )


# Response containing image list
class UnimelbImageSelectResponse(BaseModel):
    task_uuid: str = Field(description="OCR task UUID shared by all users")
    folders: dict[str, int] = Field(description="Number of images in each recognizable directory")
    total: int = Field(description="Total number of images to be recognized")
    status_counts: dict[str, int] = Field(
        description="Counts for each image state; keys are 0, 1, 2, and 3."
    )
    images: list[UnimelbImageSelectItem]


# Response after creating an OCR task
class UnimelbTaskCreatedResponse(BaseModel):
    task_uuid: str
    status: int = Field(
        ge=0,
        le=3,
        description="Task status: 0 Pending, 1 In progress, 2 Completed, 3 Failed",
    )
    total: int
    started: bool = Field(description="Did this request actually trigger OCR?")