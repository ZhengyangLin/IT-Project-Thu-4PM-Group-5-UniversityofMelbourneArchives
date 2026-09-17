from __future__ import annotations

from pydantic import BaseModel


# Health check response model
class HealthResponse(BaseModel):
    status: str
    service: str
    version: str