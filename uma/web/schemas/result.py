from __future__ import annotations

import time
from typing import Generic, TypeVar

from pydantic import BaseModel, Field

# Generic type for result data
T = TypeVar("T")


# Standard API response model
class Result(BaseModel, Generic[T]):
    success: bool = Field(default=True, description="Sign of success")
    message: str = Field(default="", description="Return processing message")
    code: int = Field(default=200, description="Return code")
    result: T | None = Field(default=None, description="Return data object")
    timestamp: int = Field(
        default_factory=lambda: int(time.time() * 1000),
        description="Response timestamp (milliseconds)",
    )

    # Create a successful response
    @classmethod
    def ok(cls, result: T | None = None,
           message: str = "Operation successful.") -> "Result[T]":
        return cls(
            success=True,
            message=message,
            code=200,
            result=result,
        )

    # Create an error response
    @classmethod
    def error(cls, message: str = "Operation failed", code: int = 500,
              result: T | None = None) -> "Result[T]":
        return cls(
            success=False,
            message=message,
            code=code,
            result=result,
        )

    # Create an access denied response
    @classmethod
    def noauth(cls, message: str = "Access denied") -> "Result[T]":
        return cls.error(message=message, code=403)