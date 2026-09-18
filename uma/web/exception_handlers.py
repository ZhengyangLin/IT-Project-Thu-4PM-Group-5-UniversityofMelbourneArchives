from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .schemas import Result

log = logging.getLogger(__name__)


def _json(
    result: Result[Any],
    status_code: int,
    *,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=result.model_dump(mode="json"),
        headers=headers,
    )


def register_exception_handlers(application: FastAPI) -> None:

    @application.exception_handler(HTTPException)
    async def handle_http_exception(
        request: Request, exc: HTTPException,
    ) -> JSONResponse:
        return _json(
            Result[Any].error(message=str(exc.detail), code=exc.status_code),
            exc.status_code,
            headers=exc.headers,
        )

    @application.exception_handler(RequestValidationError)
    async def handle_validation_exception(
        request: Request, exc: RequestValidationError,
    ) -> JSONResponse:
        return _json(
            Result[list[dict[str, Any]]].error(
                message="Request validation failed",
                code=422,
                result=exc.errors(),
            ),
            422,
        )

    @application.exception_handler(Exception)
    async def handle_unexpected_exception(
        request: Request, exc: Exception,
    ) -> JSONResponse:
        log.exception("Unhandled API exception: method=%s path=%s",
                      request.method, request.url.path)
        return _json(Result[Any].error(message="Internal server error"), 500)
