from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from collections.abc import Iterable
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import BaseAPI, UnimelbAPI
from .exception_handlers import register_exception_handlers
from .schemas import HealthResponse, Result
from .services import UnimelbService
from .services.impl import UnimelbServiceImpl

log = logging.getLogger(__name__)

API_VERSION = "0.1.0"
CORS_ORIGINS_ENV = "UMA_CORS_ORIGINS"
DEFAULT_CORS_ORIGINS: tuple[str, ...] = ("*",)


# Get CORS origins from environment variables
def _cors_origins_from_env() -> tuple[str, ...]:
    raw = os.getenv(CORS_ORIGINS_ENV, "*")
    origins = tuple(dict.fromkeys(
        origin.strip() for origin in raw.split(",") if origin.strip()
    ))
    return origins or DEFAULT_CORS_ORIGINS


# Create and configure the FastAPI application
def create_app(
    cors_origins: Iterable[str] | None = None,
    unimel_service: UnimelbService | None = None,
) -> FastAPI:
    # Configure CORS origins
    origins = (
        tuple(cors_origins)
        if cors_origins is not None
        else _cors_origins_from_env()
    )
    origins = tuple(dict.fromkeys(
        origin.strip() for origin in origins if origin.strip()
    ))
    if not origins:
        origins = DEFAULT_CORS_ORIGINS

    # Create the service and API modules
    service = unimel_service or UnimelbServiceImpl()
    unimel_api = UnimelbAPI(service)
    api_modules: tuple[BaseAPI, ...] = (unimel_api,)

    # Manage service startup and shutdown
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await service.start()
        try:
            yield
        finally:
            await service.close()

    # Create the FastAPI application
    application = FastAPI(
        title="UMA OCR API",
        description="OCR and metadata extraction service for architectural drawing title blocks",
        version=API_VERSION,
        lifespan=lifespan,
    )
    application.state.unimel_service = service

    # Configure CORS middleware
    allow_all_origins = "*" in origins
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if allow_all_origins else list(origins),
        allow_credentials=not allow_all_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Retry-After", "Content-Disposition"],
        max_age=600,
    )

    # Register exception handlers
    register_exception_handlers(application)

    # Health check endpoint
    @application.get(
        "/health",
        tags=["system"],
        summary="Service health check",
        response_model=Result[HealthResponse],
    )
    async def health() -> Result[HealthResponse]:
        data = HealthResponse(
            status="ok", service="uma-ocr", version=API_VERSION)
        return Result[HealthResponse].ok(data)

    # Register API routers
    for api_module in api_modules:
        application.include_router(api_module.router)

    log.info(
        "FastAPI application created: version=%s api_modules=%s cors_origins=%s",
        API_VERSION,
        [type(module).__name__ for module in api_modules],
        origins,
    )
    return application


app = create_app()