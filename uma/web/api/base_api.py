from __future__ import annotations

from abc import ABC, abstractmethod

from fastapi import APIRouter


# Base class for API routers
class BaseAPI(ABC):

    # Route prefix and Swagger tags
    prefix: str = ""
    tags: list[str] = []

    def __init__(self) -> None:
        # Create the router
        self.router = APIRouter(prefix=self.prefix, tags=self.tags)

        # Register routes
        self.register_routes()

    # Must be implemented by subclasses
    @abstractmethod
    def register_routes(self) -> None:
        raise NotImplementedError