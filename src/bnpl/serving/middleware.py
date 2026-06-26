"""FastAPI middleware: request logging and rate limiting.

Provides RequestLoggingMiddleware and RateLimitMiddleware for
production API observability and protection.
"""

from __future__ import annotations

import time
from collections import defaultdict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from bnpl.logger import get_logger

logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every HTTP request with method, path, status, and duration.

    Provides per-request observability for debugging and monitoring.
    Logs at INFO level for successful requests and WARNING for errors.

    Usage::

        app.add_middleware(RequestLoggingMiddleware)

    Depends on:
        - Starlette BaseHTTPMiddleware
        - bnpl.logger for structured logging
    """

    async def dispatch(self, request: Request, call_next: object) -> Response:
        """Process a request and log its details.

        Args:
            request: The incoming HTTP request.
            call_next: The next middleware or route handler.

        Returns:
            Response: The HTTP response from downstream.
        """
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start

        logger.info(
            "%s %s -> %d (%.3fs)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
        )
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiter returning 429 on excess requests.

    Tracks request counts per client IP within a sliding window.
    Defaults to 100 requests per 60 seconds, configurable at init.

    Usage::

        app.add_middleware(RateLimitMiddleware, max_requests=100, window_seconds=60)

    Depends on:
        - Starlette BaseHTTPMiddleware
    """

    def __init__(
        self,
        app: object,
        max_requests: int = 100,
        window_seconds: int = 60,
    ) -> None:
        """Initialize rate limiter.

        Args:
            app: The ASGI application.
            max_requests: Maximum requests per window per IP.
            window_seconds: Time window in seconds.
        """
        super().__init__(app)
        self._max_requests = max_requests
        self._window = window_seconds
        self._counts: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next: object) -> Response:
        """Check rate limit before processing the request.

        Args:
            request: The incoming HTTP request.
            call_next: The next middleware or route handler.

        Returns:
            Response: 429 if rate limited, otherwise downstream response.
        """
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        timestamps = self._counts[client_ip]
        cutoff = now - self._window
        self._counts[client_ip] = [t for t in timestamps if t > cutoff]

        if len(self._counts[client_ip]) >= self._max_requests:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again later."},
            )

        self._counts[client_ip].append(now)
        return await call_next(request)
