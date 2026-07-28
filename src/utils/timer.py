"""Performance timing utilities."""

from __future__ import annotations

import time
from collections.abc import Callable
from functools import wraps
from typing import Any


def timed(func: Callable) -> Callable:
    """Decorator to log function execution time."""

    @wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000
        print(f"[timer] {func.__name__} took {elapsed:.1f}ms")
        return result

    @wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000
        print(f"[timer] {func.__name__} took {elapsed:.1f}ms")
        return result

    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper
