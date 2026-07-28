"""API dependency injection helpers.

We use FastAPI app.state for simple DI rather than full DI framework.
This module provides factory functions for services when needed outside request context.
"""

from __future__ import annotations
