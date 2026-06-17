"""
Core cross-cutting concerns: configuration, logging, and security.
"""

from backend.core.config import Settings, get_settings
from backend.core.logging import JSONFormatter, setup_logging, get_logger
from backend.core.security import (
    API_KEY_HEADER,
    MAX_QUERY_LENGTH,
    validate_api_key,
    sanitize_query,
)

__all__ = [
    # Config
    "Settings",
    "get_settings",
    # Logging
    "JSONFormatter",
    "setup_logging",
    "get_logger",
    # Security
    "API_KEY_HEADER",
    "MAX_QUERY_LENGTH",
    "validate_api_key",
    "sanitize_query",
]
