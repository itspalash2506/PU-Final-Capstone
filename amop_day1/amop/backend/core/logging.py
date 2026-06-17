import logging
import json
import sys
from datetime import datetime, timezone
from typing import Any


class JSONFormatter(logging.Formatter):
    """
    Emits log records as single-line JSON objects.
    Fields: timestamp, level, logger, message, plus any extras passed
    via logger.info("msg", extra={"request_id": "...", "agent": "..."})
    """

    RESERVED = {"message", "asctime", "levelname", "name", "pathname", "lineno"}

    def format(self, record: logging.LogRecord) -> str:
        log_obj: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Attach any extra fields the caller passed in
        for key, value in record.__dict__.items():
            if key not in self.RESERVED and not key.startswith("_"):
                if key not in logging.LogRecord.__dict__:
                    log_obj[key] = value

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj, default=str)


def setup_logging(level: str = "INFO") -> None:
    """
    Call once at application startup (in main.py lifespan).
    Replaces the root logger handler with our JSON formatter.
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove any existing handlers (uvicorn adds its own)
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    root.addHandler(handler)

    # Quiet down noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Convenience wrapper — use in every module:
        logger = get_logger(__name__)
    """
    return logging.getLogger(name)
