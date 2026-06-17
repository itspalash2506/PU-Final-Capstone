import os
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator

from backend.db.models import Base
from backend.core.config import get_settings
from backend.core.logging import get_logger

logger = get_logger(__name__)


def _get_engine():
    settings = get_settings()

    # Ensure the data directory exists
    db_dir = os.path.dirname(settings.sqlite_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    engine = create_engine(
        f"sqlite:///{settings.sqlite_path}",
        connect_args={"check_same_thread": False},  # Required for FastAPI async
        echo=False,
    )

    # Enable WAL mode on every new connection.
    # WAL allows concurrent reads while agents write traces/logs.
    # Without this, a write will block all reads on the connection.
    @event.listens_for(engine, "connect")
    def set_wal_mode(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")   # Safe with WAL, much faster
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return engine


# Module-level singletons — created once at import time
_engine = _get_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)


def init_db() -> None:
    """
    Create all tables if they don't exist.
    Safe to call on every startup — idempotent.
    """
    Base.metadata.create_all(bind=_engine)
    logger.info("SQLite database initialized", extra={"path": get_settings().sqlite_path})


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that yields a database session and ensures cleanup.

    Usage in a route:
        @router.post("/feedback")
        async def submit_feedback(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
