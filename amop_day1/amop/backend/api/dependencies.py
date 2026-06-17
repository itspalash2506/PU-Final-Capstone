from fastapi import Depends
from sqlalchemy.orm import Session

from backend.db.session import get_db
from backend.core.config import get_settings, Settings
from backend.core.security import validate_api_key

# Re-export for convenient importing in routes
__all__ = ["get_db", "get_settings", "validate_api_key"]
