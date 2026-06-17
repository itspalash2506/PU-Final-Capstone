import re
from fastapi import HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader

from backend.core.config import get_settings

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# Characters allowed in a query — printable ASCII minus control chars
_ALLOWED_QUERY_RE = re.compile(r"^[\x20-\x7E\n\t]+$")

# Simple prompt injection patterns to catch obvious attacks
_INJECTION_PATTERNS = [
    r"ignore (all |previous |above )?instructions",
    r"system prompt",
    r"you are now",
    r"disregard (all |your )?",
    r"forget (everything|all)",
    r"act as (a |an )?(?!technician|engineer|maintenance)",  # Allow legitimate roles
]
_INJECTION_RE = re.compile(
    "|".join(_INJECTION_PATTERNS), re.IGNORECASE
)

MAX_QUERY_LENGTH = 500


def validate_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    """
    FastAPI dependency — inject into any route that requires auth.
    Returns the key if valid, raises 401 otherwise.

    Usage:
        @router.post("/query")
        async def query(api_key: str = Depends(validate_api_key)):
            ...
    """
    settings = get_settings()

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header",
        )

    if api_key != settings.amop_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    return api_key


def sanitize_query(query: str) -> str:
    """
    Validate and sanitize a user query string.
    Raises HTTPException 400 on any violation.
    Returns the stripped query on success.
    """
    if not query or not query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query must not be empty",
        )

    query = query.strip()

    if len(query) > MAX_QUERY_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Query exceeds maximum length of {MAX_QUERY_LENGTH} characters",
        )

    if not _ALLOWED_QUERY_RE.match(query):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query contains disallowed characters",
        )

    if _INJECTION_RE.search(query):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query contains disallowed patterns",
        )

    return query
