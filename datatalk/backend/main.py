"""DataTalk application entry point.

Run with:
    uvicorn backend.main:app --reload
"""

try:
    from .interface import app
except ImportError:
    from interface import app

__all__ = ["app"]
