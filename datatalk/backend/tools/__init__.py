"""Outils backend DataTalk."""

try:
    from backend.llm import claude_generate
except ImportError:  # pragma: no cover
    from llm import claude_generate

__all__ = ["claude_generate"]
