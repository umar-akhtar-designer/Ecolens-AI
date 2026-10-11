"""Material guidance engine backed by a local knowledge base."""

from .engine import (
    GuidanceEngine,
    GuidanceItem,
    KnowledgeBaseError,
    get_engine,
)

__all__ = ["GuidanceEngine", "GuidanceItem", "KnowledgeBaseError", "get_engine"]
