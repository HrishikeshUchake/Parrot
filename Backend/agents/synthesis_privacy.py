from __future__ import annotations

import logging

from ..config import settings
from ..services.privacy import PresidioPrivatizer, NoOpPrivatizer

logger = logging.getLogger(__name__)


def _init_privatizer():
    """Initialize the appropriate privatizer (Presidio or NoOp)."""
    if not settings.privacy_enabled:
        return NoOpPrivatizer()

    if settings.privacy_anonymizer.lower() == "presidio":
        try:
            return PresidioPrivatizer()
        except ImportError:
            logger.warning("Presidio not available, using NoOp privatizer")
            return NoOpPrivatizer()

    return NoOpPrivatizer()


_privatizer = _init_privatizer()


def privatize_context(text: str) -> str:
    return _privatizer.privatize_context(text)

def add_known_entities(entities: list[str]) -> None:
    if hasattr(_privatizer, "add_known_entities"):
        _privatizer.add_known_entities(entities)

def restore_text(text: str) -> str:
    if hasattr(_privatizer, "restore"):
        return _privatizer.restore(text)
    return text


def build_privacy_debug_payload(original_text: str, anonymized_text: str) -> dict:
    """Return structured privacy debug info for CLI/API diagnostics."""
    mappings = {}
    if hasattr(_privatizer, "get_current_mappings"):
        mappings = _privatizer.get_current_mappings()

    return {
        "privacy_enabled": bool(settings.privacy_enabled),
        "privacy_anonymizer": settings.privacy_anonymizer,
        "privatizer_class": _privatizer.__class__.__name__,
        "pii_detected": anonymized_text != original_text,
        "mappings": mappings,
    }


def log_privacy_debug(original_text: str, anonymized_text: str) -> None:
    logger.debug("Original context length: %s", len(original_text))
    logger.debug("Anonymized context length: %s", len(anonymized_text))
    if settings.privacy_enabled and hasattr(_privatizer, "get_current_mappings"):
        logger.debug("PII mappings: %s", _privatizer.get_current_mappings())
