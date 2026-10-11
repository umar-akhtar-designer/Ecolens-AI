"""Combines model output with local material knowledge to produce guidance.

The engine is deliberately conservative: it only surfaces material-specific
guidance when the model's material hint is reliable enough. Otherwise it
explains the ambiguity and offers next steps.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from ..config import SETTINGS
from ..logging_config import get_logger

logger = get_logger("guidance")


class KnowledgeBaseError(RuntimeError):
    pass


@dataclass(frozen=True)
class GuidanceItem:
    category_slug: str
    display_name: str
    description: str
    common_examples: list[str]
    reuse_guidance: str
    disposal_guidance: str
    safety_notes: str
    contamination_notes: str
    recyclability_conditions: str
    source_url: str
    geographic_scope: str
    reviewed_at: str

    def searchable_text(self) -> str:
        parts = [
            self.display_name,
            self.description,
            " ".join(self.common_examples),
            self.reuse_guidance,
            self.disposal_guidance,
            self.safety_notes,
        ]
        return " ".join(parts).lower()


@dataclass
class GuidanceResult:
    """The guidance decisions rendered for a single analysis."""

    confidence_sufficient: bool
    headline: str
    primary: GuidanceItem | None = None
    alternatives: list[GuidanceItem] = field(default_factory=list)
    reason: str = ""
    safety_notes: list[str] = field(default_factory=list)
    severity: str = "info"  # info | caution | warning

    def to_payload(self) -> dict[str, Any]:
        return {
            "confidence_sufficient": self.confidence_sufficient,
            "headline": self.headline,
            "reason": self.reason,
            "severity": self.severity,
            "safety_notes": self.safety_notes,
            "primary": self.primary.category_slug if self.primary else None,
            "primary_display": self.primary.display_name if self.primary else None,
            "primary_disposal": self.primary.disposal_guidance if self.primary else None,
            "primary_reuse": self.primary.reuse_guidance if self.primary else None,
            "primary_source": self.primary.source_url if self.primary else None,
            "alternatives": [a.category_slug for a in self.alternatives],
        }


class GuidanceEngine:
    def __init__(self, knowledge_path=None):
        self.knowledge_path = knowledge_path or SETTINGS.knowledge_path
        self._items: dict[str, GuidanceItem] = {}
        self._order: list[str] = []
        self._metadata: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        try:
            raw = json.loads(self.knowledge_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise KnowledgeBaseError(f"Knowledge base not found at {self.knowledge_path}") from exc
        except json.JSONDecodeError as exc:
            raise KnowledgeBaseError(f"Knowledge base is not valid JSON: {exc}") from exc

        self._metadata = raw.get("metadata", {})
        for entry in raw.get("categories", []):
            slug = entry["category_slug"]
            self._items[slug] = GuidanceItem(
                category_slug=slug,
                display_name=entry.get("display_name", slug),
                description=entry.get("description", ""),
                common_examples=list(entry.get("common_examples", [])),
                reuse_guidance=entry.get("reuse_guidance", ""),
                disposal_guidance=entry.get("disposal_guidance", ""),
                safety_notes=entry.get("safety_notes", ""),
                contamination_notes=entry.get("contamination_notes", ""),
                recyclability_conditions=entry.get("recyclability_conditions", ""),
                source_url=entry.get("source_url", ""),
                geographic_scope=entry.get("geographic_scope", ""),
                reviewed_at=entry.get("reviewed_at", ""),
            )
            self._order.append(slug)

    # -- Access ------------------------------------------------------------
    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def all_categories(self) -> list[GuidanceItem]:
        return [self._items[s] for s in self._order]

    def get(self, slug: str) -> GuidanceItem | None:
        return self._items.get(slug)

    def get_or_mixed(self, slug: str | None) -> GuidanceItem:
        if slug and slug in self._items:
            return self._items[slug]
        return self._items["mixed_uncertain"]

    def search(self, query: str) -> list[GuidanceItem]:
        q = (query or "").strip().lower()
        if not q:
            return self.all_categories()
        return [item for item in self.all_categories() if q in item.searchable_text()]

    # -- Guidance synthesis ------------------------------------------------
    def build_guidance(self, analysis) -> GuidanceResult:
        """Turn an :class:`AnalysisResult` into a :class:`GuidanceResult`.

        Never claims certainty the model did not provide.
        """
        from ..vision import uncertainty as u

        status = analysis.uncertainty.status if analysis.uncertainty else u.UNKNOWN
        top = analysis.top1
        hint = analysis.material_hint
        label = top.class_name if top else "unknown object"

        safety: list[str] = []
        if analysis.hazard_note:
            safety.append(analysis.hazard_note)

        if status in (u.LOW_CONFIDENCE, u.UNKNOWN) or hint is None:
            # Not enough to make a material claim.
            mixed = self.get_or_mixed("mixed_uncertain")
            if status in (u.LOW_CONFIDENCE, u.UNKNOWN):
                reason = "The model was not confident enough to support material-specific guidance."
            else:
                reason = (
                    f"The model recognised an object ('{label}') but this does not "
                    "map to a reliable material category."
                )
            return GuidanceResult(
                confidence_sufficient=False,
                headline="General guidance only",
                primary=mixed,
                reason=reason,
                safety_notes=safety,
                severity="caution" if safety else "info",
            )

        primary = self.get_or_mixed(hint.category_slug)
        alternatives: list[GuidanceItem] = []
        for cand in analysis.predictions[1:]:
            if cand.material_slug and cand.material_slug != hint.category_slug:
                alt = self.get(cand.material_slug)
                if alt and alt not in alternatives:
                    alternatives.append(alt)
            if len(alternatives) >= 3:
                break

        severity = "info"
        if safety or primary.category_slug in ("e_waste", "battery"):
            severity = "warning" if primary.category_slug in ("e_waste", "battery") else "caution"

        return GuidanceResult(
            confidence_sufficient=True,
            headline=f"Possible material: {primary.display_name}",
            primary=primary,
            alternatives=alternatives,
            reason=hint.reason + " Material is inferred from the recognised object, not measured.",
            safety_notes=safety,
            severity=severity,
        )


@lru_cache(maxsize=1)
def get_engine() -> GuidanceEngine:
    return GuidanceEngine()
