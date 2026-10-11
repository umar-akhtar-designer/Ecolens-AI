"""Tests for the knowledge base and guidance engine."""

from __future__ import annotations

from types import SimpleNamespace

from src.guidance.engine import GuidanceEngine
from src.vision import uncertainty as u
from src.vision.inference import PredictionCandidate
from src.vision.material_mapping import PLASTIC, MaterialHint


def _engine() -> GuidanceEngine:
    return GuidanceEngine()


def test_all_categories_loaded():
    engine = _engine()
    slugs = [c.category_slug for c in engine.all_categories()]
    assert "plastic" in slugs
    assert "e_waste" in slugs
    assert "mixed_uncertain" in slugs


def test_get_known_category():
    assert _engine().get("glass") is not None
    assert _engine().get("does-not-exist") is None


def test_search_matches_text():
    engine = _engine()
    results = engine.search("bottle")
    assert results
    assert all("bottle" in r.searchable_text() for r in results)


def test_search_empty_returns_all():
    engine = _engine()
    assert len(engine.search("")) == len(engine.all_categories())


def _analysis(status, hint, label="water bottle"):
    report = SimpleNamespace(status=status)
    return SimpleNamespace(
        uncertainty=report,
        top1=PredictionCandidate(1, 0, label, 0.8, hint.category_slug if hint else None),
        material_hint=hint,
        predictions=[PredictionCandidate(1, 0, label, 0.8, hint.category_slug if hint else None)],
        hazard_note=None,
    )


def test_confident_prediction_returns_material_guidance():
    engine = _engine()
    hint = MaterialHint(PLASTIC, "test", "water bottle")
    result = engine.build_guidance(_analysis(u.CONFIDENT, hint))
    assert result.confidence_sufficient is True
    assert result.primary.category_slug == "plastic"


def test_low_confidence_returns_general_guidance():
    engine = _engine()
    hint = MaterialHint(PLASTIC, "test", "water bottle")
    result = engine.build_guidance(_analysis(u.LOW_CONFIDENCE, hint))
    assert result.confidence_sufficient is False
    assert result.primary.category_slug == "mixed_uncertain"


def test_no_hint_returns_general_guidance():
    engine = _engine()
    result = engine.build_guidance(_analysis(u.CONFIDENT, None, label="tench"))
    assert result.confidence_sufficient is False
    assert "not map" in result.reason


def test_hazard_note_forces_warning_severity():
    engine = _engine()
    hint = MaterialHint("e_waste", "test", "iPod")
    analysis = _analysis(u.CONFIDENT, hint, label="iPod")
    analysis.hazard_note = "battery caution"
    result = engine.build_guidance(analysis)
    assert result.severity == "warning"
    assert result.safety_notes
