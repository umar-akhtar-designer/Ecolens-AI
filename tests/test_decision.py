"""Tests for the decision-support layer."""

from __future__ import annotations

from types import SimpleNamespace

from test_repository import _fake_guidance, _fake_result

from src.database import repository
from src.guidance.decision import build_decision, decision_from_record


def _analysis(*, label="water bottle", hint="plastic", status="confident", hazard=None):
    top = SimpleNamespace(class_name=label, raw_score=0.9)
    uncertainty = SimpleNamespace(
        status=status,
        explanation="stub",
        abstain=(status in ("low_confidence", "unknown")),
        abstain_reason="stub reason",
    )
    return SimpleNamespace(
        top1=top,
        uncertainty=uncertainty,
        material_hint=(
            SimpleNamespace(
                category_slug=hint,
                matched_label=label,
                reason="stub",
                evidence="evidence",
                is_direct_material_classifier=False,
            )
            if hint
            else None
        ),
        hazard_note=hazard,
    )


def test_confident_material_is_verified():
    decision = build_decision(_analysis(), _fake_guidance(sufficient=True))
    # With object-label inference, we never mark as verified.
    assert decision.material_verified is False
    assert decision.material_category == "plastic"
    assert "plastic" in decision.next_action
    assert len(decision.layers()) == 3


def test_low_confidence_not_verified():
    decision = build_decision(_analysis(status="low_confidence"), _fake_guidance(sufficient=False))
    assert decision.material_verified is False
    assert decision.material_category is None


def test_hazard_takes_priority():
    decision = build_decision(
        _analysis(hazard="Contains a battery."), _fake_guidance(sufficient=True)
    )
    assert decision.next_action.startswith("Safety first")


def test_no_hint_is_honest():
    decision = build_decision(_analysis(hint=None), _fake_guidance(sufficient=False))
    assert decision.material_verified is False
    assert "No material category" in decision.material_evidence


def test_decision_from_record(fresh_db):
    dto = repository.save_analysis(
        _fake_result(),
        _fake_guidance(),
        original_filename="x.png",
        stored_image_path=None,
    )
    decision = decision_from_record(dto)
    assert decision.material_category == "plastic"
    assert len(decision.layers()) == 3
