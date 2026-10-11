"""Tests for additive user feedback."""

from __future__ import annotations

import pytest
from test_repository import _fake_guidance, _fake_result

from src.database import repository


def _save():
    return repository.save_analysis(
        _fake_result(), _fake_guidance(), original_filename="x.png", stored_image_path=None
    )


def test_add_and_read_feedback(fresh_db):
    dto = _save()
    added = repository.add_feedback(
        dto.public_id, verdict="incorrect", material_correction="glass", note="it was glass"
    )
    assert added is not None
    fetched = repository.get_by_public_id(dto.public_id)
    assert len(fetched.feedback) == 1
    assert fetched.feedback[0].verdict == "incorrect"
    assert fetched.feedback[0].material_correction == "glass"


def test_feedback_does_not_mutate_prediction(fresh_db):
    dto = _save()
    repository.add_feedback(dto.public_id, verdict="incorrect", material_correction="metal")
    fetched = repository.get_by_public_id(dto.public_id)
    assert fetched.predicted_class == "water bottle"
    assert fetched.material_category == "plastic"


def test_invalid_verdict_rejected(fresh_db):
    dto = _save()
    with pytest.raises(ValueError):
        repository.add_feedback(dto.public_id, verdict="banana")


def test_feedback_missing_analysis(fresh_db):
    assert repository.add_feedback("an_missing", verdict="correct") is None


def test_feedback_summary(fresh_db):
    dto = _save()
    repository.add_feedback(dto.public_id, verdict="correct")
    repository.add_feedback(dto.public_id, verdict="incorrect")
    summary = repository.feedback_summary()
    assert summary["total"] == 1
    assert summary["incorrect"] == 1
    assert summary["correct"] == 0
    assert summary["incorrect_rate"] == 1.0


def test_feedback_summary_empty(fresh_db):
    assert repository.feedback_summary() == {}


def test_feedback_cascades_on_delete(fresh_db):
    dto = _save()
    repository.add_feedback(dto.public_id, verdict="correct")
    repository.delete_analysis(dto.public_id)
    assert repository.feedback_summary() == {}
