"""Tests for persistence, filtering and deletion."""

from __future__ import annotations

from types import SimpleNamespace

from conftest import make_image_bytes

from src.database import repository
from src.database.initialization import session_scope
from src.database.models import Analysis
from src.security.uploads import save_image


def _fake_result(label="water bottle", score=0.9, status="confident", material="plastic"):
    from src.vision.inference import PredictionCandidate

    cand = PredictionCandidate(1, 0, label, score, material)
    return SimpleNamespace(
        image_sha256="a" * 64,
        image_width=100,
        image_height=100,
        model_key="mobilenet_v3_large",
        model_name="MobileNetV3-Large",
        model_version="v1",
        score_type="softmax_score_uncalibrated",
        top1=cand,
        uncertainty=SimpleNamespace(status=status),
        material_hint=SimpleNamespace(category_slug=material) if material else None,
        predictions=[cand],
        inference_duration_ms=42.0,
        explanation=None,
        error=None,
    )


def _fake_guidance(sufficient=True):
    return SimpleNamespace(
        headline="Possible material: Plastic",
        confidence_sufficient=sufficient,
        primary=SimpleNamespace(
            category_slug="plastic",
            reuse_guidance="reuse",
            disposal_guidance="dispose",
            source_url="http://example.com",
            geographic_scope="General",
        ),
        safety_notes=[],
    )


def test_save_and_fetch(fresh_db):
    _, rel = save_image(make_image_bytes(), "x.png")
    dto = repository.save_analysis(
        _fake_result(),
        _fake_guidance(),
        original_filename="x.png",
        stored_image_path=rel,
    )
    assert dto.public_id.startswith("an_")
    fetched = repository.get_by_public_id(dto.public_id)
    assert fetched is not None
    assert fetched.predicted_class == "water bottle"
    assert len(fetched.predictions) == 1
    assert fetched.guidance is not None


def test_get_invalid_id_returns_none(fresh_db):
    assert repository.get_by_public_id("an_doesnotexist") is None


def test_list_and_filter(fresh_db):
    for label, mat in [("water bottle", "plastic"), ("wine bottle", "glass")]:
        repository.save_analysis(
            _fake_result(label=label, material=mat),
            _fake_guidance(),
            original_filename="f.png",
            stored_image_path=None,
        )
    _all_records, total = repository.list_analyses()
    assert total == 2
    filtered, ftotal = repository.list_analyses(material_category="glass")
    assert ftotal == 1
    assert filtered[0].material_category == "glass"
    _searched, stotal = repository.list_analyses(search="water")
    assert stotal == 1


def test_sort_orders(fresh_db):
    repository.save_analysis(
        _fake_result(label="a"), _fake_guidance(), original_filename=None, stored_image_path=None
    )
    repository.save_analysis(
        _fake_result(label="b"), _fake_guidance(), original_filename=None, stored_image_path=None
    )
    newest, _ = repository.list_analyses(sort="newest")
    oldest, _ = repository.list_analyses(sort="oldest")
    assert newest[0].id != oldest[0].id or len(newest) == 1


def test_delete_analysis_cascades(fresh_db):
    dto = repository.save_analysis(
        _fake_result(), _fake_guidance(), original_filename=None, stored_image_path=None
    )
    assert repository.delete_analysis(dto.public_id) is True
    assert repository.get_by_public_id(dto.public_id) is None
    with session_scope() as session:
        assert session.query(Analysis).count() == 0


def test_delete_missing_returns_false(fresh_db):
    assert repository.delete_analysis("an_missing") is False


def test_missing_image_file_reported(fresh_db):
    dto = repository.save_analysis(
        _fake_result(),
        _fake_guidance(),
        original_filename=None,
        stored_image_path="uploads/1970/01/01/gone.png",
    )
    assert dto.image_exists() is False


def test_sort_by_score(fresh_db):
    repository.save_analysis(
        _fake_result(label="low", score=0.2),
        _fake_guidance(),
        original_filename=None,
        stored_image_path=None,
    )
    repository.save_analysis(
        _fake_result(label="high", score=0.95),
        _fake_guidance(),
        original_filename=None,
        stored_image_path=None,
    )
    high, _ = repository.list_analyses(sort="score_high")
    low, _ = repository.list_analyses(sort="score_low")
    assert high[0].predicted_class == "high"
    assert low[0].predicted_class == "low"


def test_model_filter_and_distinct_keys(fresh_db):
    repository.save_analysis(
        _fake_result(), _fake_guidance(), original_filename=None, stored_image_path=None
    )
    keys = repository.distinct_model_keys()
    assert "mobilenet_v3_large" in keys
    _filtered, total = repository.list_analyses(model_key="mobilenet_v3_large")
    assert total == 1
    _none, none_total = repository.list_analyses(model_key="does_not_exist")
    assert none_total == 0
