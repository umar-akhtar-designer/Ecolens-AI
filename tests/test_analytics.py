"""Tests for analytics aggregation from real records."""

from __future__ import annotations

from types import SimpleNamespace

from src.analytics.metrics import compute_analytics
from src.database import repository
from src.vision.inference import PredictionCandidate


def _result(label, material, status):
    cand = PredictionCandidate(1, 0, label, 0.9, material)
    return SimpleNamespace(
        image_sha256="b" * 64,
        image_width=10,
        image_height=10,
        model_key="m",
        model_name="M",
        model_version="v",
        score_type="softmax_score_uncalibrated",
        top1=cand,
        uncertainty=SimpleNamespace(status=status),
        material_hint=SimpleNamespace(category_slug=material) if material else None,
        predictions=[cand],
        inference_duration_ms=100.0,
        explanation=None,
        error=None,
    )


def _guidance(material):
    return SimpleNamespace(
        headline="h",
        confidence_sufficient=True,
        primary=SimpleNamespace(
            category_slug=material,
            reuse_guidance="r",
            disposal_guidance="d",
            source_url="u",
            geographic_scope="G",
        ),
        safety_notes=[],
    )


def test_empty_database_reports_no_data(fresh_db):
    summary = compute_analytics()
    assert summary.total_analyses == 0
    assert summary.has_data is False
    assert summary.category_distribution == []


def test_aggregates_match_seeded_rows(fresh_db):
    repository.save_analysis(
        _result("water bottle", "plastic", "confident"),
        _guidance("plastic"),
        original_filename=None,
        stored_image_path=None,
    )
    repository.save_analysis(
        _result("water bottle", "plastic", "confident"),
        _guidance("plastic"),
        original_filename=None,
        stored_image_path=None,
    )
    repository.save_analysis(
        _result("wine bottle", "glass", "uncertain"),
        _guidance("glass"),
        original_filename=None,
        stored_image_path=None,
    )

    summary = compute_analytics()
    assert summary.total_analyses == 3
    assert summary.confident_count == 2
    assert summary.uncertain_count == 1
    assert summary.distinct_materials == 2
    dist = dict(summary.category_distribution)
    assert dist["plastic"] == 2
    assert dist["glass"] == 1
    assert summary.avg_inference_ms == 100.0
    assert summary.over_time  # at least one day bucket
