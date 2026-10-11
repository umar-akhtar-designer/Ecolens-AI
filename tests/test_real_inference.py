"""Tests that exercise the *real* pretrained model (requires cached weights).

These are separated so mocked logic tests stay fast. If weights are genuinely
unavailable offline, the test is skipped rather than failing the suite.
"""

from __future__ import annotations

import pytest
from conftest import make_image_bytes

from src.vision.inference import analyze_image_bytes
from src.vision.model_registry import ModelLoadError, get_model_spec


@pytest.fixture(scope="module")
def _spec_or_skip():
    try:
        return get_model_spec()
    except ModelLoadError as exc:  # pragma: no cover - offline
        pytest.skip(f"Model weights unavailable: {exc}")


def test_real_model_produces_structured_output(_spec_or_skip):
    result = analyze_image_bytes(make_image_bytes(seed=1), with_explanation=False)
    assert result.ok, result.error
    assert result.top1 is not None
    assert result.top1.class_name
    assert result.model_key == _spec_or_skip.key
    assert result.score_type == "softmax_score_uncalibrated"
    assert 0.0 <= result.top1.raw_score <= 1.0
    assert result.inference_duration_ms >= 0


def test_real_topk_ordering_and_bounds(_spec_or_skip):
    result = analyze_image_bytes(make_image_bytes(seed=2), with_explanation=False)
    assert result.ok
    scores = [p.raw_score for p in result.predictions]
    assert scores == sorted(scores, reverse=True)
    assert len(result.predictions) <= 5


def test_real_gradcam_explanation(_spec_or_skip):
    result = analyze_image_bytes(make_image_bytes(seed=3), with_explanation=True)
    assert result.ok
    assert result.explanation is not None
    assert result.explanation.overlay is not None
    assert result.explanation.method == "Grad-CAM"
