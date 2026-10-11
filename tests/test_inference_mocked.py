"""Inference tests using a deterministic mocked model (no weights needed)."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import make_image_bytes

from src.vision import inference
from src.vision.material_mapping import PLASTIC


class _DummySpec:
    key = "dummy"
    display_name = "Dummy classifier"
    architecture = "dummy"
    version = "test"
    num_classes = 4
    categories = ("water bottle", "wine bottle", "banana", "stone")
    is_direct_material_classifier = False

    def transform(self, image):
        import torch

        return torch.zeros(3, 224, 224)


@pytest.fixture()
def mock_model(monkeypatch):
    logits = np.array([2.0, 1.0, 0.5, 0.1], dtype=np.float32)

    monkeypatch.setattr(inference, "get_model_spec", lambda key=None: _DummySpec())
    monkeypatch.setattr(inference, "get_device", lambda: "cpu")
    monkeypatch.setattr(inference, "_run_model", lambda image, key: (_DummySpec(), logits))
    return logits


def test_topk_ordering_descending(mock_model):
    result = inference.analyze_image_bytes(make_image_bytes(), with_explanation=False)
    assert result.ok
    scores = [p.raw_score for p in result.predictions]
    assert scores == sorted(scores, reverse=True)
    assert result.top1.class_name == "water bottle"


def test_top1_material_hint_mapped(mock_model):
    result = inference.analyze_image_bytes(make_image_bytes(), with_explanation=False)
    assert result.material_hint is not None
    assert result.material_hint.category_slug == PLASTIC


def test_decode_failure_returns_not_ok():
    result = inference.analyze_image_bytes(b"garbage", with_explanation=False)
    assert result.ok is False
    assert result.error


def test_explanation_failure_falls_back(monkeypatch, mock_model):
    from src.vision.explanations import ExplanationError

    def boom(*_a, **_k):
        raise ExplanationError("gradcam unavailable")

    monkeypatch.setattr(inference, "generate_gradcam", boom)
    result = inference.analyze_image_bytes(make_image_bytes(), with_explanation=True)
    assert result.ok is True
    assert result.explanation is None
    assert "gradcam unavailable" in (result.explanation_error or "")
    assert result.top1 is not None
