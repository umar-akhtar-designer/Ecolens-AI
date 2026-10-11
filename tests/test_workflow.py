"""End-to-end workflow: upload -> validate -> infer -> guide -> persist -> history
-> analytics. Uses the real model when available."""

from __future__ import annotations

from io import BytesIO

import pytest
from conftest import make_image_bytes

from src.analytics.metrics import compute_analytics
from src.database import repository
from src.guidance.engine import get_engine
from src.security.uploads import save_explanation, save_image, validate_and_decode
from src.vision.inference import analyze_image_bytes
from src.vision.model_registry import ModelLoadError, get_model_spec


@pytest.fixture(scope="module")
def _has_model():
    try:
        get_model_spec()
    except ModelLoadError as exc:  # pragma: no cover
        pytest.skip(f"Model weights unavailable: {exc}")


def test_full_workflow(fresh_db, _has_model):
    data = make_image_bytes(256, 256, seed=7)

    decoded = validate_and_decode(data, "sample.png", "image/png")
    assert decoded.width == 256

    result = analyze_image_bytes(data, filename="sample.png", with_explanation=True)
    assert result.ok, result.error

    guidance = get_engine().build_guidance(result)
    assert guidance.headline

    _, rel_image = save_image(data, "sample.png")
    explanation_rel = None
    if result.explanation is not None:
        buf = BytesIO()
        result.explanation.overlay.save(buf, format="PNG")
        _, explanation_rel = save_explanation(buf.getvalue())

    record = repository.save_analysis(
        result,
        guidance,
        original_filename="sample.png",
        stored_image_path=rel_image,
        explanation_path=explanation_rel,
    )
    assert record.image_exists()

    fetched = repository.get_by_public_id(record.public_id)
    assert fetched is not None
    assert fetched.predicted_class == result.top1.class_name

    listed, total = repository.list_analyses()
    assert total >= 1
    assert any(r.public_id == record.public_id for r in listed)

    summary = compute_analytics()
    assert summary.total_analyses >= 1
    assert summary.has_data is True

    assert repository.delete_analysis(record.public_id) is True
    assert repository.get_by_public_id(record.public_id) is None


def test_service_end_to_end(fresh_db, _has_model):
    """The orchestration service drives the same real pipeline end-to-end."""
    from src.services.analysis_service import run_analysis

    data = make_image_bytes(256, 256, seed=11)
    bundle = run_analysis(data, "service.png", persist=True)

    assert bundle.ok, bundle.error
    assert bundle.record is not None
    assert bundle.decision is not None
    assert bundle.guidance is not None
    assert len(bundle.decision.layers()) == 3

    fetched = repository.get_by_public_id(bundle.record.public_id)
    assert fetched is not None
    assert fetched.predicted_class == bundle.result.top1.class_name

    repository.add_feedback(bundle.record.public_id, verdict="correct")
    assert repository.feedback_summary().get("total") == 1
