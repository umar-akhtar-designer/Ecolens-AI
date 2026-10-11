"""Tests for the analysis orchestration service."""

from __future__ import annotations

import numpy as np
from conftest import make_image_bytes

from src.database import repository
from src.services import analysis_service
from src.vision import uncertainty as u
from src.vision.inference import AnalysisResult, PredictionCandidate
from src.vision.material_mapping import map_label_to_material


def _fake_analysis_result(label="water bottle", score=0.9) -> AnalysisResult:
    cands = [PredictionCandidate(1, 0, label, score, "plastic")]
    return AnalysisResult(
        ok=True,
        predictions=cands,
        top1=cands[0],
        uncertainty=u.analyse(np.array([score, 0.05, 0.05]), supported_domain=True),
        material_hint=map_label_to_material(label),
        image_width=64,
        image_height=64,
        image_sha256="a" * 64,
        model_key="mobilenet_v3_large",
        model_name="MobileNetV3-Large",
        model_version="v1",
        device="cpu",
        inference_duration_ms=12.0,
        is_direct_material_classifier=False,
    )


def _patch(monkeypatch, result=None):
    result = result or _fake_analysis_result()
    monkeypatch.setattr(analysis_service, "analyze_image_bytes", lambda *a, **k: result)
    return result


def test_run_analysis_persists(fresh_db, monkeypatch):
    _patch(monkeypatch)
    bundle = analysis_service.run_analysis(make_image_bytes(), "x.png", persist=True)
    assert bundle.ok
    assert bundle.record is not None
    assert bundle.decision is not None
    # Object-label inference cannot be marked verified.
    assert bundle.decision.material_verified is False
    fetched = repository.get_by_public_id(bundle.record.public_id)
    assert fetched is not None
    assert fetched.predicted_class == "water bottle"


def test_run_analysis_no_persist(fresh_db, monkeypatch):
    _patch(monkeypatch)
    bundle = analysis_service.run_analysis(make_image_bytes(), "x.png", persist=False)
    assert bundle.ok
    assert bundle.record is None

    _records, total = repository.list_analyses()
    assert total == 0


def test_run_analysis_invalid_upload(fresh_db, monkeypatch):
    _patch(monkeypatch)
    bundle = analysis_service.run_analysis(b"not an image", "bad.txt", persist=False)
    assert bundle.ok is False
    assert bundle.error


def test_run_analysis_failure_is_captured(fresh_db, monkeypatch):
    failure = AnalysisResult(ok=False, error="boom")
    monkeypatch.setattr(analysis_service, "analyze_image_bytes", lambda *a, **k: failure)
    bundle = analysis_service.run_analysis(make_image_bytes(), "x.png", persist=True)
    assert bundle.ok is False
    assert bundle.error == "boom"
    assert bundle.record is None


def test_run_batch_success(fresh_db, monkeypatch):
    _patch(monkeypatch)
    files = [("a.png", make_image_bytes(), "image/png"), ("b.png", make_image_bytes(), "image/png")]
    outcome = analysis_service.run_batch(files, persist=True)
    assert len(outcome.items) == 2
    assert outcome.succeeded == 2
    assert outcome.failed == 0


def test_run_batch_isolates_failures(fresh_db, monkeypatch):
    ok_result = _fake_analysis_result()
    calls = {"n": 0}

    def fake(*a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            return AnalysisResult(ok=False, error="second failed")
        return ok_result

    monkeypatch.setattr(analysis_service, "analyze_image_bytes", fake)
    files = [("a.png", make_image_bytes(), "image/png") for _ in range(3)]
    outcome = analysis_service.run_batch(files, persist=False)
    assert outcome.succeeded == 2
    assert outcome.failed == 1


def test_run_batch_skips_oversize(fresh_db, monkeypatch):
    _patch(monkeypatch)
    from src.config import SETTINGS

    big = b"\x00" * (SETTINGS.max_upload_bytes + 1)
    files = [("big.png", big, "image/png"), ("ok.png", make_image_bytes(), "image/png")]
    outcome = analysis_service.run_batch(files, persist=False)
    assert len(outcome.skipped) == 1
    assert outcome.items[0].filename == "ok.png"


def test_run_batch_limit(fresh_db, monkeypatch):
    _patch(monkeypatch)
    files = [("a.png", make_image_bytes(), "image/png") for _ in range(3)]
    outcome = analysis_service.run_batch(files, persist=False, max_files=1)
    assert outcome.limit_reached is True
    assert len(outcome.items) == 1
