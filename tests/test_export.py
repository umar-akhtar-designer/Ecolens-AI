"""Tests for CSV export helpers."""

from __future__ import annotations

import csv
import io

from test_repository import _fake_guidance, _fake_result

from src.analytics.export import analyses_to_csv, batch_outcome_to_csv
from src.database import repository
from src.services.analysis_service import AnalysisBundle, BatchOutcome
from src.vision import uncertainty as u
from src.vision.inference import AnalysisResult, PredictionCandidate
from src.vision.material_mapping import map_label_to_material


def _rows(csv_text):
    return list(csv.reader(io.StringIO(csv_text)))


def test_analyses_csv(fresh_db):
    dto = repository.save_analysis(
        _fake_result(), _fake_guidance(), original_filename="x.png", stored_image_path=None
    )
    records, _ = repository.list_analyses()
    text = analyses_to_csv(records)
    rows = _rows(text)
    assert rows[0][0] == "public_id"
    assert rows[1][0] == dto.public_id
    assert "water bottle" in text


def test_batch_csv_includes_skipped():
    cands = [PredictionCandidate(1, 0, "water bottle", 0.9, "plastic")]
    result = AnalysisResult(
        ok=True,
        predictions=cands,
        top1=cands[0],
        uncertainty=u.analyse([0.9, 0.05, 0.05]),
        material_hint=map_label_to_material("water bottle"),
    )
    outcome = BatchOutcome(
        items=[AnalysisBundle(filename="a.png", ok=True, result=result)],
        skipped=[("big.png", "too big")],
    )
    rows = _rows(batch_outcome_to_csv(outcome))
    assert rows[0][0] == "filename"
    statuses = {r[1] for r in rows[1:]}
    assert "ok" in statuses
    assert "skipped" in statuses
