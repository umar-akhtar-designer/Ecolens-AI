"""CSV export helpers.

Exports are derived purely from stored records or in-memory analysis results.
No columns are fabricated; scores are exported exactly as stored (and labelled
as uncalibrated softmax scores).
"""

from __future__ import annotations

import csv
import io


def analyses_to_csv(records) -> str:
    """Serialise a list of :class:`AnalysisDTO` to CSV text."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "public_id",
            "created_at",
            "original_filename",
            "predicted_class",
            "score_type",
            "confidence_score",
            "uncertainty_status",
            "material_category",
            "model_key",
            "inference_duration_ms",
            "explanation_available",
            "feedback_verdicts",
        ]
    )
    for r in records:
        writer.writerow(
            [
                r.public_id,
                r.created_at.isoformat() if r.created_at else "",
                r.original_filename or "",
                r.predicted_class or "",
                r.score_type or "",
                "" if r.confidence_score is None else f"{r.confidence_score:.6f}",
                r.uncertainty_status or "",
                r.material_category or "",
                r.model_key or "",
                "" if r.inference_duration_ms is None else f"{r.inference_duration_ms:.2f}",
                str(r.explanation_available),
                ";".join(f.verdict for f in r.feedback),
            ]
        )
    return buffer.getvalue()


def batch_outcome_to_csv(outcome) -> str:
    """Serialise a :class:`BatchOutcome` to CSV text (one row per file)."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "filename",
            "status",
            "predicted_class",
            "confidence_score",
            "uncertainty_status",
            "material_category",
            "public_id",
            "error",
        ]
    )
    for item in outcome.items:
        result = item.result
        writer.writerow(
            [
                item.filename,
                "ok" if item.ok else "failed",
                (result.top1.class_name if result and result.top1 else ""),
                ("" if not result or not result.top1 else f"{result.top1.raw_score:.6f}"),
                (result.uncertainty.status if result and result.uncertainty else ""),
                (result.material_hint.category_slug if result and result.material_hint else ""),
                item.record.public_id if item.record else "",
                item.error or "",
            ]
        )
    for filename, reason in outcome.skipped:
        writer.writerow([filename, "skipped", "", "", "", "", "", reason])
    return buffer.getvalue()
