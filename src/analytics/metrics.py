"""Real analytics derived from the ``analyses`` table.

Every value is a direct aggregate of stored records. When there are no records,
callers must render an honest empty state and **must not** invent numbers.
Outcome metrics (items diverted, emissions avoided) are intentionally absent
because they are not measured by this system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select

from ..database.initialization import session_scope
from ..database.models import Analysis
from ..vision import uncertainty as u


@dataclass
class AnalyticsSummary:
    total_analyses: int = 0
    confident_count: int = 0
    uncertain_count: int = 0
    low_confidence_count: int = 0
    avg_inference_ms: float = 0.0
    distinct_materials: int = 0
    category_distribution: list[tuple[str, int]] = field(default_factory=list)
    over_time: list[tuple[str, int]] = field(default_factory=list)
    uncertainty_breakdown: list[tuple[str, int]] = field(default_factory=list)
    top_predicted_classes: list[tuple[str, int]] = field(default_factory=list)
    processing_time_series: list[tuple[str, float]] = field(default_factory=list)
    model_distribution: list[tuple[str, int]] = field(default_factory=list)
    abstention_count: int = 0
    unsupported_material_count: int = 0
    feedback: dict = field(default_factory=dict)
    first_record_at: datetime | None = None
    last_record_at: datetime | None = None

    @property
    def has_data(self) -> bool:
        return self.total_analyses > 0

    @property
    def confident_rate(self) -> float:
        if self.total_analyses == 0:
            return 0.0
        return round(self.confident_count / self.total_analyses, 4)

    @property
    def abstention_rate(self) -> float:
        if self.total_analyses == 0:
            return 0.0
        return round(self.abstention_count / self.total_analyses, 4)


def compute_analytics(days: int = 30) -> AnalyticsSummary:
    summary = AnalyticsSummary()
    with session_scope() as session:
        total = int(session.execute(select(func.count(Analysis.id))).scalar_one())
        summary.total_analyses = total
        if total == 0:
            return summary

        def count_where(cond) -> int:
            return int(session.execute(select(func.count(Analysis.id)).where(cond)).scalar_one())

        summary.confident_count = count_where(Analysis.uncertainty_status == u.CONFIDENT)
        summary.uncertain_count = count_where(Analysis.uncertainty_status == u.UNCERTAIN)
        summary.low_confidence_count = count_where(Analysis.uncertainty_status == u.LOW_CONFIDENCE)

        # Abstention = flagged for human review (uncertain or low confidence).
        summary.abstention_count = count_where(
            Analysis.uncertainty_status.in_([u.UNCERTAIN, u.LOW_CONFIDENCE])
        )
        # An object was recognised but no material claim could be made.
        summary.unsupported_material_count = count_where(
            Analysis.material_category.is_(None) & Analysis.predicted_class.is_not(None)
        )

        avg_ms = session.execute(select(func.avg(Analysis.inference_duration_ms))).scalar()
        summary.avg_inference_ms = float(avg_ms) if avg_ms is not None else 0.0

        cat_rows = session.execute(
            select(Analysis.material_category, func.count(Analysis.id))
            .where(Analysis.material_category.is_not(None))
            .group_by(Analysis.material_category)
            .order_by(func.count(Analysis.id).desc())
        ).all()
        summary.category_distribution = [(c or "unknown", n) for c, n in cat_rows]
        summary.distinct_materials = len(summary.category_distribution)

        unc_rows = session.execute(
            select(Analysis.uncertainty_status, func.count(Analysis.id))
            .where(Analysis.uncertainty_status.is_not(None))
            .group_by(Analysis.uncertainty_status)
        ).all()
        summary.uncertainty_breakdown = [(s, n) for s, n in unc_rows]

        cls_rows = session.execute(
            select(Analysis.predicted_class, func.count(Analysis.id))
            .where(Analysis.predicted_class.is_not(None))
            .group_by(Analysis.predicted_class)
            .order_by(func.count(Analysis.id).desc())
            .limit(8)
        ).all()
        summary.top_predicted_classes = [(c, n) for c, n in cls_rows]

        model_rows = session.execute(
            select(Analysis.model_key, func.count(Analysis.id))
            .where(Analysis.model_key.is_not(None))
            .group_by(Analysis.model_key)
            .order_by(func.count(Analysis.id).desc())
        ).all()
        summary.model_distribution = [(m, n) for m, n in model_rows]

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        day_rows = session.execute(
            select(
                func.strftime("%Y-%m-%d", Analysis.created_at),
                func.count(Analysis.id),
            )
            .where(Analysis.created_at >= cutoff)
            .group_by(func.strftime("%Y-%m-%d", Analysis.created_at))
            .order_by(func.strftime("%Y-%m-%d", Analysis.created_at))
        ).all()
        summary.over_time = [(d, n) for d, n in day_rows if d]

        pt_rows = session.execute(
            select(Analysis.created_at, Analysis.inference_duration_ms)
            .where(Analysis.inference_duration_ms.is_not(None))
            .order_by(Analysis.created_at.asc())
            .limit(200)
        ).all()
        summary.processing_time_series = [
            (dt.isoformat(), float(ms)) for dt, ms in pt_rows if dt is not None
        ]

        bounds = session.execute(
            select(func.min(Analysis.created_at), func.max(Analysis.created_at))
        ).one()
        summary.first_record_at, summary.last_record_at = bounds[0], bounds[1]

    from ..database import repository

    summary.feedback = repository.feedback_summary()
    return summary
