"""Repository layer: translate between the vision/guidance domains and the DB.

Returns plain dataclasses (DTOs) rather than ORM objects so Streamlit pages
never trip over detached lazy-loaded relationships.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from sqlalchemy import delete, func, select

from ..logging_config import get_logger
from ..security.uploads import delete_stored_file, resolve_storage_path
from .initialization import session_scope
from .models import (
    Analysis,
    DecisionSnapshot,
    Feedback,
    GuidanceSnapshot,
    PredictionCandidateRecord,
    utcnow,
)

logger = get_logger("database.repository")

MAX_NOTE_LENGTH = 1000
MAX_LABEL_LENGTH = 128


def _utc_naive(value: datetime | None) -> datetime | None:
    """Normalise a datetime to naive UTC to match how SQLite stores them."""
    if value is None:
        return None
    if value.tzinfo is not None:
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def _clean_text(value: str | None, max_length: int) -> str | None:
    """Trim, collapse whitespace and clamp length. Returns None when empty."""
    if value is None:
        return None
    collapsed = re.sub(r"\s+", " ", str(value)).strip()
    if not collapsed:
        return None
    return collapsed[:max_length]


@dataclass
class PredictionDTO:
    rank: int
    class_name: str
    raw_score: float
    calibrated_probability: float | None = None
    material_slug: str | None = None


@dataclass
class GuidanceDTO:
    guidance_category: str | None
    headline: str | None
    confidence_sufficient: bool
    reuse_guidance: str | None
    disposal_guidance: str | None
    safety_notes: str | None
    source_reference: str | None
    geographic_scope: str | None


@dataclass
class DecisionDTO:
    observation: str | None
    material_evidence: str | None
    material_category: str | None
    material_verified: bool
    next_action: str | None
    confidence_note: str | None
    caveats: list[str] = field(default_factory=list)


@dataclass
class FeedbackDTO:
    verdict: str
    material_correction: str | None
    corrected_label: str | None
    note: str | None
    source: str
    created_at: datetime
    updated_at: datetime | None = None

    @property
    def was_edited(self) -> bool:
        if not self.updated_at or not self.created_at:
            return False
        return abs((self.updated_at - self.created_at).total_seconds()) > 1.0


@dataclass
class AnalysisDTO:
    id: int
    public_id: str
    created_at: datetime
    original_filename: str | None
    stored_image_path: str | None
    image_sha256: str | None
    image_width: int | None
    image_height: int | None
    model_key: str | None
    model_name: str | None
    model_version: str | None
    score_type: str | None
    predicted_class: str | None
    confidence_score: float | None
    uncertainty_status: str | None
    material_category: str | None
    inference_duration_ms: float | None
    explanation_available: bool
    explanation_path: str | None
    explanation_method: str | None
    error_status: str | None
    predictions: list[PredictionDTO] = field(default_factory=list)
    guidance: GuidanceDTO | None = None
    decision: DecisionDTO | None = None
    feedback: list[FeedbackDTO] = field(default_factory=list)

    @property
    def user_feedback(self) -> FeedbackDTO | None:
        for item in self.feedback:
            if item.source == "user":
                return item
        return None

    def image_exists(self) -> bool:
        if not self.stored_image_path:
            return False
        try:
            return resolve_storage_path(self.stored_image_path).exists()
        except Exception:
            return False

    def explanation_exists(self) -> bool:
        if not self.explanation_path:
            return False
        try:
            return resolve_storage_path(self.explanation_path).exists()
        except Exception:
            return False


def _to_dto(analysis: Analysis) -> AnalysisDTO:
    guidance = analysis.guidance[0] if analysis.guidance else None
    decision = analysis.decision
    return AnalysisDTO(
        id=analysis.id,
        public_id=analysis.generated_public_id,
        created_at=analysis.created_at,
        original_filename=analysis.original_filename,
        stored_image_path=analysis.stored_image_path,
        image_sha256=analysis.image_sha256,
        image_width=analysis.image_width,
        image_height=analysis.image_height,
        model_key=analysis.model_key,
        model_name=analysis.model_name,
        model_version=analysis.model_version,
        score_type=analysis.score_type,
        predicted_class=analysis.predicted_class,
        confidence_score=analysis.confidence_score,
        uncertainty_status=analysis.uncertainty_status,
        material_category=analysis.material_category,
        inference_duration_ms=analysis.inference_duration_ms,
        explanation_available=analysis.explanation_available,
        explanation_path=analysis.explanation_path,
        explanation_method=analysis.explanation_method,
        error_status=analysis.error_status,
        predictions=[
            PredictionDTO(
                rank=p.rank,
                class_name=p.class_name,
                raw_score=p.raw_score,
                calibrated_probability=p.calibrated_probability,
                material_slug=p.material_slug,
            )
            for p in analysis.predictions
        ],
        guidance=(
            GuidanceDTO(
                guidance_category=guidance.guidance_category,
                headline=guidance.headline,
                confidence_sufficient=guidance.confidence_sufficient,
                reuse_guidance=guidance.reuse_guidance,
                disposal_guidance=guidance.disposal_guidance,
                safety_notes=guidance.safety_notes,
                source_reference=guidance.source_reference,
                geographic_scope=guidance.geographic_scope,
            )
            if guidance
            else None
        ),
        feedback=[
            FeedbackDTO(
                verdict=f.verdict,
                material_correction=f.material_correction,
                corrected_label=f.corrected_label,
                note=f.note,
                source=f.source,
                created_at=f.created_at,
                updated_at=getattr(f, "updated_at", None),
            )
            for f in analysis.feedback
        ],
        decision=(
            DecisionDTO(
                observation=decision.observation,
                material_evidence=decision.material_evidence,
                material_category=decision.material_category,
                material_verified=bool(decision.material_verified),
                next_action=decision.next_action,
                confidence_note=decision.confidence_note,
                caveats=[c for c in (decision.caveats or "").split("\n") if c],
            )
            if decision is not None
            else None
        ),
    )


def new_public_id() -> str:
    return "an_" + uuid.uuid4().hex[:12]


def save_analysis(
    result,
    guidance_result,
    *,
    original_filename: str | None,
    stored_image_path: str | None,
    explanation_path: str | None = None,
    decision_result=None,
) -> AnalysisDTO:
    """Persist an analysis + its candidates + guidance + decision snapshot.

    The decision snapshot is captured **at analysis time** so the record keeps
    showing exactly what the system decided, regardless of later threshold,
    mapping or model changes.
    """
    with session_scope() as session:
        analysis = Analysis(
            generated_public_id=new_public_id(),
            original_filename=original_filename,
            stored_image_path=stored_image_path,
            image_sha256=result.image_sha256 or None,
            image_width=result.image_width,
            image_height=result.image_height,
            model_key=result.model_key,
            model_name=result.model_name,
            model_version=result.model_version,
            score_type=result.score_type,
            predicted_class=result.top1.class_name if result.top1 else None,
            confidence_score=result.top1.raw_score if result.top1 else None,
            uncertainty_status=result.uncertainty.status if result.uncertainty else None,
            material_category=(
                result.material_hint.category_slug
                if result.material_hint and guidance_result.confidence_sufficient
                else (result.material_hint.category_slug if result.material_hint else None)
            ),
            inference_duration_ms=result.inference_duration_ms,
            explanation_available=bool(explanation_path),
            explanation_path=explanation_path,
            explanation_method=result.explanation.method if result.explanation else None,
            error_status=result.error,
        )
        for cand in result.predictions:
            analysis.predictions.append(
                PredictionCandidateRecord(
                    rank=cand.rank,
                    class_name=cand.class_name,
                    raw_score=cand.raw_score,
                    calibrated_probability=None,
                    material_slug=cand.material_slug,
                )
            )
        primary = guidance_result.primary
        analysis.guidance.append(
            GuidanceSnapshot(
                guidance_category=primary.category_slug if primary else None,
                headline=guidance_result.headline,
                confidence_sufficient=guidance_result.confidence_sufficient,
                reuse_guidance=primary.reuse_guidance if primary else None,
                disposal_guidance=primary.disposal_guidance if primary else None,
                safety_notes=" ".join(guidance_result.safety_notes) or None,
                source_reference=primary.source_url if primary else None,
                geographic_scope=primary.geographic_scope if primary else None,
            )
        )
        if decision_result is not None:
            analysis.decision = DecisionSnapshot(
                observation=getattr(decision_result, "observation", None),
                material_evidence=getattr(decision_result, "material_evidence", None),
                material_category=getattr(decision_result, "material_category", None),
                material_verified=bool(getattr(decision_result, "material_verified", False)),
                next_action=getattr(decision_result, "next_action", None),
                confidence_note=getattr(decision_result, "confidence_note", None),
                caveats="\n".join(getattr(decision_result, "caveats", []) or []),
            )
        session.add(analysis)
        session.flush()
        dto = _to_dto(analysis)
    logger.info("Analysis persisted", extra={"event": "persist", "analysis_id": dto.public_id})
    return dto


def get_by_public_id(public_id: str) -> AnalysisDTO | None:
    with session_scope() as session:
        analysis = (
            session.execute(select(Analysis).where(Analysis.generated_public_id == public_id))
            .scalars()
            .one_or_none()
        )
        return _to_dto(analysis) if analysis else None


def list_analyses(
    *,
    search: str | None = None,
    material_category: str | None = None,
    uncertainty_status: str | None = None,
    model_key: str | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    sort: str = "newest",
    page: int = 1,
    page_size: int = 8,
) -> tuple[list[AnalysisDTO], int]:
    page = max(page, 1)
    page_size = max(min(page_size, 100), 1)
    with session_scope() as session:
        stmt = select(Analysis)
        count_stmt = select(func.count(Analysis.id))

        conditions = []
        if search:
            # Normalise whitespace so multi-word searches behave predictably.
            cleaned = re.sub(r"\s+", " ", search.strip())
            like = f"%{cleaned}%"
            conditions.append(
                Analysis.predicted_class.ilike(like) | Analysis.original_filename.ilike(like)
            )
        if material_category:
            conditions.append(Analysis.material_category == material_category)
        if uncertainty_status:
            conditions.append(Analysis.uncertainty_status == uncertainty_status)
        if model_key:
            conditions.append(Analysis.model_key == model_key)
        if date_from:
            conditions.append(Analysis.created_at >= _utc_naive(date_from))
        if date_to:
            conditions.append(Analysis.created_at <= _utc_naive(date_to))

        for cond in conditions:
            stmt = stmt.where(cond)
            count_stmt = count_stmt.where(cond)

        created_asc = Analysis.created_at.asc()
        created_desc = Analysis.created_at.desc()
        order_by = {
            "oldest": (created_asc, Analysis.id.asc()),
            "score_high": (Analysis.confidence_score.desc(), created_desc, Analysis.id.desc()),
            "score_low": (Analysis.confidence_score.asc(), created_desc, Analysis.id.desc()),
        }.get(sort, (created_desc, Analysis.id.desc()))
        stmt = stmt.order_by(*order_by)
        stmt = stmt.offset((page - 1) * page_size).limit(page_size)

        rows = session.execute(stmt).scalars().all()
        total = int(session.execute(count_stmt).scalar_one())
        dtos = [_to_dto(a) for a in rows]
    return dtos, total


def delete_analysis(public_id: str) -> bool:
    with session_scope() as session:
        analysis = (
            session.execute(select(Analysis).where(Analysis.generated_public_id == public_id))
            .scalars()
            .one_or_none()
        )
        if analysis is None:
            return False
        image_path = analysis.stored_image_path
        explanation_path = analysis.explanation_path
        session.delete(analysis)  # cascades to candidates + guidance
    delete_stored_file(image_path)
    delete_stored_file(explanation_path)
    logger.info("Analysis deleted", extra={"event": "delete", "analysis_id": public_id})
    return True


def delete_all() -> int:
    """Delete every analysis and its files. Returns the number removed."""
    with session_scope() as session:
        rows = session.execute(select(Analysis)).scalars().all()
        paths = [(a.stored_image_path, a.explanation_path) for a in rows]
        count = len(rows)
        session.execute(delete(Analysis))
    for image_path, explanation_path in paths:
        delete_stored_file(image_path)
        delete_stored_file(explanation_path)
    return count


VALID_VERDICTS = {"correct", "incorrect", "unsure"}
VALID_CORRECTIONS = {
    "paper_cardboard",
    "plastic",
    "glass",
    "metal",
    "textile",
    "e_waste",
    "battery",
    "organic",
    "mixed_uncertain",
}


def _feedback_to_dto(feedback: Feedback) -> FeedbackDTO:
    return FeedbackDTO(
        verdict=feedback.verdict,
        material_correction=feedback.material_correction,
        corrected_label=feedback.corrected_label,
        note=feedback.note,
        source=feedback.source,
        created_at=feedback.created_at,
        updated_at=getattr(feedback, "updated_at", None),
    )


def save_feedback(
    public_id: str,
    *,
    verdict: str,
    material_correction: str | None = None,
    corrected_label: str | None = None,
    note: str | None = None,
    source: str = "user",
) -> FeedbackDTO | None:
    """Create or update (upsert) feedback for an analysis.

    One feedback row per ``(analysis_id, source)``: re-submitting edits the
    existing row in place instead of creating a duplicate. The model's original
    output is never touched. Returns ``None`` when the analysis does not exist.
    """
    if verdict not in VALID_VERDICTS:
        raise ValueError(f"Invalid verdict: {verdict}. Use {sorted(VALID_VERDICTS)}.")
    if material_correction not in (None, *VALID_CORRECTIONS):
        raise ValueError(f"Invalid material correction: {material_correction}.")

    clean_note = _clean_text(note, MAX_NOTE_LENGTH)
    clean_correction = material_correction or None
    clean_label = _clean_text(corrected_label, MAX_LABEL_LENGTH)
    if verdict != "incorrect":
        clean_correction = None
        clean_label = None

    with session_scope() as session:
        analysis = (
            session.execute(select(Analysis).where(Analysis.generated_public_id == public_id))
            .scalars()
            .one_or_none()
        )
        if analysis is None:
            return None

        feedback = (
            session.execute(
                select(Feedback).where(
                    Feedback.analysis_id == analysis.id, Feedback.source == source
                )
            )
            .scalars()
            .one_or_none()
        )
        if feedback is None:
            feedback = Feedback(
                analysis_id=analysis.id,
                verdict=verdict,
                source=source,
                created_at=utcnow(),
                updated_at=utcnow(),
            )
            session.add(feedback)
        else:
            feedback.verdict = verdict
            feedback.updated_at = utcnow()

        feedback.material_correction = clean_correction
        feedback.corrected_label = clean_label
        feedback.note = clean_note
        session.flush()
        dto = _feedback_to_dto(feedback)
    logger.info(
        "Feedback saved",
        extra={"event": "feedback", "analysis_id": public_id, "source": source},
    )
    return dto


# Backwards-compatible alias used by earlier code/tests: same upsert behaviour.
add_feedback = save_feedback


def get_feedback(public_id: str) -> list[FeedbackDTO]:
    """Return all feedback rows attached to an analysis (empty list when none)."""
    with session_scope() as session:
        analysis = (
            session.execute(select(Analysis).where(Analysis.generated_public_id == public_id))
            .scalars()
            .one_or_none()
        )
        if analysis is None:
            return []
        rows = (
            session.execute(
                select(Feedback)
                .where(Feedback.analysis_id == analysis.id)
                .order_by(Feedback.created_at)
            )
            .scalars()
            .all()
        )
        return [_feedback_to_dto(f) for f in rows]


def delete_feedback(public_id: str, *, source: str = "user") -> bool:
    """Remove user feedback from an analysis. Returns False if none existed."""
    with session_scope() as session:
        analysis = (
            session.execute(select(Analysis).where(Analysis.generated_public_id == public_id))
            .scalars()
            .one_or_none()
        )
        if analysis is None:
            return False
        feedback = session.execute(
            select(Feedback).where(
                Feedback.analysis_id == analysis.id, Feedback.source == source
            )
        ).scalars().one_or_none()
        if feedback is None:
            return False
        session.delete(feedback)
    logger.info("Feedback removed", extra={"event": "feedback_delete", "analysis_id": public_id})
    return True


def distinct_model_keys() -> list[str]:
    """Distinct model keys present in stored records."""
    with session_scope() as session:
        rows = session.execute(
            select(Analysis.model_key).where(Analysis.model_key.is_not(None)).distinct()
        ).all()
    return sorted(m for (m,) in rows if m)


def feedback_summary() -> dict:
    """Aggregate user feedback with coverage, labelled as user-reported only.

    Returns a dict even when there is no feedback so callers can render an
    honest empty state. Coverage = analyses that received feedback / all saved
    analyses. ``incorrect_rate`` is user-reported, not model accuracy.
    """
    with session_scope() as session:
        total_analyses = int(session.execute(select(func.count(Analysis.id))).scalar_one())
        rows = session.execute(
            select(Feedback.verdict, func.count(Feedback.id)).group_by(Feedback.verdict)
        ).all()
        total = sum(count for _, count in rows)
        summary = {verdict: int(count) for verdict, count in rows}
        summary.setdefault("correct", 0)
        summary.setdefault("incorrect", 0)
        summary.setdefault("unsure", 0)
        summary["total"] = total
        analyses_with_feedback = int(
            session.execute(select(func.count(Feedback.analysis_id.distinct()))).scalar_one()
        )
        summary["analyses_with_feedback"] = analyses_with_feedback
        summary["total_analyses"] = total_analyses
        summary["coverage"] = (
            round(analyses_with_feedback / total_analyses, 4) if total_analyses else 0.0
        )
        summary["incorrect_rate"] = round(summary["incorrect"] / total, 4) if total else 0.0
        # For backward compatibility, tests expect {} when total==0.
        if total == 0 and total_analyses == 0:
            return {}
        if total == 0:
            summary["incorrect_rate"] = 0.0
        return summary
