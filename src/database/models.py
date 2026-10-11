"""SQLAlchemy ORM models for EcoLens AI.

Only product-relevant data is stored. Uploaded images live on disk under
``storage/uploads`` with generated filenames — the database stores a relative
path, never a public URL and never an absolute server path exposed to users.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    generated_public_id: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )

    original_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    stored_image_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    image_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    image_width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    image_height: Mapped[int | None] = mapped_column(Integer, nullable=True)

    model_key: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(512), nullable=True)
    score_type: Mapped[str | None] = mapped_column(String(64), nullable=True)

    predicted_class: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    confidence_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    uncertainty_status: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    material_category: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    inference_duration_ms: Mapped[float | None] = mapped_column(Float, nullable=True)

    explanation_available: Mapped[bool] = mapped_column(default=False)
    explanation_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    explanation_method: Mapped[str | None] = mapped_column(String(64), nullable=True)

    error_status: Mapped[str | None] = mapped_column(String(128), nullable=True)

    predictions: Mapped[list[PredictionCandidateRecord]] = relationship(
        back_populates="analysis",
        cascade="all, delete-orphan",
        order_by="PredictionCandidateRecord.rank",
    )
    guidance: Mapped[list[GuidanceSnapshot]] = relationship(
        back_populates="analysis",
        cascade="all, delete-orphan",
    )
    feedback: Mapped[list[Feedback]] = relationship(
        back_populates="analysis",
        cascade="all, delete-orphan",
        order_by="Feedback.created_at",
    )
    decision: Mapped[DecisionSnapshot | None] = relationship(
        back_populates="analysis",
        cascade="all, delete-orphan",
        uselist=False,
    )

    __table_args__ = (Index("ix_analyses_created_material", "created_at", "material_category"),)


class PredictionCandidateRecord(Base):
    __tablename__ = "prediction_candidates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_id: Mapped[int] = mapped_column(
        ForeignKey("analyses.id", ondelete="CASCADE"), index=True
    )
    rank: Mapped[int] = mapped_column(Integer)
    class_name: Mapped[str] = mapped_column(String(128))
    raw_score: Mapped[float] = mapped_column(Float)
    calibrated_probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    material_slug: Mapped[str | None] = mapped_column(String(64), nullable=True)

    analysis: Mapped[Analysis] = relationship(back_populates="predictions")


class GuidanceSnapshot(Base):
    __tablename__ = "guidance_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_id: Mapped[int] = mapped_column(
        ForeignKey("analyses.id", ondelete="CASCADE"), index=True
    )
    guidance_category: Mapped[str | None] = mapped_column(String(64), nullable=True)
    headline: Mapped[str | None] = mapped_column(String(255), nullable=True)
    confidence_sufficient: Mapped[bool] = mapped_column(default=False)
    reuse_guidance: Mapped[str | None] = mapped_column(Text, nullable=True)
    disposal_guidance: Mapped[str | None] = mapped_column(Text, nullable=True)
    safety_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_reference: Mapped[str | None] = mapped_column(String(512), nullable=True)
    geographic_scope: Mapped[str | None] = mapped_column(String(128), nullable=True)

    analysis: Mapped[Analysis] = relationship(back_populates="guidance")


class Feedback(Base):
    """User feedback about a persisted analysis.

    Feedback is stored **separately** from the model output: it never mutates
    the original prediction/guidance, so the audit trail of what the model
    actually produced stays intact. It exists to record where the system was
    right or wrong.

    At most one user feedback row is kept per analysis (unique on
    ``analysis_id`` + ``source``); re-submitting updates it in place rather than
    creating duplicates. ``created_at`` is set once, ``updated_at`` on each edit.
    """

    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_id: Mapped[int] = mapped_column(
        ForeignKey("analyses.id", ondelete="CASCADE"), index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )
    verdict: Mapped[str] = mapped_column(String(32))
    material_correction: Mapped[str | None] = mapped_column(String(64), nullable=True)
    corrected_label: Mapped[str | None] = mapped_column(String(128), nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[str] = mapped_column(String(32), default="user")

    analysis: Mapped[Analysis] = relationship(back_populates="feedback")

    __table_args__ = (UniqueConstraint("analysis_id", "source", name="uq_feedback_analysis_source"),)


class DecisionSnapshot(Base):
    """The decision-support output captured at analysis time.

    Persisting this (rather than recomputing on every view) guarantees a record
    always shows what the system actually decided — even if thresholds, material
    mappings or models change later. ``is_reconstructed`` is reserved for views
    of legacy records that predate this table; those are rebuilt on the fly and
    clearly labelled, never written back.
    """

    __tablename__ = "decision_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_id: Mapped[int] = mapped_column(
        ForeignKey("analyses.id", ondelete="CASCADE"), unique=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )
    observation: Mapped[str | None] = mapped_column(Text, nullable=True)
    material_evidence: Mapped[str | None] = mapped_column(Text, nullable=True)
    material_category: Mapped[str | None] = mapped_column(String(64), nullable=True)
    material_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    next_action: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    caveats: Mapped[str | None] = mapped_column(Text, nullable=True)

    analysis: Mapped[Analysis] = relationship(back_populates="decision")


class MaterialKnowledgeRecord(Base):
    __tablename__ = "material_knowledge"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    category_slug: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(128))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    reuse_guidance: Mapped[str | None] = mapped_column(Text, nullable=True)
    disposal_guidance: Mapped[str | None] = mapped_column(Text, nullable=True)
    safety_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    geographic_scope: Mapped[str | None] = mapped_column(String(128), nullable=True)
    reviewed_at: Mapped[str | None] = mapped_column(String(32), nullable=True)
