"""Analysis orchestration service.

This module is the single seam between the Streamlit UI and the underlying
vision / guidance / persistence layers. Keeping orchestration here means:

* the UI stays thin and testable;
* the exact same pipeline powers single and batch analysis;
* failures are captured as data (never raised into the UI).

The service performs **no** inventing of results: every value it returns comes
from the live pipeline or the database.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO

from ..config import SETTINGS
from ..database import repository
from ..guidance.decision import DecisionSupport, build_decision
from ..guidance.engine import get_engine
from ..logging_config import get_logger
from ..security.uploads import (
    UploadValidationError,
    save_explanation,
    save_image,
    validate_and_decode,
)
from ..vision.inference import AnalysisResult, analyze_image_bytes

logger = get_logger("services.analysis")


@dataclass
class AnalysisBundle:
    """Everything produced for one analysis request."""

    filename: str
    ok: bool
    result: AnalysisResult | None = None
    guidance: object | None = None
    decision: DecisionSupport | None = None
    record: repository.AnalysisDTO | None = None
    image_bytes: bytes | None = None
    error: str | None = None


def run_analysis(
    data: bytes,
    filename: str,
    *,
    mime_type: str | None = None,
    model_key: str | None = None,
    persist: bool = True,
    with_explanation: bool = True,
) -> AnalysisBundle:
    """Validate, analyse, guide, and (optionally) persist a single image."""
    bundle = AnalysisBundle(filename=filename or "upload", ok=False, image_bytes=data)

    try:
        validate_and_decode(data, filename, mime_type)
    except UploadValidationError as exc:
        bundle.error = str(exc)
        return bundle

    result = analyze_image_bytes(
        data, filename=filename, model_key=model_key, with_explanation=with_explanation
    )
    bundle.result = result
    if not result.ok:
        bundle.error = result.error or "Analysis failed."
        return bundle

    engine = get_engine()
    bundle.guidance = engine.build_guidance(result)
    bundle.decision = build_decision(result, bundle.guidance)
    bundle.ok = True

    if persist:
        try:
            _, rel_image = save_image(data, filename)
            explanation_rel = None
            if result.explanation is not None:
                buf = BytesIO()
                result.explanation.overlay.save(buf, format="PNG")
                _, explanation_rel = save_explanation(buf.getvalue())
            bundle.record = repository.save_analysis(
                result,
                bundle.guidance,
                original_filename=filename,
                stored_image_path=rel_image,
                explanation_path=explanation_rel,
                decision_result=bundle.decision,
            )
        except Exception as exc:  # persistence must not destroy the result
            logger.exception("Persistence failed")
            bundle.error = f"Analysis succeeded but saving failed: {exc}"

    return bundle


@dataclass
class BatchOutcome:
    items: list[AnalysisBundle] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    limit_reached: bool = False

    @property
    def succeeded(self) -> int:
        return sum(1 for item in self.items if item.ok)

    @property
    def failed(self) -> int:
        return sum(1 for item in self.items if not item.ok)


def run_batch(
    files: list[tuple[str, bytes, str | None]],
    *,
    model_key: str | None = None,
    persist: bool = True,
    with_explanation: bool = False,
    max_files: int | None = None,
) -> BatchOutcome:
    """Analyse multiple uploads independently.

    Each file is isolated: one failure never aborts the batch. Files that
    exceed the per-file size limit are skipped (recorded in ``skipped`` rather
    than silently dropped).
    """
    limit = max_files or SETTINGS.max_batch_files
    outcome = BatchOutcome()

    for index, (filename, data, mime) in enumerate(files):
        if index >= limit:
            outcome.limit_reached = True
            break
        if len(data) > SETTINGS.max_upload_bytes:
            outcome.skipped.append((filename, f"Exceeds {SETTINGS.max_upload_mb} MB limit."))
            continue
        bundle = run_analysis(
            data,
            filename,
            mime_type=mime,
            model_key=model_key,
            persist=persist,
            with_explanation=with_explanation,
        )
        outcome.items.append(bundle)

    logger.info(
        "Batch complete",
        extra={
            "event": "batch",
            "count": len(outcome.items),
            "succeeded": outcome.succeeded,
            "failed": outcome.failed,
        },
    )
    return outcome
