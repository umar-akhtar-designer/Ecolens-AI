"""End-to-end image analysis pipeline.

``analyze_image_bytes`` is the single entry point used by the application and
the tests. It validates, decodes, preprocesses, runs the model, estimates
uncertainty, derives a (non-authoritative) material hint, and optionally
produces a Grad-CAM explanation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from PIL import Image

from ..config import SETTINGS
from ..logging_config import get_logger
from . import uncertainty as uncertainty_mod
from .explanations import ExplanationError, ExplanationResult, generate_gradcam
from .material_mapping import (
    MaterialHint,
    hazard_note_for_label,
    map_label_to_material,
)
from .model_registry import ModelLoadError, get_device, get_model, get_model_spec
from .preprocessing import DecodedImage, ImageDecodeError, decode_image_bytes

logger = get_logger("vision.inference")


class InferenceError(RuntimeError):
    """Raised when inference cannot be completed."""


@dataclass
class PredictionCandidate:
    rank: int
    class_index: int
    class_name: str
    raw_score: float
    material_slug: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "class_index": self.class_index,
            "class_name": self.class_name,
            "raw_score": self.raw_score,
            "material_slug": self.material_slug,
        }


@dataclass
class AnalysisResult:
    ok: bool
    predictions: list[PredictionCandidate] = field(default_factory=list)
    top1: PredictionCandidate | None = None
    uncertainty: uncertainty_mod.UncertaintyReport | None = None
    material_hint: MaterialHint | None = None
    hazard_note: str | None = None
    explanation: ExplanationResult | None = None
    explanation_error: str | None = None
    image_width: int = 0
    image_height: int = 0
    image_sha256: str = ""
    inference_duration_ms: float = 0.0
    model_key: str = ""
    model_name: str = ""
    model_version: str = ""
    score_type: str = uncertainty_mod.SCORE_TYPE
    is_direct_material_classifier: bool = False
    device: str = "cpu"
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def display_label(self) -> str:
        if self.top1 is None:
            return "No prediction"
        return self.top1.class_name


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def _run_model(image: Image.Image, model_key: str | None):
    spec = get_model_spec(model_key)
    model = get_model(model_key)
    device = get_device()

    tensor = spec.transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
    return spec, logits.detach().cpu().numpy()[0]


def analyze_image_bytes(
    data: bytes,
    *,
    filename: str | None = None,
    model_key: str | None = None,
    with_explanation: bool = True,
) -> AnalysisResult:
    """Analyse raw image bytes and return a structured result.

    Never raises for expected failures — decode/model errors are captured in
    ``AnalysisResult.error`` with ``ok=False`` so the UI can render gracefully.
    """
    started = time.perf_counter()
    try:
        decoded: DecodedImage = decode_image_bytes(data)
    except ImageDecodeError as exc:
        return AnalysisResult(ok=False, error=str(exc))

    spec = get_model_spec(model_key)
    result = AnalysisResult(
        ok=True,
        image_width=decoded.width,
        image_height=decoded.height,
        image_sha256=decoded.sha256,
        model_key=spec.key,
        model_name=spec.display_name,
        model_version=spec.version,
        score_type=uncertainty_mod.SCORE_TYPE,
        is_direct_material_classifier=spec.is_direct_material_classifier,
        device=str(get_device()),
    )

    try:
        _, logits = _run_model(decoded.image, model_key)
    except ModelLoadError as exc:
        result.ok = False
        result.error = f"Model could not be loaded: {exc}"
        return result
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Inference failed")
        result.ok = False
        result.error = f"Inference failed: {exc}"
        return result

    probs = _softmax(logits.astype(np.float64))
    top_k = min(SETTINGS.top_k, probs.size)
    order = np.argsort(probs)[::-1][:top_k]

    candidates: list[PredictionCandidate] = []
    for rank, idx in enumerate(order, start=1):
        label = spec.categories[idx] if idx < len(spec.categories) else f"class_{idx}"
        hint = map_label_to_material(label)
        candidates.append(
            PredictionCandidate(
                rank=rank,
                class_index=int(idx),
                class_name=label,
                raw_score=float(probs[idx]),
                material_slug=hint.category_slug if hint else None,
            )
        )

    result.predictions = candidates
    result.top1 = candidates[0] if candidates else None

    if result.top1 is not None:
        result.material_hint = map_label_to_material(result.top1.class_name)
        result.hazard_note = hazard_note_for_label(result.top1.class_name)

    # The supported-domain flag lets the uncertainty layer abstain from material
    # claims for objects outside the curated material map.
    result.uncertainty = uncertainty_mod.analyse(
        probs,
        num_classes=spec.num_classes,
        supported_domain=result.material_hint is not None,
    )

    if result.uncertainty and result.uncertainty.status in (
        uncertainty_mod.LOW_CONFIDENCE,
        uncertainty_mod.UNKNOWN,
    ):
        result.warnings.append("Low-confidence prediction: no material-specific guidance is shown.")

    if with_explanation and result.top1 is not None:
        try:
            result.explanation = generate_gradcam(
                decoded.image, result.top1.class_index, model_key=spec.key
            )
        except ExplanationError as exc:
            result.explanation_error = str(exc)
            logger.warning(
                "Explanation failed", extra={"event": "explanation_failure", "error": str(exc)}
            )
        except Exception as exc:  # pragma: no cover - defensive
            result.explanation_error = f"Unexpected explanation error: {exc}"
            logger.exception("Explanation crashed")

    result.inference_duration_ms = round((time.perf_counter() - started) * 1000.0, 2)
    logger.info(
        "Analysis complete",
        extra={
            "event": "analysis",
            "model": spec.key,
            "duration_ms": result.inference_duration_ms,
        },
    )
    return result


def warm_up(model_key: str | None = None) -> None:
    """Load the model and run a tiny forward pass to prime caches."""
    spec = get_model_spec(model_key)
    model = get_model(model_key)
    dummy = torch.zeros(1, 3, 224, 224, device=get_device())
    with torch.no_grad():
        model(dummy)
    logger.info("Model warmed up", extra={"event": "warmup", "model": spec.key})
