"""Uncertainty handling and confidence-state assessment.

The values here are *softmax scores* from a pretrained model. They are **not**
calibrated probabilities — a neural network's softmax output is typically
overconfident. We expose this distinction explicitly (``score_type``) and use
simple, transparent **heuristics** to flag low-confidence predictions and to
abstain when the evidence is insufficient.

The thresholds are operational safeguards derived from inspection, **not**
statistically validated calibration. No accuracy guarantee is implied.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..config import SETTINGS

# Confidence states used throughout the app.
CONFIDENT = "confident"  # type: ignore[misc]
UNCERTAIN = "uncertain"
LOW_CONFIDENCE = "low_confidence"
UNKNOWN = "unknown"

# Human-facing labels (used by the UI so wording stays consistent).
STATUS_LABELS = {
    CONFIDENT: "Confident",
    UNCERTAIN: "Review recommended",
    LOW_CONFIDENCE: "Unable to identify reliably",
    UNKNOWN: "Unable to identify reliably",
}

SCORE_TYPE = "softmax_score_uncalibrated"

# Above this top score the prediction is treated as usable observation evidence.
# Documented as a heuristic, not a calibrated probability.
CONFIDENT_THRESHOLD = SETTINGS.confidence_threshold


@dataclass(frozen=True)
class UncertaintyReport:
    top1: float
    top2: float
    margin: float
    entropy: float
    normalized_entropy: float
    status: str
    explanation: str
    supported_domain: bool = True
    abstain: bool = False
    abstain_reason: str = ""
    heuristics: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return STATUS_LABELS.get(self.status, "Unknown")


def normalize_entropy(probs: np.ndarray) -> float:
    """Shannon entropy normalised to [0, 1] by log(1000) (the model's support)."""
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0
    entropy = float(-(probs * np.log(probs)).sum())
    return min(entropy / max(math.log(1000), 1e-9), 1.0)


def analyse(
    probs: np.ndarray,
    num_classes: int | None = None,
    supported_domain: bool = True,
) -> UncertaintyReport:
    """Summarise a probability vector into a transparency-friendly report.

    ``supported_domain`` reflects whether the top label maps into the
    application's supported material domain. When it does not, the system
    abstains from any material claim even if the object score is high.
    """
    probs = np.asarray(probs, dtype=np.float64).ravel()
    order = np.argsort(probs)[::-1]
    top1 = float(probs[order[0]])
    top2 = float(probs[order[1]]) if probs.size > 1 else 0.0
    margin = top1 - top2

    n = num_classes or probs.size
    positive = probs[probs > 0]
    entropy = float(-(positive * np.log(positive)).sum()) if positive.size else 0.0
    norm_entropy = min(entropy / max(math.log(max(n, 2)), 1e-9), 1.0)

    threshold = CONFIDENT_THRESHOLD
    entropy_threshold = SETTINGS.uncertainty_entropy_threshold

    abstain = False
    abstain_reason = ""

    if top1 < threshold:
        status = LOW_CONFIDENCE
        alternates = ", ".join(f"'{i}'" for i in range(1, min(4, probs.size)))
        explanation = (
            f"The top score ({top1:.3f}) is below the operating threshold "
            f"({threshold:.2f}). Treat this as a weak guess; other labels "
            f"(ranks {alternates}) are plausible."
        )
        abstain = True
        abstain_reason = "Top score below the confidence threshold."
    elif norm_entropy > entropy_threshold or margin < 0.10:
        status = UNCERTAIN
        explanation = (
            "Several classes received similar scores, so the model is not "
            "confident. Review visually before acting; the material is likely "
            "to remain unverified."
        )
    else:
        status = CONFIDENT
        explanation = (
            "The model favours one object class clearly. Softmax scores are not "
            "calibrated probabilities, and object recognition does not establish "
            "material composition — visual confirmation is still advised."
        )

    if not supported_domain:
        # Even a confident object prediction that we cannot map to a material
        # category must not produce a material claim.
        if status == CONFIDENT:
            status = UNCERTAIN
        abstain = True
        abstain_reason = (
            "Recognised object is outside the supported material domain; no "
            "material inference is offered."
        )
        explanation += (
            " This object is outside EcoLens's supported material categories, so "
            "no material guidance is offered."
        )

    return UncertaintyReport(
        top1=top1,
        top2=top2,
        margin=margin,
        entropy=entropy,
        normalized_entropy=norm_entropy,
        status=status,
        explanation=explanation,
        supported_domain=supported_domain,
        abstain=abstain,
        abstain_reason=abstain_reason,
        heuristics={
            "confidence_threshold": threshold,
            "entropy_threshold": entropy_threshold,
            "margin_threshold": 0.10,
            "calibrated": False,
        },
    )
