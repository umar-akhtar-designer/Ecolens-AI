"""Decision-support layer.

Turns a raw prediction into a three-layer explanation designed to help a human
*decide* rather than to pretend the model is authoritative:

* **Layer A — Observation:** what the model actually did (recognised object +
  score, with the honest caveat that this is an uncalibrated softmax score).
* **Layer B — Material evidence:** whether a material category can be inferred,
  and — critically — *how* (object-label inference, never measurement), or why
  the system abstains.
* **Layer C — Recommended next action:** the single most useful next step,
  adapted to the confidence state and any safety hazard.

This module contains **no** hidden logic and **no** invented statistics. Every
statement is derived from values produced by the running pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..vision import uncertainty as uncertainty_mod
from ..vision.material_mapping import is_container_label

MATERIAL_BASIS_OBJECT = "object_label_inference"
MATERIAL_BASIS_DIRECT = "material_classifier"
MATERIAL_BASIS_NONE = "none"


@dataclass
class DecisionSupport:
    observation: str
    material_evidence: str
    material_category: str | None
    material_verified: bool
    next_action: str
    confidence_note: str
    caveats: list[str] = field(default_factory=list)
    material_basis: str = MATERIAL_BASIS_NONE
    is_reconstructed: bool = False

    def layers(self) -> list[tuple[str, str]]:
        return [
            ("Observation", self.observation),
            ("Material evidence", self.material_evidence),
            ("Recommended next action", self.next_action),
        ]


def build_decision(analysis, guidance) -> DecisionSupport:
    """Build decision support from an :class:`AnalysisResult` and guidance."""
    top = getattr(analysis, "top1", None)
    unc = getattr(analysis, "uncertainty", None)
    hint = getattr(analysis, "material_hint", None)
    hazard = getattr(analysis, "hazard_note", None)
    confidence_sufficient = bool(getattr(guidance, "confidence_sufficient", False))
    safety = list(getattr(guidance, "safety_notes", []) or [])
    label = top.class_name if top is not None else None

    # --- Layer A: observation -------------------------------------------------
    if top is None:
        observation = "The model returned no prediction for this image."
    else:
        observation = (
            f"The model's closest match is the object '{label}' with a softmax "
            f"score of {top.raw_score:.3f} ({uncertainty_mod.SCORE_TYPE}). "
            "This score is uncalibrated and is not the probability of being correct."
        )

    # --- Layer B: material evidence -------------------------------------------
    material_verified = False
    material_category = None
    material_basis = MATERIAL_BASIS_NONE
    container = is_container_label(label)

    if hint is None:
        material_evidence = (
            "No material category could be inferred: the recognised object is "
            "outside EcoLens's curated material map, so the material is left "
            "unknown rather than guessed."
        )
    elif not confidence_sufficient:
        material_evidence = (
            f"A possible material ('{hint.category_slug}') was suggested by the "
            f"object label, but confidence is insufficient to stand behind it. "
            f"{hint.evidence}"
        )
    else:
        material_category = hint.category_slug
        material_verified = bool(hint.is_direct_material_classifier)
        material_basis = (
            MATERIAL_BASIS_DIRECT if hint.is_direct_material_classifier else MATERIAL_BASIS_OBJECT
        )
        material_evidence = (
            f"Probable material: '{hint.category_slug}'. {hint.evidence}"
            + (
                ""
                if hint.is_direct_material_classifier
                else " Confirm against a visible label, cap or film before acting."
            )
        )

    if container:
        if material_category:
            material_evidence += (
                " This tells you about the container itself — the *contents* of "
                "the bin cannot be inferred from the image."
            )
        else:
            material_evidence += " The contents of the container are not inferred."

    # --- Layer C: recommended next action -------------------------------------
    if hazard:
        next_action = f"Safety first: {hazard}"
    elif container:
        next_action = (
            "Inspect the item(s) directly and any visible labels. The model only "
            "identified the container, so sort based on what you can see, or "
            "consult your local waste authority."
        )
    elif top is None or (unc is not None and unc.abstain and not material_category):
        next_action = (
            "Take a clearer, closer, well-lit photo of a single item on a plain "
            "background, then re-run the analysis. Do not rely on this result to "
            "sort the item."
        )
    elif material_category:
        next_action = (
            f"Verify the material visually, then follow your local recycling "
            f"rules for '{material_category}'. Use the Guidance tab for reuse and "
            "disposal options."
        )
    else:
        next_action = (
            "Treat the material as unverified. Check the item's label or film "
            "directly and consult your local authority before disposing."
        )

    if unc is not None:
        confidence_note = unc.explanation
        if unc.abstain and unc.abstain_reason:
            confidence_note = f"{confidence_note} ({unc.abstain_reason})"
    else:
        confidence_note = "Confidence state unavailable."

    caveats = list(dict.fromkeys(safety))
    if hint is not None and not hint.is_direct_material_classifier:
        caveats.append(
            "Material is inferred from an object classifier; contamination, "
            "coatings and multi-material items are not detected."
        )
    if container:
        caveats.append(
            "Recognising a bin or truck says nothing about what is inside it."
        )

    return DecisionSupport(
        observation=observation,
        material_evidence=material_evidence,
        material_category=material_category,
        material_verified=material_verified,
        next_action=next_action,
        confidence_note=confidence_note,
        caveats=caveats,
        material_basis=material_basis,
        is_reconstructed=False,
    )


def decision_for_record(dto) -> DecisionSupport:
    """Return the decision support for a stored record.

    Prefers the **persisted** decision snapshot captured at analysis time. When a
    legacy record predates decision snapshots, rebuilds it from the stored
    prediction and marks it explicitly as reconstructed.
    """
    snap = getattr(dto, "decision", None)
    if snap is not None:
        return DecisionSupport(
            observation=snap.observation,
            material_evidence=snap.material_evidence,
            material_category=snap.material_category,
            material_verified=bool(snap.material_verified),
            next_action=snap.next_action,
            confidence_note=snap.confidence_note,
            caveats=list(getattr(snap, "caveats", []) or []),
            material_basis=getattr(snap, "material_basis", None) or MATERIAL_BASIS_OBJECT,
            is_reconstructed=False,
        )
    return decision_from_record(dto)


def decision_from_record(dto) -> DecisionSupport:
    """Rebuild decision support for a stored analysis without a snapshot.

    The result is explicitly identified as reconstructed: it is derived from the
    stored uncertainty status, which may differ from what was originally shown
    if thresholds have changed.
    """

    class _Top:
        def __init__(self, name, score):
            self.class_name = name
            self.raw_score = score

    class _Unc:
        def __init__(self, status, explanation):
            self.status = status
            self.explanation = explanation
            self.abstain = status in (
                uncertainty_mod.LOW_CONFIDENCE,
                uncertainty_mod.UNKNOWN,
            ) or not (status == uncertainty_mod.CONFIDENT)
            self.abstain_reason = (
                "Stored record flagged below the confidence threshold."
                if status == uncertainty_mod.LOW_CONFIDENCE
                else (
                    "Stored record was not confident enough to support a material claim."
                    if status == uncertainty_mod.UNCERTAIN
                    else ""
                )
            )

    class _Analysis:
        pass

    class _Guidance:
        pass

    top1 = dto.predictions[0] if dto.predictions else None
    a = _Analysis()
    a.top1 = (
        _Top(top1.class_name, top1.raw_score) if top1 is not None and dto.predicted_class else None
    )
    a.material_hint = None
    if dto.material_category:
        from ..vision.material_mapping import MaterialHint

        a.material_hint = MaterialHint(
            category_slug=dto.material_category,
            reason="Stored material hint.",
            matched_label=dto.predicted_class or "unknown",
        )
    a.hazard_note = None
    a.uncertainty = _Unc(
        dto.uncertainty_status or uncertainty_mod.UNKNOWN,
        "Rebuilt from the stored uncertainty status (thresholds may have changed).",
    )

    g = _Guidance()
    g.confidence_sufficient = bool(dto.guidance and dto.guidance.confidence_sufficient)
    g.safety_notes = (
        [] if not dto.guidance or not dto.guidance.safety_notes else [dto.guidance.safety_notes]
    )
    decision = build_decision(a, g)
    decision.is_reconstructed = True
    return decision