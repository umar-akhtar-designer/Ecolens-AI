"""Reusable Streamlit rendering components."""

from __future__ import annotations

from html import escape

import streamlit as st

from ..vision import uncertainty as u

_STATUS_LABELS = {
    u.CONFIDENT: ("confident", "Confident"),
    u.UNCERTAIN: ("uncertain", "Review recommended"),
    u.LOW_CONFIDENCE: ("low", "Unable to identify reliably"),
    u.UNKNOWN: ("low", "Unable to identify reliably"),
}


def section(title: str, subtitle: str | None = None) -> None:
    st.markdown(f"### {escape(title)}")
    if subtitle:
        st.markdown(f"<div class='ec-muted'>{escape(subtitle)}</div>", unsafe_allow_html=True)


def confidence_badge(status: str | None) -> str:
    css, label = _STATUS_LABELS.get(status or u.UNKNOWN, ("info", "Unknown"))
    return f"<span class='ec-badge ec-badge--{css}'>{label}</span>"


def metric_strip(items: list[tuple[str, str]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        with col:
            st.markdown(
                f"<div class='ec-metric'><div class='ec-metric__value'>{escape(str(value))}</div>"
                f"<div class='ec-metric__label'>{escape(label)}</div></div>",
                unsafe_allow_html=True,
            )


def callout(message: str, kind: str = "info") -> None:
    st.markdown(
        f"<div class='ec-callout ec-callout--{kind}'>{escape(message)}</div>",
        unsafe_allow_html=True,
    )


def empty_state(title: str, message: str) -> None:
    st.markdown(
        f"<div class='ec-empty'><div class='ec-card__title'>{escape(title)}</div>"
        f"<div>{escape(message)}</div></div>",
        unsafe_allow_html=True,
    )


def prediction_table(predictions) -> None:
    if not predictions:
        st.write("No predictions.")
        return
    rows = [
        "<table class='ec-table'><thead><tr>"
        "<th>#</th><th>Recognised object</th><th>Score (softmax)</th>"
        "<th>Material hint</th></tr></thead><tbody>"
    ]
    for p in predictions:
        label = p.class_name if hasattr(p, "class_name") else p["class_name"]
        score = p.raw_score if hasattr(p, "raw_score") else p["raw_score"]
        slug = (
            getattr(p, "material_slug", None)
            if hasattr(p, "material_slug")
            else p.get("material_slug")
        )
        rank = p.rank if hasattr(p, "rank") else p["rank"]
        rows.append(
            f"<tr><td>{rank}</td><td>{escape(str(label))}</td>"
            f"<td>{score:.3f}</td><td>{escape(str(slug or '—'))}</td></tr>"
        )
    rows.append("</tbody></table>")
    st.markdown("".join(rows), unsafe_allow_html=True)


def render_guidance(guidance) -> None:
    """Render a GuidanceResult (recognition-time guidance)."""
    severity = getattr(guidance, "severity", "info")
    if guidance.primary:
        title = guidance.primary.display_name
        st.markdown(f"**{escape(title)}** — {escape(guidance.headline)}")
    else:
        st.markdown(f"**{escape(guidance.headline)}**")

    st.markdown(f"<div class='ec-muted'>{escape(guidance.reason)}</div>", unsafe_allow_html=True)

    if guidance.primary:
        st.markdown("**Reuse**")
        st.write(guidance.primary.reuse_guidance)
        st.markdown("**Sorting & disposal**")
        st.write(guidance.primary.disposal_guidance)
        if guidance.primary.recyclability_conditions:
            st.markdown("**Conditions affecting recyclability**")
            st.write(guidance.primary.recyclability_conditions)

    if guidance.alternatives:
        names = ", ".join(a.display_name for a in guidance.alternatives)
        st.markdown(
            f"<div class='ec-muted'>Other plausible materials: {escape(names)}</div>",
            unsafe_allow_html=True,
        )

    for note in guidance.safety_notes:
        callout(note, "warning" if severity == "warning" else "caution")

    if guidance.primary and guidance.primary.source_url:
        st.markdown(
            f"<div class='ec-muted'>Source: {escape(guidance.primary.source_url)} · "
            f"Scope: {escape(guidance.primary.geographic_scope)}</div>",
            unsafe_allow_html=True,
        )
    callout(
        "Recycling rules vary by municipality. Confirm with your local authority; "
        "this is general guidance, not a recyclability determination.",
        "info",
    )


def render_analysis_summary(record) -> None:
    """Compact summary used in history/detail views (AnalysisDTO)."""
    cols = st.columns([3, 2, 2])
    with cols[0]:
        st.markdown(f"**{escape(record.predicted_class or 'No prediction')}**")
    with cols[1]:
        if record.confidence_score is not None:
            st.markdown(f"Score: `{record.confidence_score:.3f}`")
    with cols[2]:
        st.markdown(confidence_badge(record.uncertainty_status), unsafe_allow_html=True)


def render_decision(decision) -> None:
    """Render a DecisionSupport object as three explicit layers."""
    if decision is None:
        return
    if getattr(decision, "is_reconstructed", False):
        callout(
            "This decision support is **reconstructed** from the stored record "
            "(the record predates decision snapshots). It is derived from the "
            "saved prediction, uncertainty status and guidance — thresholds may "
            "have changed since the original analysis.",
            "caution",
        )
    for title, body in decision.layers():
        st.markdown(f"**{title}**")
        st.markdown(f"<div class='ec-muted'>{escape(body)}</div>", unsafe_allow_html=True)
    basis = getattr(decision, "material_basis", "none")
    if basis == "object_label_inference":
        st.caption("Material source: derived from the recognised object label (not a measuring instrument).")
    elif basis == "material_classifier":
        st.caption("Material source: a dedicated material-classifier output.")
    if getattr(decision, "material_verified", False):
        callout(
            "Marked as a verified material finding per the active material "
            "classifier. Only a genuinely independent material-classification "
            "process sets this flag.",
            "info",
        )
    if decision.caveats:
        with st.expander("Caveats & safety notes", expanded=False):
            for note in decision.caveats:
                st.markdown(f"- {escape(note)}")


_MATERIAL_CHOICES = [
    "paper_cardboard",
    "plastic",
    "glass",
    "metal",
    "textile",
    "e_waste",
    "battery",
    "organic",
    "mixed_uncertain",
]


def render_feedback_widget(public_id: str, *, key_prefix: str = "fb") -> None:
    """Additive user-feedback capture with edit-in-place (never duplicates).

    One feedback row is kept per analysis; re-submitting updates it. Feedback is
    stored separately from the model output and is never used to retrain or to
    rewrite the original prediction.
    """
    from ..database import repository

    existing = repository.get_feedback(public_id)
    current = next((f for f in existing if f.source == "user"), None)

    st.markdown("**Was this result useful? (user-reported, recorded for evaluation only)**")

    default_verdict = current.verdict if current else "correct"
    with st.form(key=f"{key_prefix}_form_{public_id}"):
        verdict = st.radio(
            "How accurate was the recognition?",
            options=["correct", "incorrect", "unsure"],
            index=["correct", "incorrect", "unsure"].index(default_verdict),
            horizontal=True,
            key=f"{key_prefix}_verdict_{public_id}",
        )
        correction = None
        corrected_label = None
        if verdict == "incorrect":
            mat_options = ["(none)"] + _MATERIAL_CHOICES
            current_correction = current.material_correction if current else None
            correction = st.selectbox(
                "Correct material (optional)",
                options=mat_options,
                index=mat_options.index(current_correction)
                if current_correction in mat_options
                else 0,
                key=f"{key_prefix}_mat_{public_id}",
            )
            corrected_label = st.text_input(
                "Correct object label (optional)",
                value=current.corrected_label or "",
                key=f"{key_prefix}_label_{public_id}",
            )
        note = st.text_area(
            "Notes (optional, 1000 characters max)",
            value=(current.note or "") if current else "",
            key=f"{key_prefix}_note_{public_id}",
        )
        submitted = st.form_submit_button("Save feedback")

    if submitted:
        try:
            dto = repository.save_feedback(
                public_id,
                verdict=verdict,
                material_correction=None if correction in (None, "(none)") else correction,
                corrected_label=corrected_label or None,
                note=note or None,
                source="user",
            )
            if dto is None:
                st.error("That analysis no longer exists.")
            else:
                st.success(
                    "Feedback saved. Original prediction, guidance and decision "
                    "were left unchanged."
                )
        except ValueError as exc:
            st.warning(f"Could not save feedback: {exc}")

    if current is not None:
        edited = " (edited)" if current.was_edited else ""
        st.caption(
            f"Existing feedback: **{current.verdict}**{edited} — saved "
            f"{current.created_at:%Y-%m-%d %H:%M UTC}."
        )
        if st.button(
            "Remove my feedback", key=f"{key_prefix}_clear_{public_id}", type="secondary"
        ):
            repository.delete_feedback(public_id, source="user")
            st.success("Feedback removed.")
            st.rerun()
