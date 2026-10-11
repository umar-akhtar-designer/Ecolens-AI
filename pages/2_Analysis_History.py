"""Analysis History — searchable, filterable, deletable, with a rich detail view.

Every value shown is read from the stored record for the selected analysis.
Viewing history never loads the model and never reruns inference.
"""

from __future__ import annotations

from datetime import datetime, time, timezone

import streamlit as st

from src.analytics.export import analyses_to_csv
from src.config import SETTINGS
from src.database import repository
from src.guidance.decision import decision_for_record
from src.guidance.engine import get_engine
from src.security.uploads import resolve_storage_path
from src.ui.bootstrap import setup_page, sidebar_footer
from src.ui.components import (
    callout,
    confidence_badge,
    empty_state,
    prediction_table,
    render_decision,
    render_feedback_widget,
    section,
)
from src.vision import uncertainty as u

setup_page("Analysis History")
sidebar_footer()

st.markdown("## Analysis History")
st.markdown(
    "<div class='ec-muted'>Every saved analysis, stored locally. "
    "Search, filter, review and delete records. Opening a record never reloads "
    "the model or reruns inference.</div>",
    unsafe_allow_html=True,
)

engine = get_engine()
category_options = ["All", *[c.category_slug for c in engine.all_categories()]]
status_options = ["All", u.CONFIDENT, u.UNCERTAIN, u.LOW_CONFIDENCE]
model_options = ["All", *repository.distinct_model_keys()]


def _load_file(relative_path: str | None):
    if not relative_path:
        return None
    try:
        if not resolve_storage_path(relative_path).exists():
            return None
        return resolve_storage_path(relative_path).read_bytes()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
fcol1, fcol2, fcol3 = st.columns([2, 1, 1])
with fcol1:
    search = st.text_input("Search (object name or filename)", value="", key="hist_search")
with fcol2:
    material = st.selectbox("Material", category_options, index=0, key="hist_material")
with fcol3:
    status = st.selectbox("Confidence state", status_options, index=0, key="hist_status")

fcol4, fcol5 = st.columns([1, 1])
with fcol4:
    model_filter = st.selectbox("Model", model_options, index=0, key="hist_model")
with fcol5:
    sort = st.selectbox(
        "Sort", ["newest", "oldest", "score_high", "score_low"], index=0, key="hist_sort"
    )

dcol1, dcol2 = st.columns(2)
with dcol1:
    date_from = st.date_input("From date", value=None, key="hist_from")
with dcol2:
    date_to = st.date_input("To date", value=None, key="hist_to")

page_size = SETTINGS.history_page_size

search_key = (search, material, status, model_filter, sort, date_from, date_to)
# Reset the page whenever filters change so the user never lands on an empty page.
if state_key := st.session_state.get("hist_filter_key"):
    if state_key != search_key:
        st.session_state["hist_page"] = 1
st.session_state["hist_filter_key"] = search_key
if "hist_page" not in st.session_state:
    st.session_state["hist_page"] = 1
try:
    requested_page = int(st.session_state["hist_page"])
except (TypeError, ValueError):
    requested_page = 1

df = None
dt = None
if date_from:
    df = datetime.combine(date_from, time.min, tzinfo=timezone.utc)
if date_to:
    dt = datetime.combine(date_to, time.max, tzinfo=timezone.utc)

records, total = repository.list_analyses(
    search=search or None,
    material_category=None if material == "All" else material,
    uncertainty_status=None if status == "All" else status,
    model_key=None if model_filter == "All" else model_filter,
    date_from=df,
    date_to=dt,
    sort=sort,
    page=requested_page,
    page_size=page_size,
)

max_page = max(1, -(-total // page_size)) if total else 1
# Clamp the page to the filtered result set.
if requested_page > max_page:
    st.session_state["hist_page"] = max_page
    st.rerun()

st.session_state["hist_page"] = min(max(requested_page, 1), max_page)
page = int(st.session_state["hist_page"])

head1, head2, head3 = st.columns([2, 1, 1])
with head1:
    st.caption(f"{total} record(s) · page {page} of {max_page}")
with head2:
    if records:
        st.download_button(
            "Export page as CSV",
            data=analyses_to_csv(records),
            file_name="ecolens_history.csv",
            mime="text/csv",
        )
with head3:
    if total > page_size:
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("← Prev", disabled=(page <= 1), key="hist_prev"):
                st.session_state["hist_page"] = max(1, page - 1)
                st.rerun()
        with col_b:
            if st.button("Next →", disabled=(page >= max_page), key="hist_next"):
                st.session_state["hist_page"] = min(max_page, page + 1)
                st.rerun()

if not records:
    empty_state(
        "No analyses found",
        "Try adjusting the search, filters or date range, or run a new analysis "
        "in the Image Intelligence workspace.",
    )
    st.stop()

selected_id = st.session_state.get("selected_analysis")

# ---------------------------------------------------------------------------
# Record list
# ---------------------------------------------------------------------------
for record in records:
    with st.container(border=True):
        c1, c2, c3 = st.columns([1, 3, 2])
        with c1:
            img = _load_file(record.stored_image_path)
            if img:
                st.image(img, width="stretch")
            else:
                st.markdown(
                    "<div class='ec-empty' style='padding:.8rem'>image<br/>missing</div>",
                    unsafe_allow_html=True,
                )
        with c2:
            st.markdown(f"**{record.predicted_class or 'No prediction'}**")
            st.markdown(
                f"<span class='ec-muted'>{record.created_at:%Y-%m-%d %H:%M UTC} · "
                f"{record.original_filename or 'unknown file'}</span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"{confidence_badge(record.uncertainty_status)} "
                f"<span class='ec-muted'>material: {record.material_category or 'unknown'}</span>",
                unsafe_allow_html=True,
            )
            if record.user_feedback:
                st.caption(
                    f"Feedback on file: `{record.user_feedback.verdict}`"
                    + (
                        f" → {record.user_feedback.material_correction}"
                        if record.user_feedback.material_correction
                        else ""
                    )
                )
        with c3:
            if st.button("View detail", key=f"view_{record.public_id}"):
                st.session_state["selected_analysis"] = record.public_id
                st.rerun()
            if st.button("Delete", key=f"del_{record.public_id}"):
                st.session_state["confirm_delete"] = record.public_id

# ---------------------------------------------------------------------------
# Delete confirmation
# ---------------------------------------------------------------------------
confirm = st.session_state.get("confirm_delete")
if confirm:
    st.markdown("---")
    callout(
        f"Delete analysis `{confirm}` and its stored image/explanation? "
        "This cannot be undone.",
        "caution",
    )
    cc1, cc2 = st.columns([1, 3])
    with cc1:
        if st.button("Confirm delete", type="primary"):
            repository.delete_analysis(confirm)
            if st.session_state.get("selected_analysis") == confirm:
                st.session_state.pop("selected_analysis", None)
            st.session_state.pop("confirm_delete", None)
            st.session_state["hist_page"] = 1
            st.success("Deleted.")
            st.rerun()
    with cc2:
        if st.button("Cancel"):
            st.session_state.pop("confirm_delete", None)
            st.rerun()

# ---------------------------------------------------------------------------
# Detail view
# ---------------------------------------------------------------------------
if not selected_id:
    st.stop()

# Always re-fetch the record by its id so the detail view can never display a
# stale result from a different record.
record = repository.get_by_public_id(selected_id)
st.markdown("---")
if record is None:
    callout("That analysis no longer exists.", "warning")
    st.stop()

decision = decision_for_record(record)

section("Analysis detail", f"Public ID `{record.public_id}`")
st.caption(
    f"Captured {record.created_at:%Y-%m-%d %H:%M:%S UTC} ·"
    f" {record.model_name or 'model unknown'} · stored locally on this machine."
)

# --- A. Original image and explanation --------------------------------------
st.markdown("### A. Original image and explanation")
img = _load_file(record.stored_image_path)
expl = _load_file(record.explanation_path)
show_overlay = st.checkbox("Show Grad-CAM overlay next to original", value=True)
a1, a2 = st.columns(2)
with a1:
    if img:
        st.image(img, caption="Original image", width="stretch")
    else:
        callout("The stored image file is missing on disk.", "caution")
with a2:
    if expl is not None and show_overlay:
        st.image(expl, caption=f"{record.explanation_method or 'Grad-CAM'} overlay", width="stretch")
    elif expl is None and record.explanation_available:
        callout("The stored explanation file is missing on disk.", "caution")
    else:
        st.markdown("<div class='ec-muted'>No explanation was captured for this record.</div>", unsafe_allow_html=True)

callout(
    "Grad-CAM highlights image regions that most influenced the model's "
    "prediction for a class. It is an attention aid, not a verification that the "
    "object or its material is correct.",
    "info",
)

# --- B. Prediction results ---------------------------------------------------
st.markdown("### B. Prediction results")
status_readable = u.STATUS_LABELS.get(record.uncertainty_status or u.UNKNOWN, "Unknown")
st.markdown(
    f"{confidence_badge(record.uncertainty_status)} "
    f"**{status_readable}**"
    f"<span class='ec-muted'> · score type: {record.score_type or 'unknown'}</span>",
    unsafe_allow_html=True,
)
if record.confidence_score is not None:
    st.markdown(
        f"**Top score:** `{record.confidence_score:.3f}` *(uncalibrated softmax "
        "score — not a probability, not verified accuracy)*"
    )
    if len(record.predictions) >= 2:
        gap = record.predictions[0].raw_score - record.predictions[1].raw_score
        st.markdown(f"**Top-two score difference:** `{gap:.3f}`")
else:
    st.markdown("**Top score:** not recorded")

st.markdown("**Ranked predictions**")
if record.predictions:
    _top = record.predictions[0].raw_score or 0.001
    # manual ranked bars (no chart dependency needed)
    rows = ["<table class='ec-table'><thead><tr><th>Rank</th><th>Recognised object</th>"
            "<th>Score (softmax)</th><th>Material hint</th></tr></thead><tbody>"]
    for p in record.predictions:
        pct = min(100.0, 100.0 * (p.raw_score / _top)) if _top else 0.0
        rows.append(
            f"<tr><td>{p.rank}</td><td>{p.class_name}</td>"
            f"<td><div style='background:#1B4332;border-radius:4px;height:16px;"
            f"width:{pct:.0f}%'></div><small>{p.raw_score:.3f}</small></td>"
            f"<td>{p.material_slug or 'no hint'}</td></tr>"
        )
    rows.append("</tbody></table>")
    st.markdown("".join(rows), unsafe_allow_html=True)
else:
    st.write("No prediction candidates were recorded.")

if record.confidence_score is not None and record.confidence_score < SETTINGS.confidence_threshold:
    callout(
        f"The top score ({record.confidence_score:.3f}) is below the operating "
        f"threshold ({SETTINGS.confidence_threshold:.2f}). The model gives only "
        "weak, competing evidence — treat this as 'Unable to identify reliably'.",
        "caution",
    )

with st.expander("Why this was flagged / confidence explanation", expanded=False):
    if record.uncertainty_status == u.CONFIDENT:
        st.write(
            "The model clearly favoured one object class. Confidence heuristics "
            "(top score, score margin, entropy) are operational safeguards — "
            "they are not measured accuracy."
        )
    elif record.uncertainty_status == u.UNCERTAIN:
        st.write(
            "The model showed competing object categories or the recognised "
            "object is outside EcoLens's material map, so no material claim was "
            "made. Review the image manually."
        )
    else:
        st.write(
            "The model could not identify the object reliably from this image. "
            "No material claim is made."
        )
    if decision is not None and decision.confidence_note:
        st.markdown(f"<div class='ec-muted'>{decision.confidence_note}</div>", unsafe_allow_html=True)

st.markdown(
    "Confidence state is derived from **heuristics** (score + margin + entropy). "
    "It is **not** a measured accuracy or a calibration."
)

# --- C. Material evidence ----------------------------------------------------
st.markdown("### C. Material evidence")
if decision is not None:
    render_decision(decision)
    mat = decision.material_category or record.material_category or "unknown"
    st.markdown(
        f"**Stored material category:** `{mat}`"
        + (
            ""
            if decision.material_category
            else " — displayed as *unknown*: no independently supported material classification exists for this record."
        )
    )
else:
    st.write("No material evidence was recorded.")

# --- D. Recommended next action and guidance ---------------------------------
st.markdown("### D. Recommended next action")
if record.guidance:
    g = record.guidance
    st.markdown(f"**{g.headline}**")
    if g.disposal_guidance:
        st.markdown("**Sorting & disposal guidance**")
        st.write(g.disposal_guidance)
    if g.reuse_guidance:
        st.markdown("**Reuse guidance**")
        st.write(g.reuse_guidance)
    if g.safety_notes:
        callout(g.safety_notes, "warning")
    if g.source_reference:
        st.markdown(
            f"<span class='ec-muted'>General reference: {g.source_reference} · "
            f"scope: {g.geographic_scope or 'general'}</span>",
            unsafe_allow_html=True,
        )
    callout(
        "This is general reuse/disposal guidance, not verified local recycling "
        "rules. Confirm with your municipality before disposing.",
        "info",
    )
else:
    st.write("No guidance snapshot was recorded.")

# --- Model metadata ----------------------------------------------------------
with st.expander("Model & metadata", expanded=False):
    meta = [
        ("Model name", record.model_name),
        ("Model key", record.model_key),
        ("Weights version", record.model_version),
        ("Score type", record.score_type),
        ("Inference time", f"{record.inference_duration_ms:.1f} ms" if record.inference_duration_ms else "—"),
        ("Image size", f"{record.image_width}×{record.image_height}px" if record.image_width else "—"),
        ("SHA-256", record.image_sha256),
        ("Explanation captured", str(record.explanation_available)),
        ("Recorded error", record.error_status),
    ]
    st.markdown(
        "<table class='ec-table'>"
        + "".join(f"<tr><td>{k}</td><td>{v or '—'}</td></tr>" for k, v in meta)
        + "</table>",
        unsafe_allow_html=True,
    )

# --- Feedback (record-scoped; can never cross to another analysis) ----------
st.markdown("---")
render_feedback_widget(record.public_id, key_prefix="detail_fb_")
for other in record.feedback:
    if other.source != "user":
        st.caption(f"Additional record: `{other.source}` · {other.verdict}")

b1,b2 = st.columns([1,4])
with b1:
    if st.button("Delete this record", key="del_detail"):
        st.session_state["confirm_delete"] = record.public_id
with b2:
    if st.button("Close detail", key="hist_close_detail"):
        st.session_state.pop("selected_analysis", None)
        st.rerun()