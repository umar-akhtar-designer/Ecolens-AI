"""Image Intelligence Workspace — the core analysis workflow."""

from __future__ import annotations

import streamlit as st

from src.analytics.export import batch_outcome_to_csv
from src.config import SETTINGS
from src.services.analysis_service import AnalysisBundle, run_analysis, run_batch
from src.ui.bootstrap import setup_page, sidebar_footer
from src.ui.components import (
    callout,
    confidence_badge,
    metric_strip,
    prediction_table,
    render_decision,
    render_feedback_widget,
    render_guidance,
    section,
)
from src.vision.explanations import blend_overlay, heatmap_image
from src.vision.preprocessing import ImageDecodeError, decode_image_bytes

setup_page("Image Intelligence")
sidebar_footer()

st.markdown("## Image Intelligence Workspace")
st.markdown(
    "<div class='ec-muted'>Upload an image, run a real analysis, and receive "
    "honest predictions with practical guidance.</div>",
    unsafe_allow_html=True,
)

with st.expander("Model in use & limitations", expanded=False):
    st.markdown(
        f"- **Model:** `{SETTINGS.model_key}` — an ImageNet-1k **object** classifier "
        "served via torchvision.\n"
        "- **Material labels are derived hints**, not direct material measurements.\n"
        f"- **Upload limit:** {SETTINGS.max_upload_mb} MB per file. "
        f"**Confidence threshold:** {SETTINGS.confidence_threshold:.2f}.\n"
        "- Softmax scores are **uncalibrated**; they are not true probabilities.\n"
        "- Images are stored **locally** on this machine only."
    )

state = st.session_state

mode = st.radio(
    "Analysis mode",
    ["Single image", "Batch analysis"],
    horizontal=True,
    key="analysis_mode",
)


def _reset_single() -> None:
    for key in ("single_bundle",):
        state.pop(key, None)


def _render_result(bundle: AnalysisBundle, *, allow_feedback: bool) -> None:
    result = bundle.result
    if result is None:
        return
    top = result.top1
    metric_strip(
        [
            ("Prediction", top.class_name if top else "—"),
            ("Score", f"{top.raw_score:.3f}" if top else "—"),
            ("Processing", f"{result.inference_duration_ms:.0f} ms"),
            ("Device", result.device),
        ]
    )
    st.markdown(
        f"{confidence_badge(result.uncertainty.status)} "
        f"<span class='ec-muted'>· {result.image_width}×{result.image_height}px · "
        f"{result.model_name}</span>",
        unsafe_allow_html=True,
    )
    if result.uncertainty:
        st.markdown(
            f"<div class='ec-muted'>{result.uncertainty.explanation}</div>", unsafe_allow_html=True
        )
        if result.uncertainty.abstain:
            callout(
                "EcoLens is abstaining from a material claim for this image. "
                "Treat the recognition as a weak suggestion only.",
                "caution",
            )

    tab_decision, tab_pred, tab_expl, tab_guide, tab_meta = st.tabs(
        ["Decision support", "Predictions", "Visual explanation", "Guidance", "Model & metadata"]
    )

    with tab_decision:
        render_decision(bundle.decision)

    with tab_pred:
        st.markdown("**Top predictions (softmax scores, uncalibrated)**")
        prediction_table(result.predictions)
        if result.is_direct_material_classifier:
            callout("This model directly classifies materials.", "info")
        else:
            callout(
                "Scores are softmax outputs from an object classifier; they are not "
                "calibrated probabilities and material hints are derived, not measured.",
                "info",
            )

    with tab_expl:
        if result.explanation is not None:
            alpha = st.slider(
                "Heatmap opacity", min_value=0.0, max_value=1.0, value=0.45, step=0.05
            )
            original_bytes = bundle.image_bytes
            try:
                pil = decode_image_bytes(original_bytes).image if original_bytes else None
            except ImageDecodeError:
                pil = None
            c1a, c1b, c1c = st.columns(3)
            with c1a:
                if original_bytes:
                    st.image(original_bytes, caption="Original", width="stretch")
            with c1b:
                if pil is not None:
                    st.image(
                        blend_overlay(pil, result.explanation.heatmap, alpha),
                        caption=f"{result.explanation.method} overlay",
                        width="stretch",
                    )
            with c1c:
                st.image(
                    heatmap_image(result.explanation.heatmap),
                    caption="Raw saliency",
                    width="stretch",
                )
            callout(result.explanation.note, "info")
        elif result.explanation_error:
            callout(
                f"Visual explanation unavailable: {result.explanation_error}. "
                "The prediction above is still valid.",
                "caution",
            )
        else:
            st.write("No explanation was generated.")

    with tab_guide:
        if bundle.guidance is not None:
            render_guidance(bundle.guidance)
        for w in result.warnings:
            callout(w, "caution")

    with tab_meta:
        meta = [
            ("Model key", result.model_key),
            ("Model name", result.model_name),
            ("Weights version", result.model_version),
            ("Score type", result.score_type),
            ("Device", result.device),
            ("Inference time", f"{result.inference_duration_ms:.1f} ms"),
            ("Image size", f"{result.image_width}×{result.image_height}"),
            ("SHA-256", result.image_sha256),
            ("Direct material classifier", str(result.is_direct_material_classifier)),
        ]
        st.markdown(
            "<table class='ec-table'>"
            + "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in meta)
            + "</table>",
            unsafe_allow_html=True,
        )

    if allow_feedback and bundle.record is not None:
        st.markdown("---")
        st.success(f"Saved to history as `{bundle.record.public_id}`.")
        render_feedback_widget(bundle.record.public_id)
        if st.button("Open analysis detail"):
            st.switch_page("pages/2_Analysis_History.py")


def render_single() -> None:
    uploaded = st.file_uploader(
        "Upload an image (JPG, PNG, WEBP, BMP, GIF, TIFF)",
        type=list(SETTINGS.allowed_extensions),
        accept_multiple_files=False,
        key="uploader_single",
    )

    if uploaded is not None:
        raw = uploaded.getvalue()
        size_mb = len(raw) / (1024 * 1024)
        st.caption(f"Selected: **{uploaded.name}** · {size_mb:.2f} MB")
        preview_ok = size_mb * 1024 * 1024 <= SETTINGS.max_upload_bytes
        if not preview_ok:
            callout(f"File exceeds the {SETTINGS.max_upload_mb} MB limit.", "warning")

        col_preview, col_meta = st.columns([2, 1])
        with col_preview:
            if preview_ok:
                st.image(raw, caption="Preview", width="stretch")
        with col_meta:
            if preview_ok:
                try:
                    decoded = decode_image_bytes(raw)
                    metric_strip(
                        [("Width", f"{decoded.width}px"), ("Height", f"{decoded.height}px")]
                    )
                    st.caption(f"SHA-256: `{decoded.sha256[:16]}…`")
                except ImageDecodeError as exc:
                    callout(f"Preview unavailable: {exc}", "warning")
                    preview_ok = False

        b1, b2 = st.columns([1, 1])
        with b1:
            analyze_clicked = st.button("Analyze image", type="primary", disabled=not preview_ok)
        with b2:
            if st.button("Clear / start over"):
                _reset_single()
                st.rerun()

        if analyze_clicked and preview_ok:
            with st.spinner("Loading model and analysing image…"):
                bundle = run_analysis(raw, uploaded.name, mime_type=uploaded.type, persist=True)
            state["single_bundle"] = bundle

    bundle = state.get("single_bundle")
    if bundle is not None and not bundle.ok:
        callout(bundle.error or "Analysis failed.", "warning")
        if bundle.image_bytes and st.button("Retry analysis"):
            retry = run_analysis(bundle.image_bytes, bundle.filename, persist=True)
            state["single_bundle"] = retry
            st.rerun()
    elif bundle is not None and bundle.ok:
        st.markdown("---")
        section("Analysis result", "All values below are produced by the running model.")
        if bundle.error:
            callout(bundle.error, "warning")
        _render_result(bundle, allow_feedback=True)
    else:
        st.markdown(
            "<div class='ec-muted'>Awaiting an image to analyse.</div>", unsafe_allow_html=True
        )


def render_batch() -> None:
    files = st.file_uploader(
        f"Upload up to {SETTINGS.max_batch_files} images",
        type=list(SETTINGS.allowed_extensions),
        accept_multiple_files=True,
        key="uploader_batch",
    )
    callout(
        "Batch runs the same real pipeline per image. Each file is analysed "
        "independently; one failure does not stop the batch. Grad-CAM is skipped "
        "in batch mode for speed and can be viewed per item in History.",
        "info",
    )
    if files and st.button("Analyze batch", type="primary"):
        payload = [(f.name, f.getvalue(), f.type) for f in files]
        with st.spinner(f"Analysing {len(payload)} images…"):
            outcome = run_batch(payload, persist=True, with_explanation=False)
        state["batch_outcome"] = outcome

    outcome = state.get("batch_outcome")
    if outcome is None:
        st.markdown(
            "<div class='ec-muted'>Upload images to run a batch analysis.</div>",
            unsafe_allow_html=True,
        )
        return

    st.markdown("---")
    section(
        "Batch summary",
        f"{outcome.succeeded} succeeded · {outcome.failed} failed · "
        f"{len(outcome.skipped)} skipped.",
    )
    if outcome.limit_reached:
        callout(
            f"Only the first {SETTINGS.max_batch_files} files were processed (batch limit).",
            "warning",
        )

    metric_strip(
        [
            ("Processed", str(len(outcome.items))),
            ("Succeeded", str(outcome.succeeded)),
            ("Failed", str(outcome.failed)),
            ("Skipped", str(len(outcome.skipped))),
        ]
    )

    st.download_button(
        "Download results as CSV",
        data=batch_outcome_to_csv(outcome),
        file_name="ecolens_batch.csv",
        mime="text/csv",
    )

    for item in outcome.items:
        icon = "✅" if item.ok else "⚠️"
        with st.expander(f"{icon} {item.filename}", expanded=not item.ok):
            if item.ok and item.result is not None:
                top = item.result.top1
                st.markdown(
                    f"**{top.class_name if top else '—'}** · "
                    f"score `{top.raw_score:.3f}` · "
                    f"{confidence_badge(item.result.uncertainty.status)}",
                    unsafe_allow_html=True,
                )
                render_decision(item.decision)
                if item.record is not None:
                    st.caption(f"Saved as `{item.record.public_id}`")
                    render_feedback_widget(item.record.public_id, key_prefix="batch_fb")
            else:
                callout(item.error or "Analysis failed.", "warning")

    for filename, reason in outcome.skipped:
        callout(f"Skipped {filename}: {reason}", "warning")


if mode == "Single image":
    render_single()
else:
    render_batch()
