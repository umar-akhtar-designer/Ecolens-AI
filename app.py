"""EcoLens AI — Home / Overview.

See Waste Differently. Make Every Material Count.
"""

from __future__ import annotations

import streamlit as st

from src.analytics.metrics import compute_analytics
from src.ui.bootstrap import setup_page, sidebar_footer
from src.ui.components import callout, empty_state, metric_strip

setup_page("Home")
sidebar_footer()

st.markdown(
    """
    <div class="ec-hero">
      <div class="ec-eyebrow" style="color:#9FD8B8">EcoLens AI · Image Intelligence</div>
      <h1>See Waste Differently.</h1>
      <p>Upload a photo of an item and EcoLens AI runs a real computer-vision model to
      identify likely objects, estimate how confident it is, explain what it looked at,
      and connect that to practical reuse and disposal guidance.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([2, 1])
with left:
    st.markdown("### Start an analysis")
    st.write(
        "The core workflow is a real image analysis: upload → validate → preprocess → "
        "infer → explain → guide → save. Removing the vision model breaks the product."
    )
    if st.button("Open the Image Intelligence workspace", type="primary"):
        st.switch_page("pages/1_Image_Intelligence.py")

with right:
    analytics = compute_analytics()
    if analytics.has_data:
        metric_strip(
            [
                ("Analyses", str(analytics.total_analyses)),
                ("Materials", str(analytics.distinct_materials)),
            ]
        )
        st.caption("Live counts from your local database.")
    else:
        st.markdown(
            "<div class='ec-empty'><b>No analyses yet</b><br/>"
            "Run your first image analysis to populate history and insights.</div>",
            unsafe_allow_html=True,
        )

st.markdown("---")

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        "<div class='ec-card ec-card--accent'><div class='ec-card__title'>Real inference</div>"
        "A pretrained PyTorch vision model runs locally on CPU (GPU optional). "
        "No API keys, no paid services.</div>",
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        "<div class='ec-card ec-card--accent'><div class='ec-card__title'>Honest uncertainty</div>"
        "We show raw softmax scores labelled as <em>uncalibrated</em>, plus a "
        "confidence/uncertainty state — never a fake certainty.</div>",
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        "<div class='ec-card ec-card--accent'><div class='ec-card__title'>Explainable</div>"
        "Grad-CAM highlights the image regions that influenced the prediction, "
        "with a clear note on what it does and does not mean.</div>",
        unsafe_allow_html=True,
    )

st.markdown("### How it works")
st.markdown(
    """
1. **Validate & preprocess** — file type, size and real decodability are checked; EXIF
   orientation is corrected and the image is converted to RGB.
2. **Infer** — an ImageNet-pretrained *object* classifier produces top-k predictions.
3. **Qualify** — softmax scores are surfaced as uncalibrated scores; a confidence state
   flags low-confidence or ambiguous results.
4. **Explain** — Grad-CAM overlays where the architecture supports it (with graceful
   fallback if it fails).
5. **Guide** — a local knowledge base maps recognised objects to cautious reuse and
   disposal guidance, with safety warnings.
6. **Persist** — the analysis is stored locally so History and Insights reflect reality.
"""
)

st.markdown("### What it can and cannot do")
callout(
    "Honest scope: EcoLens AI recognises common objects. It does not measure chemical "
    "composition, contamination, purity or recyclability, and material labels are "
    "conservative hints derived from recognised objects — not sensor measurements.",
    "info",
)
st.markdown(
    "- ✅ Recognise familiar objects and flag materials as *possible*.\n"
    "- ✅ Communicate uncertainty and refuse to over-claim.\n"
    "- ❌ Claim an item is recyclable in your area (rules vary locally).\n"
    "- ❌ Estimate environmental impact figures (not measured)."
)

if not analytics.has_data:
    empty_state(
        "Insights will appear here",
        "Once you run analyses, the Sustainability Insights page charts real trends "
        "from your saved records.",
    )
