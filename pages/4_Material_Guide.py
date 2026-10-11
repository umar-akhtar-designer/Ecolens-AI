"""Material Guide — browse the local knowledge base without running inference."""

from __future__ import annotations

import streamlit as st

from src.guidance.engine import get_engine
from src.ui.bootstrap import setup_page, sidebar_footer
from src.ui.components import callout, empty_state

setup_page("Material Guide")
sidebar_footer()

st.markdown("## Material Guide")
st.markdown(
    "<div class='ec-muted'>General reuse and disposal guidance for supported "
    "categories. No image analysis required.</div>",
    unsafe_allow_html=True,
)

engine = get_engine()
meta = engine.metadata
callout(meta.get("disclaimer", ""), "info")

query = st.text_input("Search categories, examples or guidance", value="")
results = engine.search(query)

st.caption(f"{len(results)} category result(s)")

if not results:
    empty_state("No matching categories", "Try a different search term.")

for item in results:
    with st.expander(f"{item.display_name}", expanded=bool(query)):
        st.markdown(f"<div class='ec-muted'>{item.description}</div>", unsafe_allow_html=True)
        cols = st.columns(2)
        with cols[0]:
            st.markdown("**Common examples**")
            st.write(", ".join(item.common_examples) or "—")
            st.markdown("**Reuse**")
            st.write(item.reuse_guidance)
        with cols[1]:
            st.markdown("**Sorting & disposal**")
            st.write(item.disposal_guidance)
            st.markdown("**Conditions affecting recyclability**")
            st.write(item.recyclability_conditions)
        st.markdown("**Contamination notes**")
        st.write(item.contamination_notes)
        callout(item.safety_notes, "caution")
        st.markdown(
            f"<span class='ec-muted'>Scope: {item.geographic_scope} · "
            f"Reviewed: {item.reviewed_at} · Source: {item.source_url}</span>",
            unsafe_allow_html=True,
        )

st.markdown("---")
st.markdown(
    "<div class='ec-muted'>Local rules vary. Confirm the correct stream with your "
    "municipality or facility before relying on this guidance.</div>",
    unsafe_allow_html=True,
)
