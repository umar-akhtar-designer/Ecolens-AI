"""Sustainability Insights — analytics computed only from saved records."""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analytics.metrics import compute_analytics
from src.ui.bootstrap import setup_page, sidebar_footer
from src.ui.components import callout, empty_state, metric_strip

setup_page("Sustainability Insights")
sidebar_footer()

st.markdown("## Sustainability Insights")
st.markdown(
    "<div class='ec-muted'>Charts below are computed directly from your saved "
    "analyses. No outcome claims (items diverted, emissions) are made because "
    "those are not measured.</div>",
    unsafe_allow_html=True,
)

days = st.slider("Time window (days) for trend chart", 7, 180, 30, step=7)
analytics = compute_analytics(days=days)

if not analytics.has_data:
    empty_state(
        "No data yet",
        "Once you run image analyses, real trends and distributions will appear here. "
        "Nothing is mocked or pre-filled.",
    )
    st.stop()

metric_strip(
    [
        ("Total analyses", str(analytics.total_analyses)),
        ("Confident", str(analytics.confident_count)),
        ("Review flagged", str(analytics.abstention_count)),
        ("No material claim", str(analytics.unsupported_material_count)),
        ("Avg latency", f"{analytics.avg_inference_ms:.0f} ms"),
    ]
)
st.caption(
    f"Confident rate (of all records): {analytics.confident_rate:.0%} · "
    f"review/abstention rate: {analytics.abstention_rate:.0%}. These are counts from "
    "stored records, not model accuracy."
)

if analytics.feedback and analytics.feedback.get("total", 0) > 0:
    fb = analytics.feedback
    st.markdown(
        f"**User-reported feedback:** {fb.get('total', 0)} submission(s) — "
        f"{fb.get('correct', 0)} correct, {fb.get('incorrect', 0)} incorrect, "
        f"{fb.get('unsure', 0)} unsure. "
        f"Touches {fb.get('analyses_with_feedback', 0)} of "
        f"{fb.get('total_analyses', 0)} saved analyses "
        f"(coverage {fb.get('coverage', 0):.0%})."
    )
    callout(
        "Feedback is user-reported and is **not** model accuracy or a scientific "
        "evaluation. It reflects only the analyses users chose to rate.",
        "info",
    )

st.markdown("---")
c1, c2 = st.columns(2)

with c1:
    st.markdown("#### Predicted material distribution")
    if analytics.category_distribution:
        cats = [c for c, _ in analytics.category_distribution]
        counts = [n for _, n in analytics.category_distribution]
        fig = px.bar(
            x=counts,
            y=cats,
            orientation="h",
            labels={"x": "Analyses", "y": "Material category"},
            color_discrete_sequence=["#1B4332"],
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=340,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.write("No material categories recorded yet.")

with c2:
    st.markdown("#### Confidence state breakdown")
    if analytics.uncertainty_breakdown:
        labels = [s for s, _ in analytics.uncertainty_breakdown]
        values = [n for _, n in analytics.uncertainty_breakdown]
        fig = px.pie(
            names=labels,
            values=values,
            hole=0.5,
            color_discrete_sequence=["#2D6A4F", "#B45309", "#9B2226"],
        )
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")
    else:
        st.write("No confidence states recorded yet.")

st.markdown("#### Analyses over time")
if analytics.over_time:
    days_axis = [d for d, _ in analytics.over_time]
    counts = [n for _, n in analytics.over_time]
    fig = go.Figure(
        go.Scatter(
            x=days_axis,
            y=counts,
            mode="lines+markers",
            line=dict(color="#40916C", width=3),
            fill="tozeroy",
            fillcolor="rgba(82,183,136,0.15)",
        )
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Date",
        yaxis_title="Analyses",
    )
    st.plotly_chart(fig, width="stretch")
else:
    st.write(f"No analyses in the last {days} days.")

c3, c4 = st.columns(2)
with c3:
    st.markdown("#### Most frequently recognised objects")
    if analytics.top_predicted_classes:
        for name, n in analytics.top_predicted_classes:
            st.markdown(
                f"<div class='ec-card' style='margin-bottom:.4rem;display:flex;"
                f"justify-content:space-between'><span>{name}</span><b>{n}</b></div>",
                unsafe_allow_html=True,
            )
    else:
        st.write("No predictions recorded yet.")

with c4:
    st.markdown("#### Processing time trend")
    if analytics.processing_time_series:
        xs = [t for t, _ in analytics.processing_time_series]
        ys = [ms for _, ms in analytics.processing_time_series]
        fig = px.line(x=xs, y=ys, labels={"x": "Time", "y": "ms"})
        fig.update_traces(line_color="#1B4332")
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=300,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.write("No timing data recorded yet.")

if analytics.model_distribution:
    st.markdown("#### Model usage")
    mcols = st.columns(len(analytics.model_distribution))
    for col, (mkey, n) in zip(mcols, analytics.model_distribution):
        with col:
            st.markdown(
                f"<div class='ec-card'><b>{mkey}</b><br/>{n} analyses</div>",
                unsafe_allow_html=True,
            )

if analytics.first_record_at:
    callout(
        f"Data range: {analytics.first_record_at:%Y-%m-%d} to "
        f"{analytics.last_record_at:%Y-%m-%d} · {analytics.distinct_materials} "
        "distinct material categories observed.",
        "info",
    )
