"""Design system: colour tokens, typography and global CSS.

A restrained environmental-tech aesthetic: neutral off-white canvas, deep
forest-green accents, charcoal text, clear hierarchy.
"""

from __future__ import annotations

import streamlit as st

COLORS = {
    "canvas": "#F6F5F1",
    "surface": "#FFFFFF",
    "ink": "#1F2421",
    "ink_soft": "#4A524C",
    "forest": "#1B4332",
    "forest_soft": "#2D6A4F",
    "moss": "#40916C",
    "accent": "#52B788",
    "line": "#E2E0D8",
    "warning": "#B45309",
    "danger": "#9B2226",
    "info_bg": "#EAF4EE",
    "warn_bg": "#FBF3E6",
}

_FONT_STACK = (
    '-apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, Helvetica, Arial, sans-serif'
)

_CSS = f"""
<style>
:root {{
    --ec-canvas: {COLORS["canvas"]};
    --ec-surface: {COLORS["surface"]};
    --ec-ink: {COLORS["ink"]};
    --ec-ink-soft: {COLORS["ink_soft"]};
    --ec-forest: {COLORS["forest"]};
    --ec-forest-soft: {COLORS["forest_soft"]};
    --ec-moss: {COLORS["moss"]};
    --ec-accent: {COLORS["accent"]};
    --ec-line: {COLORS["line"]};
    --ec-warning: {COLORS["warning"]};
    --ec-danger: {COLORS["danger"]};
}}

html, body, [class*="css"] {{
    font-family: {_FONT_STACK};
    color: var(--ec-ink);
}}

.stApp {{
    background:
        radial-gradient(1200px 500px at 85% -10%, #E8F0EA 0%, rgba(232,240,234,0) 60%),
        var(--ec-canvas);
}}

/* Typography */
h1, h2, h3, h4 {{ color: var(--ec-forest); letter-spacing: -0.01em; }}
h1 {{ font-weight: 700; }}
.ec-eyebrow {{
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--ec-moss);
}}
.ec-muted {{ color: var(--ec-ink-soft); font-size: 0.9rem; }}

/* Cards */
.ec-card {{
    background: var(--ec-surface);
    border: 1px solid var(--ec-line);
    border-radius: 14px;
    padding: 1.1rem 1.25rem;
    box-shadow: 0 1px 2px rgba(27, 67, 50, 0.04);
    margin-bottom: 0.9rem;
}}
.ec-card--accent {{ border-left: 4px solid var(--ec-forest); }}
.ec-card__title {{ font-weight: 650; color: var(--ec-forest); margin-bottom: .35rem; }}

/* Hero */
.ec-hero {{
    background: linear-gradient(135deg, var(--ec-forest) 0%, var(--ec-forest-soft) 100%);
    border-radius: 18px;
    padding: 2.2rem 2rem;
    color: #F1F7F3;
    margin-bottom: 1.4rem;
}}
.ec-hero h1 {{ color: #FFFFFF; margin: 0 0 .5rem 0; font-size: 2.1rem; }}
.ec-hero p {{ color: #D8E7DD; margin: 0; font-size: 1.02rem; max-width: 46ch; }}

/* Badges */
.ec-badge {{
    display: inline-block;
    padding: 0.18rem 0.6rem;
    border-radius: 999px;
    font-size: 0.76rem;
    font-weight: 600;
    border: 1px solid transparent;
}}
.ec-badge--confident {{ background: #E4F1E8; color: var(--ec-forest); border-color: #CBE3D3; }}
.ec-badge--uncertain {{ background: #FBF3E6; color: var(--ec-warning); border-color: #F0DFC2; }}
.ec-badge--low {{ background: #FBE9E9; color: var(--ec-danger); border-color: #F2CFCF; }}
.ec-badge--info {{ background: #EAF4EE; color: var(--ec-forest-soft); border-color: #CFE6D8; }}

/* Metric strip */
.ec-metric {{ text-align: left; }}
.ec-metric__value {{ font-size: 1.6rem; font-weight: 700; color: var(--ec-forest); }}
.ec-metric__label {{ font-size: 0.78rem; color: var(--ec-ink-soft); text-transform: uppercase; letter-spacing: .08em; }}

/* Callouts */
.ec-callout {{
    border-radius: 12px;
    padding: .85rem 1rem;
    font-size: .92rem;
    margin: .5rem 0;
}}
.ec-callout--info {{ background: var(--ec-info_bg, {COLORS["info_bg"]}); border: 1px solid #CFE6D8; color: var(--ec-forest); }}
.ec-callout--caution {{ background: {COLORS["warn_bg"]}; border: 1px solid #F0DFC2; color: #7A4E12; }}
.ec-callout--warning {{ background: #FBE9E9; border: 1px solid #F2CFCF; color: var(--ec-danger); }}

/* Buttons */
.stButton > button {{
    border-radius: 10px;
    font-weight: 600;
    border: 1px solid var(--ec-forest);
}}
.stButton > button[kind="primary"] {{
    background: var(--ec-forest);
    color: #fff;
}}
.stButton > button[kind="primary"]:hover {{ background: var(--ec-forest-soft); }}

/* Sidebar */
[data-testid="stSidebar"] {{
    background: #FBFBF8;
    border-right: 1px solid var(--ec-line);
}}

/* Tables */
.ec-table {{ width: 100%; border-collapse: collapse; font-size: .9rem; }}
.ec-table th {{ text-align: left; color: var(--ec-ink-soft); font-weight: 600; padding: .4rem .5rem; border-bottom: 1px solid var(--ec-line); }}
.ec-table td {{ padding: .4rem .5rem; border-bottom: 1px solid #F0EFE9; }}

/* Empty state */
.ec-empty {{
    border: 1px dashed var(--ec-line);
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
    color: var(--ec-ink-soft);
    background: #FCFCFA;
}}

/* Focus visibility for accessibility */
:focus-visible {{ outline: 3px solid var(--ec-accent); outline-offset: 2px; }}

/* Tighten Streamlit default top padding */
.block-container {{ padding-top: 2.2rem; }}
</style>
"""


def inject_global_styles() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def page_config(title: str = "EcoLens AI") -> None:
    st.set_page_config(
        page_title=f"{title} · EcoLens AI",
        page_icon="♻️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
