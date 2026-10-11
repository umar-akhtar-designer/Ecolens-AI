"""One-call page bootstrap shared by every Streamlit page.

Order matters: ``set_page_config`` must run before any other Streamlit command,
so this must be the first thing every page calls.
"""

from __future__ import annotations

import streamlit as st

from ..config import SETTINGS
from ..database.initialization import init_db
from ..logging_config import configure_logging, get_logger
from .theme import inject_global_styles, page_config


def setup_page(title: str = "EcoLens AI") -> None:
    page_config(title)
    configure_logging(log_file=SETTINGS.storage_dir / "ecolens.log")
    try:
        init_db()
    except Exception as exc:  # pragma: no cover - surfaced in UI
        get_logger("bootstrap").warning("Database init failed: %s", exc)
    inject_global_styles()


def sidebar_footer() -> None:
    with st.sidebar:
        st.markdown("---")
        st.markdown(
            "<div class='ec-muted'>EcoLens AI · Image intelligence for "
            "material awareness. Predictions are model estimates, not "
            "certifications.</div>",
            unsafe_allow_html=True,
        )
