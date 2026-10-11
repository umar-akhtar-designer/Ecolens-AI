"""Smoke tests: every page renders without raising an exception.

Uses Streamlit's AppTest harness. These tests do not require the model or any
network access — they render the pages against an empty (isolated) database.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

_ROOT = Path(__file__).resolve().parent.parent

PAGES = [
    "app.py",
    "pages/1_Image_Intelligence.py",
    "pages/2_Analysis_History.py",
    "pages/3_Sustainability_Insights.py",
    "pages/4_Material_Guide.py",
    "pages/5_About_Methodology.py",
]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exception(page):
    app = AppTest.from_file(str(_ROOT / page), default_timeout=120).run()
    assert not app.exception, [e.value for e in app.exception]
