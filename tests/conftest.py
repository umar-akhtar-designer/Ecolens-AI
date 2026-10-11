"""Shared pytest fixtures.

Environment variables are set at import time, *before* ``src`` is imported, so
the application uses an isolated temporary storage directory and database.
"""

from __future__ import annotations

import os
import tempfile
from io import BytesIO
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="ecolens-tests-"))
os.environ.setdefault("ECOLENS_STORAGE_DIR", str(_TMP / "storage"))
os.environ.setdefault("ECOLENS_DB_PATH", str(_TMP / "storage" / "test.db"))
os.environ.setdefault("ECOLENS_DEVICE", "cpu")
os.environ.setdefault("ECOLENS_CONFIDENCE_THRESHOLD", "0.35")

import numpy as np
import pytest
from PIL import Image

from src.database.initialization import init_db, reset_caches, reset_db


@pytest.fixture(scope="session", autouse=True)
def _init() -> None:
    init_db()


@pytest.fixture()
def fresh_db():
    reset_caches()
    reset_db()
    yield
    reset_db()


def make_image_bytes(width=320, height=320, color=(120, 160, 120), fmt="PNG", seed=None) -> bytes:
    if seed is not None:
        rng = np.random.default_rng(seed)
        arr = (rng.random((height, width, 3)) * 255).astype("uint8")
    else:
        arr = np.zeros((height, width, 3), dtype="uint8")
        arr[:, :] = color
    buf = BytesIO()
    Image.fromarray(arr).save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture()
def sample_image_bytes() -> bytes:
    return make_image_bytes(seed=42)


@pytest.fixture()
def uniform_image_bytes() -> bytes:
    return make_image_bytes(color=(200, 30, 30))
