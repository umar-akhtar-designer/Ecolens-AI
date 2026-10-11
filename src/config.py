"""Central configuration for EcoLens AI.

All tunables live here so the application, vision pipeline, database and tests
share a single source of truth. Values may be overridden through environment
variables (see ``.env.example``) without code changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


_STORAGE_DIR = _env_path("ECOLENS_STORAGE_DIR", PROJECT_ROOT / "storage")


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings."""

    project_root: Path = PROJECT_ROOT
    data_dir: Path = field(
        default_factory=lambda: _env_path("ECOLENS_DATA_DIR", PROJECT_ROOT / "data")
    )
    storage_dir: Path = field(default_factory=lambda: _STORAGE_DIR)
    uploads_dir: Path = field(default_factory=lambda: _STORAGE_DIR / "uploads")
    explanations_dir: Path = field(default_factory=lambda: _STORAGE_DIR / "explanations")
    db_path: Path = field(
        default_factory=lambda: _env_path("ECOLENS_DB_PATH", _STORAGE_DIR / "ecolens.db")
    )
    knowledge_path: Path = field(
        default_factory=lambda: _env_path(
            "ECOLENS_KNOWLEDGE_PATH", PROJECT_ROOT / "data" / "material_knowledge.json"
        )
    )

    # Upload / validation limits
    max_upload_mb: int = _env_int("ECOLENS_MAX_UPLOAD_MB", 12)
    min_image_dimension: int = _env_int("ECOLENS_MIN_DIM", 32)
    max_image_dimension: int = _env_int("ECOLENS_MAX_DIM", 12000)
    allowed_extensions: tuple[str, ...] = (
        "jpg",
        "jpeg",
        "png",
        "webp",
        "bmp",
        "gif",
        "tif",
        "tiff",
    )
    allowed_mime_types: tuple[str, ...] = (
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/bmp",
        "image/gif",
        "image/tiff",
    )

    # Model / inference
    model_key: str = os.environ.get("ECOLENS_MODEL", "mobilenet_v3_large")
    top_k: int = _env_int("ECOLENS_TOP_K", 5)
    confidence_threshold: float = _env_float("ECOLENS_CONFIDENCE_THRESHOLD", 0.35)
    uncertainty_entropy_threshold: float = _env_float("ECOLENS_ENTROPY_THRESHOLD", 0.65)
    allow_weight_download: bool = os.environ.get("ECOLENS_ALLOW_DOWNLOAD", "1") not in (
        "0",
        "false",
        "False",
    )
    device: str = os.environ.get("ECOLENS_DEVICE", "auto")

    # History / analytics
    history_page_size: int = _env_int("ECOLENS_PAGE_SIZE", 8)
    max_history_rows: int = _env_int("ECOLENS_MAX_HISTORY", 100000)

    # Batch analysis
    max_batch_files: int = _env_int("ECOLENS_MAX_BATCH_FILES", 12)

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def ensure_directories(self) -> None:
        """Create writable directories. Safe to call repeatedly."""
        for directory in (self.data_dir, self.storage_dir, self.uploads_dir, self.explanations_dir):
            directory.mkdir(parents=True, exist_ok=True)


SETTINGS = Settings()
