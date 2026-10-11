"""Untrusted-upload handling: validation, safe storage, traversal protection.

Design rules:
* The declared extension and MIME type are checked, but the *decoded image* is
  the source of truth (a renamed .exe fails to decode).
* Stored filenames are generated (date partition + random hex), never taken
  from user input.
* Resolving a stored path is confined to the storage directory.
* Uploaded files are never executed or served from a public directory.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from ..config import SETTINGS
from ..logging_config import get_logger
from ..vision.preprocessing import DecodedImage, ImageDecodeError, decode_image_bytes

logger = get_logger("security.uploads")


class UploadValidationError(ValueError):
    """Raised when an upload fails validation."""


def safe_extension(filename: str) -> str:
    """Return a normalised, allow-listed extension (no leading dot)."""
    ext = Path(filename or "").suffix.lower().lstrip(".")
    if ext == "jpeg":
        ext = "jpg"
    if ext == "tif":
        ext = "tiff"
    if ext not in SETTINGS.allowed_extensions:
        raise UploadValidationError(
            f"Unsupported file extension '.{ext}'. Allowed: "
            f"{', '.join(SETTINGS.allowed_extensions)}."
        )
    return ext


def validate_and_decode(data: bytes, filename: str, mime_type: str | None = None) -> DecodedImage:
    """Validate extension/MIME/size and decode the image.

    Returns a :class:`DecodedImage`; raises :class:`UploadValidationError` on
    any failure with a user-safe message.
    """
    safe_extension(filename)  # raises for bad extension

    if mime_type and mime_type not in SETTINGS.allowed_mime_types:
        # Browsers sometimes omit or mislabel MIME; warn but rely on decode.
        logger.info("Unexpected MIME type", extra={"event": "mime", "error": mime_type})

    if len(data) > SETTINGS.max_upload_bytes:
        raise UploadValidationError(f"File exceeds the {SETTINGS.max_upload_mb} MB limit.")

    try:
        return decode_image_bytes(data)
    except ImageDecodeError as exc:
        raise UploadValidationError(str(exc)) from exc


def _date_partition() -> str:
    return datetime.now(timezone.utc).strftime("%Y/%m/%d")


def save_image(data: bytes, original_filename: str) -> tuple[str, str]:
    """Persist image bytes under a generated name.

    Returns ``(absolute_path, relative_path)`` where ``relative_path`` is
    relative to the storage directory (safe to store in the DB).
    """
    ext = safe_extension(original_filename)
    rel_dir = Path("uploads") / _date_partition()
    abs_dir = SETTINGS.storage_dir / rel_dir
    abs_dir.mkdir(parents=True, exist_ok=True)

    token = uuid.uuid4().hex
    rel_path = rel_dir / f"{token}.{ext}"
    abs_path = SETTINGS.storage_dir / rel_path

    tmp_path = abs_path.with_suffix(abs_path.suffix + ".tmp")
    tmp_path.write_bytes(data)
    os.replace(tmp_path, abs_path)  # atomic within the same filesystem

    return str(abs_path), str(rel_path)


def save_explanation(image_bytes: bytes, ext: str = "png") -> tuple[str, str]:
    """Persist a rendered explanation overlay as PNG.

    Returns ``(absolute_path, relative_path)``.
    """
    rel_dir = Path("explanations") / _date_partition()
    abs_dir = SETTINGS.storage_dir / rel_dir
    abs_dir.mkdir(parents=True, exist_ok=True)

    token = uuid.uuid4().hex
    rel_path = rel_dir / f"{token}.{ext}"
    abs_path = SETTINGS.storage_dir / rel_path
    abs_path.write_bytes(image_bytes)
    return str(abs_path), str(rel_path)


def resolve_storage_path(relative_path: str) -> Path:
    """Resolve a DB-stored relative path, rejecting traversal attempts."""
    if not relative_path:
        raise UploadValidationError("Missing stored path.")
    candidate = (SETTINGS.storage_dir / relative_path).resolve()
    storage_root = SETTINGS.storage_dir.resolve()
    if storage_root not in candidate.parents and candidate != storage_root:
        raise UploadValidationError("Resolved path escapes the storage directory.")
    return candidate


def delete_stored_file(relative_path: str | None) -> bool:
    """Delete a stored file if it exists and is inside the storage dir."""
    if not relative_path:
        return False
    try:
        path = resolve_storage_path(relative_path)
    except UploadValidationError:
        logger.warning("Refused to delete path outside storage: %s", relative_path)
        return False
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False
    except OSError as exc:  # pragma: no cover
        logger.warning("Could not delete file: %s", exc)
        return False
