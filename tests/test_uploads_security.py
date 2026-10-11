"""Tests for untrusted-upload handling and path safety."""

from __future__ import annotations

import pytest
from conftest import make_image_bytes

from src.security.uploads import (
    UploadValidationError,
    delete_stored_file,
    resolve_storage_path,
    safe_extension,
    save_image,
    validate_and_decode,
)


def test_safe_extension_whitelist():
    assert safe_extension("photo.JPG") == "jpg"
    assert safe_extension("a.jpeg") == "jpg"
    assert safe_extension("x.png") == "png"


def test_safe_extension_rejects_unknown():
    with pytest.raises(UploadValidationError):
        safe_extension("malware.exe")


def test_validate_and_decode_rejects_bad_extension():
    with pytest.raises(UploadValidationError):
        validate_and_decode(make_image_bytes(), "file.txt")


def test_validate_and_decode_rejects_corrupt():
    with pytest.raises(UploadValidationError):
        validate_and_decode(b"not-an-image", "file.png")


def test_save_and_delete_roundtrip():
    data = make_image_bytes(64, 64)
    _abs_path, rel_path = save_image(data, "thing.png")
    resolved = resolve_storage_path(rel_path)
    assert resolved.exists()
    assert resolved.read_bytes() == data
    assert delete_stored_file(rel_path) is True
    assert not resolved.exists()


def test_path_traversal_is_blocked():
    with pytest.raises(UploadValidationError):
        resolve_storage_path("../../../../etc/passwd")
    with pytest.raises(UploadValidationError):
        resolve_storage_path("../secrets.txt")


def test_delete_missing_file_returns_false():
    assert delete_stored_file("uploads/2099/01/01/nonexistent.png") is False
