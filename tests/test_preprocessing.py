"""Tests for image decoding and preprocessing."""

from __future__ import annotations

from io import BytesIO

import pytest
from conftest import make_image_bytes
from PIL import Image

from src.vision.preprocessing import ImageDecodeError, decode_image_bytes, to_display_array


def test_valid_png_decodes():
    data = make_image_bytes(120, 90)
    decoded = decode_image_bytes(data)
    assert decoded.width == 120
    assert decoded.height == 90
    assert decoded.mode == "RGB"
    assert len(decoded.sha256) == 64


def test_grayscale_converted_to_rgb():
    img = Image.new("L", (64, 64), 128)
    buf = BytesIO()
    img.save(buf, format="PNG")
    decoded = decode_image_bytes(buf.getvalue())
    assert decoded.mode == "RGB"


def test_corrupted_bytes_raise():
    with pytest.raises(ImageDecodeError):
        decode_image_bytes(b"this is not an image at all")


def test_empty_bytes_raise():
    with pytest.raises(ImageDecodeError):
        decode_image_bytes(b"")


def test_oversized_bytes_raise():
    with pytest.raises(ImageDecodeError):
        decode_image_bytes(b"x" * 100, max_bytes=50)


def test_too_small_image_raise():
    with pytest.raises(ImageDecodeError):
        decode_image_bytes(make_image_bytes(10, 10), min_dimension=32)


def test_sha256_is_stable():
    data = make_image_bytes(80, 80)
    assert decode_image_bytes(data).sha256 == decode_image_bytes(data).sha256


def test_display_array_downscales():
    data = make_image_bytes(1000, 500)
    decoded = decode_image_bytes(data)
    arr = to_display_array(decoded.image, max_side=256)
    assert max(arr.shape[0], arr.shape[1]) <= 256
