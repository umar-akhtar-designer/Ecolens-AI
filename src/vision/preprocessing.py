"""Image validation and preprocessing.

Decoding is deliberately defensive: uploaded bytes are untrusted input. We
verify the declared size, attempt a real decode, honour EXIF orientation, and
only then hand a clean RGB image to the model pipeline.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from ..config import SETTINGS

# Guard against decompression bombs (PIL raises DecompressionBombError beyond ~2x).
Image.MAX_IMAGE_PIXELS = SETTINGS.max_image_dimension * SETTINGS.max_image_dimension


class ImageDecodeError(ValueError):
    """Raised when bytes cannot be turned into a usable image."""


@dataclass
class DecodedImage:
    """A validated, orientation-corrected RGB image plus provenance."""

    image: Image.Image
    width: int
    height: int
    mode: str
    sha256: str
    original_format: str | None

    @property
    def size(self) -> tuple[int, int]:
        return self.image.size


def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def decode_image_bytes(
    data: bytes,
    *,
    max_bytes: int | None = None,
    min_dimension: int | None = None,
    max_dimension: int | None = None,
) -> DecodedImage:
    """Decode ``data`` into an RGB :class:`DecodedImage` or raise.

    Validation order: non-empty -> size limit -> decodable -> dimension limits.
    """
    max_bytes = max_bytes if max_bytes is not None else SETTINGS.max_upload_bytes
    min_dimension = min_dimension or SETTINGS.min_image_dimension
    max_dimension = max_dimension or SETTINGS.max_image_dimension

    if not data:
        raise ImageDecodeError("The uploaded file is empty.")
    if len(data) > max_bytes:
        raise ImageDecodeError(
            f"File is {len(data) / (1024 * 1024):.1f} MB, exceeding the "
            f"{max_bytes / (1024 * 1024):.0f} MB limit."
        )

    try:
        with Image.open(BytesIO(data)) as probe:
            probe.verify()  # verifies integrity without fully loading pixels
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageDecodeError(
            "The file could not be decoded as an image. It may be corrupt or "
            "not a supported format."
        ) from exc

    try:
        # Re-open after verify(): verify() leaves the file in an unusable state.
        image = Image.open(BytesIO(data))
        original_format = image.format
        image = ImageOps.exif_transpose(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        width, height = image.size
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageDecodeError("Failed to read image pixels.") from exc

    if width < min_dimension or height < min_dimension:
        raise ImageDecodeError(
            f"Image is too small ({width}x{height}px); minimum is {min_dimension}px on each side."
        )
    if width > max_dimension or height > max_dimension:
        raise ImageDecodeError(
            f"Image is too large ({width}x{height}px); maximum is {max_dimension}px on each side."
        )

    return DecodedImage(
        image=image,
        width=width,
        height=height,
        mode=image.mode,
        sha256=compute_sha256(data),
        original_format=original_format,
    )


def prepare_model_input(image: Image.Image, transform) -> np.ndarray:
    """Apply a torchvision transform and return a batched NumPy tensor.

    Returns shape ``(1, C, H, W)`` as float32, ready to move onto a device.
    """
    tensor = transform(image)
    if hasattr(tensor, "unsqueeze"):
        tensor = tensor.unsqueeze(0)
    return tensor.numpy().astype(np.float32)


def to_display_array(image: Image.Image, max_side: int = 512) -> np.ndarray:
    """Downscale for display while preserving aspect ratio."""
    work = image.copy()
    work.thumbnail((max_side, max_side))
    return np.asarray(work)
