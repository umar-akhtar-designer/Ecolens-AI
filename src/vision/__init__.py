"""Computer-vision subsystem: preprocessing, model registry, inference,
uncertainty estimation and visual explanations."""

from .preprocessing import (
    DecodedImage,
    ImageDecodeError,
    decode_image_bytes,
    prepare_model_input,
)

__all__ = [
    "DecodedImage",
    "ImageDecodeError",
    "decode_image_bytes",
    "prepare_model_input",
]
