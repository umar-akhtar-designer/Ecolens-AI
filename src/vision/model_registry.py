"""Model registry and lazy loader.

The registry describes *what* models are available and how to build them, in a
single adapter layer. Swapping in a future dedicated waste classifier means
adding one :class:`ModelSpec` here without touching the rest of the app.

Default choice rationale
------------------------
``mobilenet_v3_large`` (ImageNet-1k, torchvision) is small, CPU-friendly and
supports Grad-CAM (last convolutional block). It recognises *objects*, which we
then translate conservatively into material hints. It is **not** a dedicated
waste classifier; see ``MODEL_CARD.md``.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from ..config import SETTINGS
from ..logging_config import get_logger

logger = get_logger("vision.registry")


class ModelLoadError(RuntimeError):
    """Raised when model weights cannot be built or downloaded."""


@dataclass
class ModelSpec:
    key: str
    display_name: str
    architecture: str
    version: str
    weights_source: str
    license: str
    num_classes: int
    builder: Callable[[bool], Any]
    categories: tuple[str, ...]
    transform: Any
    target_layer_name: str
    is_direct_material_classifier: bool = False
    notes: str = ""

    metadata: dict[str, Any] = field(default_factory=dict)


_CACHE: dict[str, ModelSpec] = {}
_LOCK = threading.Lock()


def _device() -> torch.device:
    pref = SETTINGS.device.lower()
    if pref == "cpu":
        return torch.device("cpu")
    if pref in ("cuda", "gpu"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device() -> torch.device:
    return _device()


def _build_mobilenet_v3_large(download: bool):
    from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

    weights = MobileNet_V3_Large_Weights.DEFAULT if download else None
    model = mobilenet_v3_large(weights=weights)
    return model, weights


def _build_mobilenet_v3_small(download: bool):
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    weights = MobileNet_V3_Small_Weights.DEFAULT if download else None
    model = mobilenet_v3_small(weights=weights)
    return model, weights


def _build_efficientnet_b0(download: bool):
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

    weights = EfficientNet_B0_Weights.DEFAULT if download else None
    model = efficientnet_b0(weights=weights)
    return model, weights


_BUILDERS: dict[str, tuple[Callable[[bool], Any], str, str]] = {
    "mobilenet_v3_large": (
        _build_mobilenet_v3_large,
        "features.16",
        "MobileNetV3-Large (ImageNet-1k)",
    ),
    "mobilenet_v3_small": (
        _build_mobilenet_v3_small,
        "features.12",
        "MobileNetV3-Small (ImageNet-1k)",
    ),
    "efficientnet_b0": (
        _build_efficientnet_b0,
        "features.8",
        "EfficientNet-B0 (ImageNet-1k)",
    ),
}


def _resolve_spec(key: str) -> ModelSpec:
    if key not in _BUILDERS:
        raise ModelLoadError(f"Unknown model '{key}'. Available: {', '.join(sorted(_BUILDERS))}.")
    build, target_layer, display = _BUILDERS[key]

    try:
        model, weights = build(SETTINGS.allow_weight_download)
    except Exception as exc:  # pragma: no cover - network/disk dependent
        raise ModelLoadError(
            f"Could not build '{key}' weights. If the machine is offline, "
            f"pre-download the torchvision weights into the torch cache. "
            f"Original error: {exc}"
        ) from exc

    if weights is not None:
        categories = tuple(weights.meta.get("categories", []))
        transform = weights.transforms()
        version = getattr(weights, "url", "") or "torchvision-weights"
        license_ = "BSD-3-Clause (torchvision code); weights per torchvision model zoo"
    else:
        # Random-init fallback (offline, no weights available). Clearly labelled.
        categories = tuple(str(i) for i in range(model.classifier[-1].out_features))
        from torchvision import transforms

        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        version = "uninitialised-weights"
        license_ = "BSD-3-Clause (torchvision code)"

    model.eval()
    spec = ModelSpec(
        key=key,
        display_name=display,
        architecture=key,
        version=version,
        weights_source="torchvision (download.pytorch.org)",
        license=license_,
        num_classes=len(categories),
        builder=lambda download: build(download)[0],
        categories=categories,
        transform=transform,
        target_layer_name=target_layer,
        is_direct_material_classifier=False,
        notes=(
            "ImageNet object classifier used conservatively; material labels are "
            "derived hints, not direct material classification."
        ),
        metadata={
            "trained_on": "ImageNet-1k (1000 classes)",
            "device": str(_device()),
            "pretrained_weights_loaded": weights is not None,
        },
    )
    return spec


def get_model_spec(key: str | None = None) -> ModelSpec:
    key = key or SETTINGS.model_key
    with _LOCK:
        if key not in _CACHE:
            _CACHE[key] = _resolve_spec(key)
        return _CACHE[key]


_MODEL_CACHE: dict[str, Any] = {}


def get_model(key: str | None = None):
    """Return a cached, eval-mode model moved to the active device."""
    spec = get_model_spec(key)
    with _LOCK:
        if spec.key not in _MODEL_CACHE:
            logger.info("Loading model", extra={"event": "model_load", "model": spec.key})
            model = (
                spec.builder(False)
                if not spec.metadata.get("pretrained_weights_loaded", True)
                else _BUILDERS[spec.key][0](True)[0]
            )
            model.eval()
            model.to(_device())
            _MODEL_CACHE[spec.key] = model
        return _MODEL_CACHE[spec.key]


def clear_cache() -> None:
    with _LOCK:
        _CACHE.clear()
        _MODEL_CACHE.clear()
