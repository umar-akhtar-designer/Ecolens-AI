"""Visual explanations via Grad-CAM.

Grad-CAM (Selvaraju et al., 2017) highlights the regions of the input that most
influenced the score of a chosen class. It is a *visual aid*, not a causal
proof. When it fails, the caller must fall back to showing the original image
and prediction — the app never depends on explanation success.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from ..logging_config import get_logger
from .model_registry import get_device, get_model, get_model_spec

logger = get_logger("vision.explanations")


class ExplanationError(RuntimeError):
    """Raised when a visual explanation cannot be produced."""


@dataclass
class ExplanationResult:
    overlay: Image.Image
    heatmap: np.ndarray  # float [0,1], shape (H, W)
    method: str
    target_class_index: int
    note: str


def _find_target_layer(model, layer_name: str):
    module = model
    for part in layer_name.split("."):
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


def heatmap_to_color(heatmap: np.ndarray) -> np.ndarray:
    """Map a float ``[0,1]`` heatmap to a jet-like RGB array (no matplotlib)."""
    h = np.clip(heatmap, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * h - 3.0), 0, 1)
    g = np.clip(1.5 - np.abs(4.0 * h - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(4.0 * h - 1.0), 0, 1)
    return np.stack([r, g, b], axis=-1)


def heatmap_image(heatmap: np.ndarray) -> Image.Image:
    """Return the raw colour-mapped heatmap as a standalone image."""
    color = heatmap_to_color(heatmap)
    return Image.fromarray((color * 255).astype(np.uint8))


def blend_overlay(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Blend a jet-coloured heatmap onto the original RGB image at ``alpha``."""
    alpha = float(min(max(alpha, 0.0), 1.0))
    base = image.convert("RGB").copy()
    base = base.resize((heatmap.shape[1], heatmap.shape[0]))
    base_arr = np.asarray(base, dtype=np.float32) / 255.0
    color = heatmap_to_color(heatmap)
    blended = (1 - alpha) * base_arr + alpha * color
    out = (np.clip(blended, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(out)


# Backwards-compatible alias for the previous private helper name.
_build_overlay = blend_overlay


def generate_gradcam(
    image: Image.Image,
    class_index: int,
    model_key: str | None = None,
) -> ExplanationResult:
    """Produce a Grad-CAM overlay for ``class_index`` on ``image``."""
    spec = get_model_spec(model_key)
    try:
        model = get_model(model_key)
        target_layer = _find_target_layer(model, spec.target_layer_name)
    except Exception as exc:  # pragma: no cover - depends on architecture
        raise ExplanationError(f"Could not access model layers: {exc}") from exc

    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}

    def forward_hook(_module, _inputs, output):
        activations["value"] = output.detach()

    def backward_hook(_module, _grad_in, grad_out):
        gradients["value"] = grad_out[0].detach()

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    device = get_device()
    try:
        tensor = spec.transform(image).unsqueeze(0).to(device)
        tensor.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        output = model(tensor)
        if class_index < 0 or class_index >= output.shape[1]:
            raise ExplanationError("Target class index out of range.")
        score = output[0, class_index]
        score.backward()

        if "value" not in gradients or "value" not in activations:
            raise ExplanationError("Gradients were not captured for the target layer.")

        grads = gradients["value"][0]  # (C, H, W)
        acts = activations["value"][0]  # (C, H, W)
        weights = grads.mean(dim=(1, 2))  # (C,)
        cam = torch.relu((weights[:, None, None] * acts).sum(dim=0))
        cam = cam.detach().cpu().numpy()

        if not np.isfinite(cam).all() or cam.max() <= 0:
            raise ExplanationError("Saliency map was empty or non-finite.")

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        heatmap = (
            np.asarray(
                Image.fromarray((cam * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR),
                dtype=np.float32,
            )
            / 255.0
        )
    finally:
        fh.remove()
        bh.remove()

    overlay = _build_overlay(image, heatmap)
    return ExplanationResult(
        overlay=overlay,
        heatmap=heatmap,
        method="Grad-CAM",
        target_class_index=class_index,
        note=(
            "Grad-CAM highlights image regions that most increased the chosen "
            "class score. Warm colours indicate higher influence. It is a model "
            "attention aid, not proof of cause."
        ),
    )
