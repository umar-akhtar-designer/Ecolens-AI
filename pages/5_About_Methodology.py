"""About & Methodology — transparency on models, data, limits and privacy."""

from __future__ import annotations

import platform

import streamlit as st
import torch

from src.config import SETTINGS
from src.ui.bootstrap import setup_page, sidebar_footer
from src.ui.components import callout
from src.vision.model_registry import get_device, get_model_spec

setup_page("About & Methodology")
sidebar_footer()

st.markdown("## About EcoLens AI")
st.markdown(
    "EcoLens AI is an image-intelligence platform for material awareness: identify "
    "likely objects in an image, communicate uncertainty honestly, explain predictions "
    "visually, and connect results to practical reuse and disposal guidance."
)

st.markdown("### Method")
st.markdown(
    """
1. **Validation** — extension, size and real decodability are checked; EXIF orientation
   is honoured; the image is converted to RGB.
2. **Preprocessing** — the model's own torchvision transform is applied (resize, crop,
   normalise), so training and inference preprocessing match.
3. **Inference** — a pretrained model runs in `eval()` mode under `torch.no_grad()`.
4. **Calibration awareness** — raw softmax scores are labelled *uncalibrated*; a
   confidence/uncertainty state is derived from the top score, score margin and entropy.
5. **Explanation** — Grad-CAM (forward + backward hooks on the last conv block) produces
   an attention overlay, with a clear statement of what it does and does not prove.
6. **Guidance** — recognised objects are mapped conservatively to material categories and
   combined with a local knowledge base.
7. **Persistence** — results, candidates and a guidance snapshot are stored in SQLite.
"""
)


def _current_model_info():
    try:
        spec = get_model_spec()
        return spec, None
    except Exception as exc:  # pragma: no cover - network dependent
        return None, str(exc)


spec, err = _current_model_info()

st.markdown("### Model")
if spec is not None:
    st.markdown(
        f"""
- **Key:** `{spec.key}`
- **Display name:** {spec.display_name}
- **Architecture:** {spec.architecture}
- **Weights source:** {spec.weights_source}
- **Weights version:** `{spec.version}`
- **Classes:** {spec.num_classes}
- **License:** {spec.license}
- **Direct material classifier:** {spec.is_direct_material_classifier}
- **Notes:** {spec.notes}
"""
    )
else:
    callout(f"Model information unavailable: {err}", "caution")

callout(
    "This is an ImageNet object classifier, not a dedicated waste-material model. "
    "Material labels are conservative hints derived from recognised objects. See "
    "MODEL_CARD.md and DATA_SOURCES.md in the repository.",
    "info",
)

st.markdown("### Environment")
st.markdown(
    f"- Python: `{platform.python_version()}`\n"
    f"- PyTorch: `{torch.__version__}`\n"
    f"- Device: `{get_device()}`\n"
    f"- Confidence threshold: `{SETTINGS.confidence_threshold:.2f}`\n"
    f"- Upload limit: `{SETTINGS.max_upload_mb} MB`"
)

st.markdown("### Known limitations")
st.markdown(
    """
- A generic object classifier recognises familiar objects, not material composition.
- Softmax scores are **not calibrated probabilities**.
- Recyclability depends on local rules and is never determined here.
- Grad-CAM is an attention aid, not a causal explanation.
- No environmental-impact statistics are produced — none are measured.
- Accuracy is only meaningful on a documented, labelled test set (see `scripts/evaluate_model.py`).
"""
)

st.markdown("### Privacy & storage")
st.markdown(
    """
- Uploaded images and explanations are stored **locally** under `storage/`.
- Metadata is stored in a local SQLite database (`storage/ecolens.db`).
- No external APIs or paid services are used for the core workflow.
- Uploaded files are treated as untrusted input and never executed.
- No personal accounts or unnecessary personal data are collected.
"""
)

st.markdown("### Implemented vs planned")
st.markdown(
    """
**Implemented:** real local inference; uncertainty handling; Grad-CAM with fallback;
persistent history with search/filter/delete; analytics from real records; material
knowledge base; automated tests; documentation.

**Planned / optional:** fine-tuning on a licensed waste dataset; calibrated probabilities;
batch analysis; multi-image comparison.
"""
)
