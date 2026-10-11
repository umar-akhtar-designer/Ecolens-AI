# Data Sources & Licenses

EcoLens AI uses a **pretrained model** and a **hand-authored knowledge base**;
it does **not** ship a training dataset. This file documents every data/model
source used, and lists candidate datasets for future fine-tuning.

## 1. Model weights

| Asset | Source | License / terms |
|---|---|---|
| MobileNetV3-Large (ImageNet-1k) | torchvision → `download.pytorch.org` | BSD-3-Clause (torchvision). Weights follow the torchvision model-zoo terms; ImageNet is subject to its own dataset terms. |
| MobileNetV3-Small (ImageNet-1k) | torchvision | as above |
| EfficientNet-B0 (ImageNet-1k) | torchvision | as above |

Weights are downloaded automatically on first use and cached locally; they are
**not** committed to the repository.

## 2. Knowledge base

`data/material_knowledge.json` is authored for this project. Guidance is general
and reflects widely published recycling information. Primary references:

- U.S. EPA — Recycling basics: https://www.epa.gov/recycle
- U.S. EPA — How do I recycle common recyclables:
  https://www.epa.gov/recycle/how-do-i-recycle-common-recyclables
- U.S. EPA — Reducing and reusing: https://www.epa.gov/recycle/reducing-and-reusing-basics
- U.S. EPA — Electronics donation and recycling:
  https://www.epa.gov/recycle/electronics-donation-and-recycling
- U.S. EPA — Used household batteries: https://www.epa.gov/recycle/used-household-batteries
- U.S. EPA — Composting at home: https://www.epa.gov/recycle/composting-home

`geographic_scope` is currently "General (US-oriented)". Local rules vary; the
app never asserts universal recyclability.

## 3. Candidate datasets for future fine-tuning (not used yet)

These are commonly cited, publicly available waste/recycling image datasets.
**Verify each license before any use**, and record the exact version/split used.

| Dataset | Notes | Check before use |
|---|---|---|
| TrashNet (Gary Thung & Mindy Yang) | ~2.5k images, 6 classes; widely used | Confirm license terms on the source repository |
| TACO — Trash Annotations in Context | COCO-style, litter in the wild | CC BY 4.0 (verify) |
| RealWaste | ~4.7k images, 9 classes | Check Kaggle/source terms |
| Waste Classification data (Kaggle) | Organic vs recyclable | Check source terms |

If a dataset is used, update `MODEL_CARD.md` with the split, class distribution,
imbalance handling, hardware, and measured metrics — and never evaluate on
training examples while calling it generalisation.

## 4. Evaluation data

None bundled. `scripts/evaluate_model.py` requires the user to supply a labelled
set plus an explicit mapping file. Until then, **no accuracy metrics are
reported** — metrics are never fabricated.

## 5. Reproducing model downloads

```bash
# Warm the torchvision cache (needs network):
python -c "from src.vision.model_registry import get_model; get_model()"
```

For offline deployment, pre-populate `~/.cache/torch/hub/checkpoints/` and set
`ECOLENS_ALLOW_DOWNLOAD=0`.
