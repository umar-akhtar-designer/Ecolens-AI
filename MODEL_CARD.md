# Model Card — EcoLens AI

## 1. Model details

| Field | Value |
|---|---|
| Default model key | `mobilenet_v3_large` |
| Architecture | MobileNetV3-Large (Howard et al., 2019) |
| Weights | ImageNet-1k pretrained, via `torchvision` |
| Weights source | `https://download.pytorch.org/models/mobilenet_v3_large-5c1a4163.pth` |
| Classes | 1000 ImageNet object classes |
| Input size | 224×224 (model's own transform) |
| Framework | PyTorch / torchvision |
| Device | CPU by default; CUDA/MPS if available (`ECOLENS_DEVICE=auto`) |
| License (code) | BSD-3-Clause (torchvision) |
| Direct material classifier? | **No** |

Registered alternatives (same adapter, selectable via `ECOLENS_MODEL`):
- `mobilenet_v3_small` — smaller/faster, lower accuracy.
- `efficientnet_b0` — higher capacity, slower on CPU.

## 2. Intended use

- **Primary:** recognise common objects in a user photo and, where an object label
  maps to a material category, offer a **cautious material hint** plus reuse and
  disposal guidance.
- **Users:** students, households, educators, community recycling programmes.
- **Out of scope:** determining chemical composition, contamination, purity,
  recyclability, or safe handling of hazardous/unknown items from a photo.

## 3. How predictions become guidance

1. The model outputs a 1000-way softmax vector.
2. Top-k labels are taken (`ECOLENS_TOP_K`, default 5).
3. Softmax scores are reported **as uncalibrated scores** (`score_type =
   softmax_score_uncalibrated`) — not probabilities.
4. An **uncertainty state** is derived from the top score, margin and entropy:
   - `low_confidence` — top score < `ECOLENS_CONFIDENCE_THRESHOLD` (0.35).
   - `uncertain` — high entropy or small margin between top classes.
   - `confident` — one class clearly favoured.
5. **Supported-domain check:** the uncertainty layer also receives whether the top
   label maps into the curated material domain. If not, the system **abstains**:
   it downgrades a confident object prediction and refuses to make a material
   claim. A `low_confidence` result likewise triggers abstention.
6. A **material hint** is produced only if the top label is in the curated
   `MATERIAL_CLASS_MAP` (~120 labels → 9 categories). Otherwise no material
   claim is made.
7. Guidance is material-specific **only** when the confidence state is
   `confident`; otherwise the app returns general guidance and asks for a
   clearer image.
8. The result is expressed as **three decision-support layers** — observation,
   material evidence (with its basis), and a recommended next action — so the
   user always sees what was decided, why, and what to do next.

## 4. Factors and limitations

- **Domain gap:** ImageNet contains many everyday objects but is not a waste
  dataset; some common recyclables (e.g. specific bottle types) are recognised
  only loosely.
- **Overconfidence:** softmax outputs are typically overconfident and are **not**
  calibrated. Thresholds are operational safeguards, not guarantees.
- **Single object assumption:** a busy scene can confuse the classifier.
- **Appearance ≠ material:** visual appearance can be misleading (e.g. a
  plastic bottle that looks like glass).
- **Hints, not measurements:** no sensor data is used; material categories are
  inferred from labels.
- **Grad-CAM:** shows attention, not causation; a missing/weak map is possible.

## 5. Metrics

**No task-specific accuracy metrics are claimed**, because the default model is
not a waste-material classifier and this project ships no labelled waste test
set. `scripts/evaluate_model.py` measures real metrics only when a licensed,
labelled dataset and an explicit label-mapping are supplied; it prints an honest
"no dataset" notice otherwise. We do not report training or generalisation
metrics we did not measure.

Measured runtime on the development machine (Apple Silicon, MPS, MobileNetV3-Large):

| Stage | Observed |
|---|---|
| First analysis (cold: weights load + warm-up) | ~2.7 s |
| Steady-state inference + Grad-CAM | ~0.10 s/image |
| Model weights size | ~21 MB |

*These are latency observations on one machine, not accuracy claims.*

## 6. Ethical / responsible-use notes

- The app refuses to over-claim and surfaces uncertainty on every result.
- Safety warnings appear for battery-bearing and sharp-glass objects.
- Local recycling rules vary; the UI always states that guidance is general.
- No personal data is required to use the tool; nothing leaves the machine.

## 7. Future work

- Fine-tune on a licensed waste dataset (`DATA_SOURCES.md` lists candidates) and
  set `is_direct_material_classifier=True`.
- Probability calibration (temperature scaling) on a held-out set.
- Multi-object detection instead of single-label classification.
