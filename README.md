# EcoLens AI

**See Waste Differently. Make Every Material Count.**

EcoLens AI is a working image-intelligence platform for waste awareness. Upload a
photo of an item and it runs a **real, local computer-vision model** to identify
likely objects, communicates **honest uncertainty**, produces a **visual
explanation** of what the model attended to, and connects the result to
**cautious, evidence-based reuse and disposal guidance** — all persisted locally.

Built for Banana Hacks 2026 (Image AI / Sustainability). This is a runnable
application, not a mockup: real inference, real SQLite persistence, real tests.

> **Honesty first.** The default model is an ImageNet *object* classifier, not a
> dedicated waste-material model. Material categories are **conservative hints
> derived from recognised objects**, never measured composition. We never
> fabricate predictions, probabilities, metrics, or environmental impact figures.

---

## 1. What it does

- **Real image analysis** — validate → preprocess → infer → explain → guide → persist.
- **Top-k predictions** with softmax scores labelled *uncalibrated*.
- **Uncertainty state** (`confident` / `uncertain` / `low_confidence`) from top
  score, score margin and entropy, with explicit **abstention** when the score
  is too low or the object is outside the supported material domain.
- **Decision support** — every result is explained in three honest layers:
  observation, material evidence (and *how* it was derived), recommended next action.
- **Grad-CAM explanation** overlay with an interactive opacity slider, raw
  saliency view, and graceful fallback if it fails.
- **Batch analysis** — analyse multiple images at once with per-file isolation,
  a results table and CSV export.
- **User feedback** — flag a result as correct / incorrect / unsure with an
  optional material correction. Feedback is stored **additively** and never
  overwrites the original prediction.
- **Material guidance engine** backed by a local knowledge base, with safety
  warnings and a clear "rules vary locally" disclaimer.
- **Persistent history** — search, filter by material/uncertainty/model/date,
  sort (newest, oldest, score), paginated, detail view, CSV export, delete with
  confirmation.
- **Sustainability Insights** — charts and abstention/confidence rates computed
  *only* from stored records.
- **Material Guide** — browse the knowledge base without running inference.
- **About & Methodology** — model, data, limits, privacy, reproducibility.

### Explicitly *not* claimed
- That an item is recyclable in your area.
- Chemical composition, contamination, or purity.
- Any environmental-impact statistic (none is measured).
- That softmax scores are calibrated probabilities.

---

## 2. Quickstart

### Prerequisites
- Python **3.10+** (developed and tested on **3.14**; a 3.11/3.12 venv works too)
- ~500 MB disk for PyTorch + model weights (MobileNetV3-Large ≈ 21 MB)

### Install & run
```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Then open the URL Streamlit prints (default <http://localhost:8501>).

On first analysis the pretrained weights download automatically from
`download.pytorch.org` and are cached in `~/.cache/torch`. After that, everything
runs offline. No API keys, subscriptions, or paid services are required.

### Configure (optional)
Copy `.env.example` to `.env` and edit if needed (model, device, thresholds,
upload limit, storage locations). All values have sensible defaults.

---

## 3. Running the tests

```bash
pytest                       # 78 tests: preprocessing, security, inference, services, DB, analytics, e2e
ruff check .                 # lint
ruff format --check .        # formatting
```

Tests run against an isolated temporary database and storage directory; they
never touch your real data. Tests that require the pretrained weights are
skipped automatically if the machine is offline and weights are not cached.

Evaluation harness (only reports real metrics on a documented dataset):

```bash
python scripts/evaluate_model.py --data-dir path/to/test --mapping mapping.json
```

Reset local data (destructive):
```bash
python scripts/reset_database.py --yes
```

---

## 4. Architecture

```
Streamlit UI (app.py + pages/)
        │
        ├── src/services/     analysis_service.py — single + batch orchestration
        │
        ├── src/ui/           design system + components
        │
        ├── src/vision/       ← the AI core
        │     ├── preprocessing.py     validation, EXIF, RGB, model transform
        │     ├── model_registry.py    lazy model loading, device selection, adapter
        │     ├── inference.py         end-to-end analyze_image_bytes()
        │     ├── uncertainty.py       softmax → confidence state + abstention
        │     ├── explanations.py      Grad-CAM overlay + opacity blending
        │     └── material_mapping.py  conservative object → material hint
        │
        ├── src/guidance/     knowledge base, guidance synthesis, decision support
        ├── src/database/     SQLAlchemy models, init, repository (DTOs)
        ├── src/analytics/    metrics + CSV export from stored records
        └── src/security/     untrusted-upload handling, safe storage
```

The vision core is fully independent of the UI and directly unit-tested. The
`src/services/analysis_service.py` layer is the single seam between the UI and
the vision/guidance/persistence stack, so the same tested pipeline powers both
single and batch analysis. Swapping in a future dedicated waste classifier means
adding one `ModelSpec` in `src/vision/model_registry.py` — no page rewrites.

### Data flow (one analysis)
`upload bytes → decode/validate → RGB → transform → model → softmax → top-k →
uncertainty (+ supported-domain check) → material hint → Grad-CAM → guidance →
decision support → SQLite + files on disk`

---

## 5. Project structure

```
ecolens-ai/
├── app.py                       # Home / Overview
├── pages/
│   ├── 1_Image_Intelligence.py  # core analysis workspace
│   ├── 2_Analysis_History.py    # history + detail + delete
│   ├── 3_Sustainability_Insights.py
│   ├── 4_Material_Guide.py
│   └── 5_About_Methodology.py
├── src/
│   ├── vision/                  # preprocessing, registry, inference, uncertainty, explanations, mapping
│   ├── services/                # analysis_service.py — single + batch orchestration
│   ├── guidance/                # engine.py, decision.py (decision support)
│   ├── database/                # models.py, initialization.py, repository.py
│   ├── analytics/               # metrics.py, export.py (CSV)
│   ├── security/                # uploads.py
│   ├── ui/                      # theme.py, components.py, bootstrap.py
│   ├── config.py
│   └── logging_config.py
├── data/material_knowledge.json # knowledge base (source of truth)
├── tests/                       # 78 automated tests
├── scripts/evaluate_model.py
├── scripts/reset_database.py
├── scripts/health_check.py
├── assets/
├── .streamlit/config.toml
├── Dockerfile
├── requirements.txt
├── pyproject.toml
├── README.md · MODEL_CARD.md · DATA_SOURCES.md · PRIVACY.md · DEMO_SCRIPT.md · LICENSE
```

---

## 6. The model (summary — see `MODEL_CARD.md`)

- **Default:** MobileNetV3-Large, ImageNet-1k weights, via `torchvision`.
- **Why:** small, fast on CPU, well-suited to Grad-CAM, no training required for
  a working demo. A generic object classifier, used honestly.
- **Device:** `auto` (CUDA → MPS → CPU). CPU inference ≈ 100 ms/image after warm-up.
- **Alternatives in the registry:** `mobilenet_v3_small`, `efficientnet_b0`.
- **Scores:** softmax, explicitly labelled **uncalibrated**.
- **Material hints:** a curated allow-list maps ~120 recognised object labels to
  9 material categories. Unmapped labels produce **no** material claim.

### Model adapter & future dedicated classifier
`ModelSpec` + `model_registry.py` isolate the architecture. To integrate a
fine-tuned waste model, add a builder and set `is_direct_material_classifier=True`;
the UI already adapts its wording to that flag.

---

## 7. Security & privacy (summary — see `PRIVACY.md`)

- Uploads validated by extension, size, and **real decode** (not just MIME).
- Generated filenames; **path-traversal protection** on all storage reads/deletes.
- Uploaded files are never executed or served from a public directory.
- No secrets committed; `.gitignore` covers `.env`, DB, uploads, weights, caches.
- All data is local. No external API calls in the core workflow.
- Not claimed to be "fully secure" — these are sensible baseline controls.

---

## 8. Database schema

- `analyses` — one row per analysis (prediction, score, uncertainty, timing, paths, model metadata).
- `prediction_candidates` — top-k rows per analysis (rank, label, raw score, material slug).
- `guidance_snapshots` — the guidance shown at analysis time.
- `feedback` — additive user feedback (verdict, optional material correction, note) linked to an analysis.
- `material_knowledge` — knowledge base mirrored from JSON for queryability.

Foreign keys cascade on delete. `init_db()` creates tables and syncs knowledge
idempotently; it **never deletes user data**, and adding the `feedback` table does
not touch existing rows.

---

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| First analysis is slow | One-time model download + load. Later analyses ≈ 100 ms. |
| `ModelLoadError` / offline | Pre-cache weights (run once online) or set `ECOLENS_ALLOW_DOWNLOAD=0` with cached weights. |
| Port already in use | `streamlit run app.py --server.port 8502` |
| MPS/GPU oddities | Set `ECOLENS_DEVICE=cpu`. |
| Reset everything | `python scripts/reset_database.py --yes` |
| `sqlite` lock errors | Ensure only one app instance is writing. |

---

## 10. Reproducing the demo (3-minute video flow)

1. Home → read scope + limitations.
2. Image Intelligence → upload a photo → **Analyze** → show prediction, score
   labelled *uncalibrated*, uncertainty badge.
3. **Decision support** tab → three honest layers: observation, material evidence
   (and *how* it was derived), recommended next action.
4. Open **Visual explanation** tab → adjust the opacity slider → compare original,
   Grad-CAM overlay and raw saliency.
5. Open **Guidance** tab → reuse + disposal + safety note + local-rules disclaimer.
6. Submit **feedback** on the result (correct / incorrect / unsure).
7. Switch to **Batch analysis** → upload several images → download the CSV.
8. Analysis History → search/filter/sort → export CSV → open detail → delete.
9. Sustainability Insights → real charts, confidence & abstention rates from records.
10. Material Guide → browse without AI.
11. About & Methodology → model, limits, privacy.

---

## 11. Submission checklist

- [x] App starts successfully; a user can upload a valid image.
- [x] The vision model really processes the image; results reflect real output.
- [x] Uncertainty and limitations communicated correctly.
- [x] Guidance connected to the analysis via a knowledge base.
- [x] History persists in SQLite; records reviewable and deletable.
- [x] Analytics computed from real records.
- [x] Invalid uploads and inference failures handled gracefully.
- [x] Batch analysis with per-file isolation + CSV export.
- [x] Additive user feedback with explicit non-mutation guarantee.
- [x] Automated tests written and passing (78).
- [x] README with reproducible setup; model/data licenses documented.
- [ ] Public GitHub push, Devpost write-up, YouTube demo *(team action)*.

---

## 12. License

Source: MIT (`LICENSE`). Pretrained weights are provided by their original
publishers under their own terms — see `DATA_SOURCES.md` and `MODEL_CARD.md`.
