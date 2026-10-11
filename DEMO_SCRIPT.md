# EcoLens AI — 3-Minute Demo Script

A tight, honest walkthrough that shows the app doing **real** work and being
**explicit** about what it cannot do. Total ≈ 3:00.

## Pre-flight (before recording)

- App running: `streamlit run app.py` (note your local URL or tunnel URL).
- Have 2–3 photos ready on the desktop:
  - one clearly recognisable object (e.g. a plastic water bottle),
  - one ambiguous / busy photo (to show abstention),
  - 2–3 extra images for the batch step.
- Run `python scripts/health_check.py` once so you know the model loads.
- Delete old history if you want a clean slate: `python scripts/reset_database.py --yes`.

## Scene 1 — Hook & honesty (0:00–0:25)

- **Home / About** page.
- Say: *"EcoLens AI looks at a photo of an item and tells you what it probably is,
  what material it might be, and how to dispose of it — while being honest that it
  uses a general object classifier, not a magic waste sensor."*
- Point at the "Explicitly not claimed" list: no recyclability determination, no
  composition, no invented impact statistics.

## Scene 2 — A real analysis (0:25–1:05)

- Go to **Image Intelligence → Single image**.
- Upload the clearly recognisable object → **Analyze image**.
- Show: prediction, the score labelled **uncalibrated**, the uncertainty badge,
  and processing time (real ms from this run).
- Open **Decision support**: read the three layers — *observation*, *material
  evidence* (note "inferred from the recognised object, not measured"), and the
  *recommended next action*.

## Scene 3 — Explanation & guidance (1:05–1:35)

- **Visual explanation** tab → drag the **opacity slider** and point out the
  original vs Grad-CAM overlay vs raw saliency.
- **Guidance** tab → reuse and disposal text, any safety note, and the
  "rules vary by municipality" disclaimer.

## Scene 4 — Abstention (the trust moment) (1:35–1:55)

- Upload the ambiguous/busy photo → **Analyze**.
- Show the system **abstaining**: low-confidence / outside supported domain, so
  **no material claim** is made and it asks for a clearer photo.
- Say: *"This is the behaviour we're proudest of — it refuses to guess."*

## Scene 5 — Batch, feedback & history (1:55–2:30)

- Switch to **Batch analysis** → upload 3 images → **Analyze batch**.
- Show the summary counts and **Download results as CSV**.
- Submit **feedback** on one result (mark it *incorrect*).
- Go to **Analysis History** → search/filter/sort → **Export page as CSV** →
  open a detail → show the reconstructed decision support, recorded feedback,
  then delete it.

## Scene 6 — Insights & close (2:30–3:00)

- **Sustainability Insights**: real charts + confident/abstention rates from
  stored records, and the feedback note that user feedback is not an accuracy claim.
- **Material Guide**: browse the knowledge base without the model.
- Close: *"Everything is local, tested (78 automated tests), and every number you
  saw came from the running model or the database — nothing is mocked."*

## Talking points for limitations (drop in anywhere)

- The model is an ImageNet **object** classifier; material is a derived hint.
- Softmax scores are **uncalibrated** and are not probabilities.
- Thresholds are operational safeguards, not statistically validated calibration.
- Grad-CAM shows model attention, not cause.
- Recycling rules vary by municipality; the app never decides recyclability.
