# EcoLens AI — Simple Clean Demo Script (3 min)

## Problem we're solving (0:00 – 0:15)
- **Screen:** Home page
- **Say:** "Most AI recycling tools guess. They say 'this is recyclable' without proof. That leads to contamination and erodes trust."
- **Say:** "EcoLens AI solves this: it tells you what it actually sees, shows its confidence, and refuses to guess when evidence isn't enough."
- **Point to:** "Explicitly not claimed" section (no recyclability guarantee, no made-up stats).

## What we need to do (0:15 – 0:25)
- **Screen:** Home page
- **Say:** "We need a tool that runs locally, uses a real model, is transparent about uncertainty, and keeps a truthful audit trail of every analysis."
- **Say:** "EcoLens does exactly that — real inference, honest limits, no paid APIs."

## Demo: Real analysis (0:25 – 1:10)
- **Screen:** Image Intelligence → Single image
- **Action:** Upload a clear plastic water bottle
- **Action:** Click "Analyze image"
- **Say:** "This is real model inference, not a mockup."
- **Show:** Prediction name, score labelled **uncalibrated softmax score** (not a probability), uncertainty badge (Confident/Review recommended/Unable to identify reliably), processing time in ms.
- **Tab:** Open **Decision support**
- **Say:** "Three layers: Observation (what the model did), Material evidence (how we inferred material — object-label inference, never measurement), and Recommended next action."
- **Tab:** Open **Visual explanation** → move the **opacity slider** to show original, Grad-CAM overlay, and raw saliency.
- **Say:** "Grad-CAM shows attention, not proof."

## Trust: When it refuses to guess (1:10 – 1:40)
- **Screen:** Still in Single image
- **Action:** Upload an ambiguous/busy photo
- **Action:** Analyze
- **Say:** "When confidence is low, or the object is outside the supported domain, EcoLens **abstains**."
- **Show:** Badge reads **Unable to identify reliably** (or Review recommended). Point out "no material claim" in Decision support. Ask for a clearer photo.
- **Say:** "This is the key point: we refuse to guess."

## Batch + Feedback (1:40 – 2:20)
- **Screen:** Image Intelligence → **Batch analysis** (top radio)
- **Action:** Upload 2–3 images
- **Action:** Click "Analyze batch"
- **Say:** "Same real pipeline per image, isolated failures, CSV export."
- **Show:** Download **results as CSV**
- **Action:** Submit feedback on one result (mark incorrect/unsure). Show confirmation it doesn't change the original record.
- **Screen:** Go to **Analysis History**
- **Say:** "History shows the original prediction, timestamp, images, explanation, decision support (persisted at analysis time), and feedback — never overwritten."
- **Show:** Export page as CSV, open a record's detail (A–D sections), close.

## Insights + Close (2:20 – 3:00)
- **Screen:** **Sustainability Insights**
- **Say:** "Charts are computed only from saved records. We show confident vs review/abstention rates, model usage, and user-reported feedback — with clear labels that it's not accuracy."
- **Screen:** **Material Guide** (optional quick glance)
- **Say:** "Knowledge base is local, versioned, and separate from inference."
- **Close (Home):** "Everything runs locally, is tested (85 automated tests), and every number you saw came from the running model or database — nothing is mocked or fabricated."
- **End.**