# MetaTwin

**MetaTwin is a glucose digital twin that fuses static EHR context with dynamic CGM/wearable/meal signals to forecast glucose 60 minutes ahead, predict a specific adverse glucose event, estimate whether the forecast is reliable, and abstain when it is not trustworthy.**

## Required clinical workflow

**Static / historical EHR + dynamic wearable/CGM data → personalized digital twin → 60-minute glucose forecast → adverse-event prediction (>180 or <70 mg/dL within 60 minutes) → calibrated reliability gate → Forecast / Caution / Abstain.**

The adverse-event layer is the primary prediction task. Reliability is a safety/uncertainty layer that controls whether an event alert is surfaced.

## What is implemented

- **Two-stream fusion:** static demographics/labs/EHR features + dynamic CGM, heart rate, activity and meal macros.
- **60-minute forecasting:** XGBoost regressor with a persistence baseline.
- **Adverse-event prediction:** XGBoost classifier for glucose >180 mg/dL or <70 mg/dL within the next 60 minutes.
- **Leakage-safe reliability:** patient-grouped OOF forecast residuals, failure label `|error| > 30 mg/dL`, XGBoost failure detector and isotonic probability calibration.
- **Trust states:** training-only risk thresholds for Forecast / Caution / Abstain; actual holdout coverage is reported rather than assumed.
- **Reliability-gated alerts:** event alerts are withheld when the reliability layer is Abstain.
- **What-if:** bounded model-based scenario demonstrations for baseline, activity and reduced-carbohydrate inputs. These are explicitly non-causal and are disabled when reliability is Abstain.
- **Clinician dashboard:** virtual patient, two data streams, forecast, event risk, trust state, reason codes, cohort triage, research evidence and model-card/safety boundary.
- **Evaluation:** temporal holdout, AUROC/AUPRC, calibration, abstention curve, precision/recall/F1, false alarms/day and patient-level bootstrap 95% confidence intervals.

## Data

### Primary real/open dataset: CGMacros

MetaTwin includes a reproducible preparation script for **CGMacros v1.0.0**, an open PhysioNet dataset containing 45 participants (15 healthy, 16 pre-diabetes and 14 T2D) over approximately ten days, with two CGMs, Fitbit activity/heart rate, meal macronutrients and baseline health measurements. The public data are licensed separately and the raw archive is intentionally **not committed to this repository**. The archive is about 627 MB, so downloading it at deployment/startup would be inappropriate. urlCGMacros on PhysioNethttps://www.physionet.org/content/cgmacros/1.0.0/

The dataset documentation reports Dexcom G6 Pro at 5-minute sampling, Libre at 15-minute sampling, Fitbit heart rate/activity, meal carbohydrate/protein/fat/fiber, and baseline HbA1c/BMI/fasting glucose and demographics. citeturn1view0turn4view0turn5view0

Prepare the real dataset locally:

```bash
mkdir -p data/raw
cd data/raw
wget -r -N -c -np https://physionet.org/files/cgmacros/1.0.0/
cd ../..
python scripts/prepare_cgmacros.py --raw-root data/raw --output-root data/processed
python scripts/evaluate.py
```

The preparation script resamples the real data to a five-minute grid, uses Dexcom as the preferred CGM stream with Libre as fallback, preserves heart rate/METs and meal macros, and creates the two challenge streams (`timeseries.csv` and `ehr.csv`).

### Synthetic development fallback

The repository currently ships synthetic development data so the public demo remains lightweight and immediately runnable. Synthetic EHR extensions are explicitly labeled as simulated. **Synthetic metrics must not be presented as real-world validation.** Once CGMacros preprocessing is run, the same modeling/evaluation pipeline can be rerun on the real/open dataset.

## Reproducible research pipeline

1. `scripts/prepare_cgmacros.py` — converts the public dataset into MetaTwin's canonical schema.
2. `scripts/evaluate.py` — trains the forecasting/event models, creates OOF residuals for reliability, calibrates risk, evaluates the temporal holdout and writes research tables.
3. `src/reliability.py` — implements patient-grouped OOF residual labeling and isotonic calibration.
4. `api.py` — exposes the clinician inference API and live holdout event metrics.

The failure detector **never uses future glucose as a prediction-time feature**. Future glucose is used only after the forecast to construct the training failure label. Patient-level bootstrap resamples whole patients rather than individual rows.

## API

- `GET /api/all` — complete demo payload
- `GET /api/patient/{id}` — virtual patient inference
- `GET /api/cohort` — cohort triage
- `GET /api/event-metrics` — adverse-event and reliability-gated metrics
- `GET /api/summary` — research summary table
- `GET /api/abstention` — coverage/error curve
- `GET /api/calibration` — calibration table
- `/docs` — FastAPI documentation

## Safety boundary

MetaTwin is a research prototype, not a medical device. It does not diagnose, prescribe treatment, recommend medication/dosing, or replace clinician judgment. The what-if layer is a bounded model scenario demonstration, not a causal treatment recommendation. No clinical-validation or population-generalization claim is made.

## Local development

Backend:

```bash
uvicorn api:app --reload --port 8000
```

Dashboard:

```bash
cd dashboard
npm install
npm run dev
```

For deployed frontend use, set `VITE_API_BASE_URL` to the deployed backend URL.

## Submission evidence checklist

- Public GitHub repository: **this repository**
- Working algorithmic model: **implemented**
- Two-stream digital twin: **implemented**
- Specific adverse event + 60-minute window: **implemented**
- Clinician-facing dashboard: **implemented**
- Reliability/abstention layer: **implemented**
- Reproducible real-data preparation: **implemented**
- Leakage-safe evaluation + patient bootstrap: **implemented**
- Architecture diagram, presentation and required demonstration video: **submission artifacts to export from the current implementation**

## Team

Riya Sojitra · BE student
