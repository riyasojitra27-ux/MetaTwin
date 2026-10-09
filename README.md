# MetaTwin

**MetaTwin is a glucose digital twin that fuses static EHR context with dynamic CGM/wearable/meal signals to forecast glucose 60 minutes ahead, predict a specific adverse glucose event, estimate whether the forecast is reliable, and abstain when it is not trustworthy.**

## Required clinical workflow

**Static / historical EHR + dynamic wearable/CGM data → personalized digital twin → 60-minute glucose forecast → adverse-event prediction (>180 or <70 mg/dL within 60 minutes) → calibrated reliability gate → Forecast / Caution / Abstain.**

The adverse-event layer is the primary prediction task. Reliability is the uncertainty/safety layer controlling whether an event alert is surfaced.

## Implemented

- Static EHR + dynamic CGM, heart rate, activity and meal-macro fusion.
- XGBoost 60-minute glucose forecasting with persistence baseline.
- XGBoost adverse-event classifier for >180 or <70 mg/dL within 60 minutes.
- Leakage-safe reliability: patient-grouped OOF forecast residuals, `|error| > 30 mg/dL` failure label, second grouped OOF classifier calibration, isotonic probability calibration.
- Training-only Forecast / Caution / Abstain thresholds.
- Reliability-gated alerts.
- Bounded, non-causal what-if scenarios; disabled while Abstain.
- Clinician dashboard with virtual patient, event risk, reliability, reason codes, cohort triage, research evidence and model-card boundary.
- Temporal holdout evaluation, AUROC/AUPRC, calibration, abstention curve, event precision/recall/F1, false alarms/day and patient-level bootstrap 95% CIs.

## Primary real dataset: CGMacros v1.0.0

MetaTwin has a reproducible pipeline for the open **CGMacros v1.0.0** PhysioNet dataset. It contains 45 participants over approximately ten days with two CGMs, Fitbit activity/heart rate, meal macronutrients and baseline clinical measurements. The raw archive is about 627 MB and is not committed to this repository. Dataset access is subject to its published CC BY-NC-SA 4.0 license.

Source: https://www.physionet.org/content/cgmacros/1.0.0/

## One-command real-data finalization

After cloning/pulling this repository, run:

```bash
bash scripts/finalize_real_data.sh
```

The script:

1. Downloads the official CGMacros archive.
2. Extracts it locally.
3. Converts it to MetaTwin's five-minute canonical schema.
4. Runs the leakage-safe research evaluation on a seven-day train / three-day temporal holdout.
5. Writes final research tables, event metrics, confidence intervals and data provenance.
6. Commits/pushes only derived research outputs; raw and processed real data stay local.
7. Runs `railway up` using the already-linked Railway project.

No GitHub/Railway reconnection is required.

### Manual prerequisite

The Mac needs Python, `curl`, `unzip`, Git and the existing Railway CLI setup. The repository already contains the Python dependencies used by the backend/evaluation environment.

## Data policy

The lightweight repository demo continues to use the included synthetic development data for fast deployment. **Synthetic data are clearly development data and must not be presented as clinical validation.** The primary research headline metrics are generated from CGMacros when `scripts/finalize_real_data.sh` is run. Raw CGMacros and participant-level processed files are intentionally excluded from GitHub.

## Reproducibility

- `scripts/prepare_cgmacros.py` — real-data preprocessing.
- `scripts/evaluate.py` — temporal holdout, forecasting, event prediction, OOF reliability, calibration, abstention and bootstrap evaluation. Set `METATWIN_DATA_ROOT` to evaluate another prepared dataset.
- `src/reliability.py` — patient-grouped OOF residual reliability detector and second OOF calibration pass.
- `api.py` — clinician inference API.

## API

- `GET /api/all` — complete demo payload.
- `GET /api/patient/{id}` — virtual patient inference.
- `GET /api/cohort` — cohort triage.
- `GET /api/event-metrics` — adverse-event and reliability-gated metrics.
- `GET /api/summary` — research summary.
- `GET /api/abstention` — coverage/error curve.
- `GET /api/calibration` — calibration table.
- `/docs` — FastAPI documentation.

## Safety boundary

MetaTwin is a research prototype, not a medical device. It does not diagnose, prescribe treatment, recommend medication/dosing, or replace clinician judgment. What-if outputs are model scenarios, not causal treatment recommendations. No clinical-validation or population-generalization claim is made.

## Submission checklist

- Public GitHub repository.
- Working algorithmic model.
- Two-stream digital twin.
- Specific adverse event + 60-minute window.
- Clinician-facing dashboard.
- Reliability/abstention layer.
- Real-data reproducibility pipeline.
- Leakage-safe evaluation + patient-level bootstrap.
- Architecture/presentation/demo artifacts can be exported from the implemented project.

## Team

Riya Sojitra · BE student
