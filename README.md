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

## Current reproducible development evaluation

The repository's current headline evaluation is explicitly labelled **synthetic development data**. It uses a temporal holdout with patient-grouped OOF reliability training and calibration. These results demonstrate the implemented pipeline and must not be presented as clinical validation or real-dataset headline results.

| Metric | Synthetic development result |
|---|---:|
| 60-min forecast MAE | **8.96 mg/dL** |
| 60-min forecast RMSE | **13.57 mg/dL** |
| Forecast MAE 95% CI | **8.57–9.38 mg/dL** |
| Forecast RMSE 95% CI | **12.93–14.28 mg/dL** |
| Failure-detector AUROC | **0.673** |
| Failure-detector AUPRC | **0.104** |
| Event precision, ungated | **93.48%** |
| Event recall, ungated | **89.83%** |
| Event F1, ungated | **91.62%** |
| False alarms/day, ungated | **6.88** |
| Event precision, reliability-gated | **93.51%** |
| Event recall, reliability-gated | **78.30%** |
| Event F1, reliability-gated | **85.23%** |
| False alarms/day, reliability-gated | **5.97** |
| Forecast trust coverage | **69.23%** |
| Caution coverage | **25.03%** |
| Abstain coverage | **5.73%** |

Reliability gating reduced false alarms/day from **6.88 to 5.97 (~13.3%)** on this synthetic holdout while precision remained essentially unchanged (93.48% → 93.51%). The corresponding recall tradeoff is reported rather than hidden.

Detailed machine-readable results are in `results/tables/event_metrics.json` and `results/tables/model_summary.csv`.

## Primary real dataset: CGMacros v1.0.0

MetaTwin also contains a reproducible pipeline for the open **CGMacros v1.0.0** PhysioNet dataset. It contains 45 participants over approximately ten days with two CGMs, Fitbit activity/heart rate, meal macronutrients and baseline clinical measurements. The raw archive is about 627 MB and is not committed to this repository. Dataset access is subject to its published CC BY-NC-SA 4.0 license.

Source: https://www.physionet.org/content/cgmacros/1.0.0/

**CGMacros is optional for the current demo workflow.** The included synthetic dataset is sufficient to run the working model, evaluation and dashboard without downloading the large raw archive. If the real-data archive is available locally, the reproducible finalization script can generate real-data research outputs.

## Optional one-command real-data finalization

After cloning/pulling this repository, run only when you intentionally want to process the full CGMacros archive:

```bash
bash scripts/finalize_real_data.sh
```

The script:

1. Downloads the official CGMacros archive with resumable verification.
2. Extracts it locally.
3. Converts it to MetaTwin's five-minute canonical schema.
4. Runs the leakage-safe research evaluation on a seven-day train / three-day temporal holdout.
5. Writes research tables, event metrics, confidence intervals and data provenance.
6. Commits/pushes only derived research outputs; raw and participant-level processed data stay local.
7. Runs `railway up` using the already-linked Railway project.

No GitHub/Railway reconnection is required.

## Local evaluation without CGMacros

The included development data can be evaluated with:

```bash
PYTHONPATH=. .venv/bin/python scripts/evaluate.py
```

Do **not** run `scripts/finalize_real_data.sh` unless you want the optional full CGMacros workflow.

## Data policy

The lightweight repository demo continues to use the included synthetic development data for fast deployment. **Synthetic data are clearly development data and must not be presented as clinical validation.** If real-data evaluation is later completed, its provenance and metrics should replace or supplement the synthetic headline table only with explicit dataset labelling.

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
- Reproducible evaluation pipeline.
- Leakage-safe evaluation + patient-level bootstrap.
- Architecture/presentation/demo artifacts can be exported from the implemented project.

## Team

Riya Sojitra · BE student
