# MetaTwin Architecture

## End-to-end flow

```text
             ┌──────────────────────────────┐
             │ Static / Historical EHR      │
             │ age, BMI, HbA1c, labs,       │
             │ diagnoses, baseline context  │
             └──────────────┬───────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────┐
│                 PERSONALIZED DIGITAL TWIN            │
│                                                     │
│  Dynamic state: CGM + HR + activity + meal macros  │
│  Feature state: trend + 30-min mean/std + time      │
│  Static context: EHR features                       │
└─────────────────────────┬───────────────────────────┘
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
┌──────────────────────┐   ┌─────────────────────────┐
│ 60-min glucose       │   │ Adverse-event classifier │
│ forecaster (XGB)     │   │ >180 or <70 mg/dL       │
└──────────┬───────────┘   │ within 60 minutes       │
           │               └────────────┬────────────┘
           │                            │
           ▼                            ▼
┌─────────────────────────────────────────────────────┐
│ Reliability layer                                   │
│ OOF patient-grouped residuals → failure model →     │
│ isotonic calibration → Forecast / Caution / Abstain │
└─────────────────────────┬───────────────────────────┘
                          │
                          ▼
             ┌──────────────────────────┐
             │ Reliability-gated alert  │
             │ + reason codes           │
             └────────────┬─────────────┘
                          │
                          ▼
             ┌──────────────────────────┐
             │ Clinician dashboard      │
             │ virtual patient          │
             │ cohort triage             │
             │ research evidence         │
             │ bounded what-if scenarios │
             └──────────────────────────┘
```

## Leakage controls

- Training uses the first seven patient-relative days; evaluation uses the later temporal holdout.
- The 60-minute forecast target is shifted from future glucose and is never included as a prediction-time feature.
- Reliability labels are created from patient-grouped out-of-fold forecast residuals rather than in-sample residuals.
- Reliability calibration uses isotonic regression on OOF predictions.
- Reliability thresholds are selected from training-only risk quantiles.
- Confidence intervals resample whole patients, not individual observations.

## Data provenance

The production research path supports CGMacros v1.0.0 from PhysioNet. Raw data are not committed because the source archive is large and separately licensed. Synthetic data remain in the repository as a lightweight development fallback.
