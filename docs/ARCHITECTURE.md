# MetaTwin Architecture

## End-to-end flow

```text
                 ┌─────────────────────────────┐
                 │      Static / Historical    │
                 │      EHR context            │
                 │ age · BMI · HbA1c · labs   │
                 └──────────────┬──────────────┘
                                │
                                │
                 ┌──────────────▼──────────────┐
                 │       Dynamic streams       │
                 │ CGM · HR · activity · meal │
                 │ macros · recent trajectory  │
                 └──────────────┬──────────────┘
                                │
                                ▼
                 ┌─────────────────────────────┐
                 │      PERSONALIZED STATE     │
                 │        MetaTwin patient     │
                 └──────────────┬──────────────┘
                                │
                 ┌──────────────┴──────────────┐
                 │                             │
                 ▼                             ▼
       ┌──────────────────┐          ┌────────────────────┐
       │ 60-min forecaster│          │ Adverse-event model│
       │     XGBoost      │          │ >180 or <70 / 60m │
       └────────┬─────────┘          └──────────┬─────────┘
                │                              │
                ▼                              │
       ┌──────────────────┐                    │
       │ Failure detector │                    │
       │ OOF residuals    │                    │
       │ + calibration    │                    │
       └────────┬─────────┘                    │
                │                              │
                ▼                              ▼
       ┌─────────────────────────────────────────────┐
       │             RELIABILITY GATE                │
       │ calibrated failure risk → Forecast/Caution │
       │ /Abstain                                    │
       └─────────────────────┬───────────────────────┘
                             │
                             ▼
                 ┌─────────────────────────────┐
                 │ Clinician-facing dashboard │
                 │ forecast · event risk      │
                 │ trust state · reason codes │
                 │ what-if · cohort triage    │
                 └─────────────────────────────┘
```

## Prediction-time leakage boundary

The failure detector is trained from out-of-fold forecast residuals. At inference time it receives only features available at prediction time. Future glucose and realized forecast error are never used as failure-detector features.

The adverse event is defined independently from the future glucose trajectory as:

> glucose >180 mg/dL or glucose <70 mg/dL within the next 60 minutes.

The event label is used for supervised training/evaluation, not as an input to the live reliability decision.

## Reliability states

- **Forecast** — model failure risk is below the caution threshold.
- **Caution** — elevated failure risk; the clinician sees the warning and supporting reason codes.
- **Abstain** — failure risk exceeds the abstention threshold; the event alert and what-if simulation are withheld rather than presented with false certainty.

Thresholds are selected from training-only calibrated OOF failure risk.

## Evaluation

The development evaluation uses a seven-day training period and a three-day temporal holdout on the canonical five-minute grid. Reliability training uses patient-grouped OOF splits. Confidence intervals are generated with patient-level bootstrap resampling.

Current repository headline metrics are explicitly labelled synthetic development results. CGMacros is supported as an optional real-data pipeline; raw participant-level data are not committed to the repository.

## Safety boundary

MetaTwin is a research prototype. It does not diagnose, prescribe treatment, recommend medication/dosing, or replace clinician judgment. What-if outputs are model scenarios and are not causal treatment recommendations.
