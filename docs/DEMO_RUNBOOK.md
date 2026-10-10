# MetaTwin Demo Runbook

## Core 3-minute story

### 0:00–0:20 — Problem

"A glucose forecast is only useful when we know when to trust it. MetaTwin is a glucose digital twin that combines static EHR context with dynamic CGM, wearable and meal signals. It predicts the glucose trajectory, predicts a specific adverse event within 60 minutes, and estimates whether that prediction is reliable enough to surface."

### 0:20–0:55 — Digital twin

Open the **Clinician** tab.

Point out:
- Static EHR stream: age, BMI, HbA1c, diabetes status and baseline context.
- Dynamic stream: CGM, heart rate, activity and meal macros.
- The virtual patient is the fused state representation.

Say:

"This is not only a retrospective chart. The two streams are fused into a current patient state that drives the next prediction."

### 0:55–1:25 — Forecast and adverse event

Show the 60-minute forecast chart and the 180/70 mg/dL thresholds.

Say:

"The primary prediction task is the adverse glucose event: glucose above 180 or below 70 mg/dL within the next 60 minutes. The forecast provides the trajectory; the event layer turns that trajectory into a clinically interpretable risk."

Open **Events** and point out event probability, thresholds, and the reliability-gated decision.

### 1:25–1:55 — Reliability / abstention

Open **Reliability**.

Say:

"The differentiator is that MetaTwin can refuse false certainty. A separate failure detector estimates when the glucose forecast is likely to be wrong using only information available at prediction time. Its calibrated risk maps to Forecast, Caution or Abstain."

Point to reason codes and the doctor's risk slider.

### 1:55–2:25 — What-if

Open **What-If**.

Show the activity and reduced-carbohydrate scenarios only when the state is not Abstain.

Say:

"What-if scenarios are deliberately bounded. They are model demonstrations, not causal treatment recommendations, and they are disabled when reliability is insufficient."

### 2:25–2:50 — Evidence

Open **Research**.

Use the synthetic-development results only with the dataset label visible.

Say:

"On the included synthetic development holdout, the 60-minute forecast MAE is 8.96 mg/dL. Reliability gating reduced false alarms from 6.88 to 5.97 per day, about 13.3%, while precision remained essentially unchanged. The recall tradeoff is shown rather than hidden."

### 2:50–3:00 — Boundary

Say:

"This is a research prototype, not a medical device. It does not diagnose or prescribe treatment. The contribution is a digital twin that predicts an adverse event and explicitly measures when its own forecast should not be trusted."

## Judge-safe claims

Use:
- "synthetic development data"
- "temporal holdout"
- "patient-grouped OOF reliability training"
- "calibrated reliability gate"
- "research prototype"
- "model scenario, not causal treatment recommendation"

Do not use:
- "clinically validated"
- "doctor-approved"
- "works for Indian patients"
- "causes glucose improvement"
- "first ever"
- "guaranteed prediction"

## Current development metrics

- Forecast MAE: 8.96 mg/dL (95% CI 8.57–9.38)
- Forecast RMSE: 13.57 mg/dL (95% CI 12.93–14.28)
- Failure-detector AUROC: 0.673
- Event precision without gating: 93.48%
- Event recall without gating: 89.83%
- False alarms/day without gating: 6.88
- Event precision with reliability gating: 93.51%
- Event recall with reliability gating: 78.30%
- False alarms/day with reliability gating: 5.97
- Trust coverage: Forecast 69.23%, Caution 25.03%, Abstain 5.73%

All values above are synthetic development results, not clinical validation.

## Expected judge questions

### Why is reliability separate from event prediction?

"The event model answers what may happen. The reliability model answers whether the forecast supporting that decision is likely to be trustworthy. Keeping those roles separate makes abstention measurable instead of hiding uncertainty inside a single score."

### Why abstain?

"A prediction system should not be forced to produce a confident alert when its own evidence indicates high failure risk. Abstention trades some recall for fewer false alarms and more transparent uncertainty."

### Is the what-if causal?

"No. It is a bounded model scenario used to demonstrate digital-twin interaction. We explicitly do not claim causal treatment effects."

### Why synthetic data?

"The included synthetic dataset keeps the public demo reproducible and lightweight. The repository also contains a reproducible CGMacros preprocessing and evaluation pipeline, but raw participant data are not committed."

### What makes this a digital twin rather than a normal classifier?

"The system maintains a virtual patient state from static and dynamic streams, forecasts the patient's near-term trajectory, derives an adverse-event prediction from that trajectory/context, and lets a clinician inspect reliability and conditional scenarios for that virtual patient."
