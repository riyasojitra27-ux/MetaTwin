# MetaTwin

**MetaTwin is a glucose digital twin that forecasts glucose 60 minutes ahead, predicts a specific adverse glucose event, estimates forecast reliability, and abstains when the forecast cannot be trusted.**

## Challenge coverage

The application is organized around the required clinician workflow:

**Static EHR + dynamic wearable/CGM/meal data → Digital Twin → 60-minute glucose forecast → adverse-event prediction (>180 or <70 mg/dL) → reliability gate → Forecast / Caution / Abstain.**

### App modules

- **Clinician:** virtual patient, static EHR stream, dynamic state, glucose forecast and gated event decision.
- **Events:** adverse-event probability, thresholds, 60-minute window and with/without reliability-gating evaluation.
- **Reliability:** Forecast/Caution/Abstain state, failure-risk reasons and clinician risk slider demonstration.
- **What-If:** bounded baseline, 20-minute walk and reduced-carbohydrate scenarios, disabled when reliability is Abstain.
- **Cohort:** ranked virtual-patient triage view.
- **Research:** forecasting and reliability metrics plus abstention/calibration evidence areas.
- **Model Card:** intended use, data, limitations, safety and non-clinical-use boundary.

## Data and validation status

The repository currently contains synthetic development data. Synthetic EHR extensions such as diagnoses, laboratory values and genetic risk are explicitly labeled as simulated. The temporal event-inference API trains on earlier days and demonstrates inference on later days; it is **not** the final headline experiment.

Final research results should use the specified out-of-fold residual pipeline, patient-level bootstrap confidence intervals, calibrated reliability detector and event-layer evaluation. CGMacros remains the planned primary real/open-data experiment when preprocessing is complete.

## Safety

MetaTwin is a research prototype, not a medical device. It does not diagnose conditions, prescribe treatment or provide medication/dosing advice. Event alerts are research outputs and do not replace clinical judgment.

## Running locally

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

Set `VITE_API_BASE_URL` to the backend URL for deployed use.

## Team

Riya Sojitra · BE student
