"""MetaTwin API — clinician-facing digital twin inference and research results."""
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBRegressor, XGBClassifier

from src.reliability import build_oof_failure_detector, calibrated_failure_probability

BASE = Path(__file__).parent
TABLES = BASE / "results" / "tables"
DATA = BASE / "data" / "processed"
H = 12  # 60 minutes at the canonical five-minute analysis grid

app = FastAPI(
    title="MetaTwin API",
    description="Glucose digital twin: forecast, adverse-event prediction and reliability gating",
    version="0.3.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])


def _summary():
    df = pd.read_csv(TABLES / "model_summary.csv")
    return dict(zip(df.metric, df.value))


def _future_max(s):
    return s.shift(-1).iloc[::-1].rolling(H, min_periods=H).max().iloc[::-1]


def _future_min(s):
    return s.shift(-1).iloc[::-1].rolling(H, min_periods=H).min().iloc[::-1]


def _features(ts, ehr):
    x = ts.sort_values(["patient_id", "timestamp"]).copy()
    g = x.groupby("patient_id").glucose
    x["glucose_slope"] = g.transform(lambda s: s.diff(3) / 15.0)
    x["glucose_mean_30"] = g.transform(lambda s: s.rolling(6, min_periods=1).mean())
    x["glucose_std_30"] = g.transform(lambda s: s.rolling(6, min_periods=2).std()).fillna(0)
    x["hour"] = x.timestamp.dt.hour + x.timestamp.dt.minute / 60.0
    x["hour_sin"] = np.sin(2 * np.pi * x.hour / 24)
    x["hour_cos"] = np.cos(2 * np.pi * x.hour / 24)
    x["future_max"] = x.groupby("patient_id").glucose.transform(_future_max)
    x["future_min"] = x.groupby("patient_id").glucose.transform(_future_min)
    x["event"] = ((x.future_max > 180) | (x.future_min < 70)).astype(int)
    first = x.groupby("patient_id").timestamp.transform("min")
    x["day"] = (x.timestamp - first).dt.total_seconds() / 86400.0
    x = x.merge(ehr, on="patient_id", how="left")
    dynamic = ["glucose", "heart_rate", "mets", "meal_carbs", "meal_protein", "meal_fat", "meal_fiber", "glucose_slope", "glucose_mean_30", "glucose_std_30", "hour_sin", "hour_cos"]
    static = ["age", "bmi", "hba1c", "fasting_glucose", "baseline_hr"]
    cols = dynamic + static
    for c in cols:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    return x, cols


@lru_cache(maxsize=1)
def build_models():
    ts = pd.read_csv(DATA / "timeseries.csv", parse_dates=["timestamp"])
    ehr = pd.read_csv(DATA / "ehr.csv")
    x, cols = _features(ts, ehr)
    usable = x.dropna(subset=cols + ["future_max", "future_min"]).copy()
    train = usable[usable.day < 7].copy()

    reg_target = train.groupby("patient_id").glucose.shift(-H)
    reg_mask = reg_target.notna()
    reg = XGBRegressor(n_estimators=180, max_depth=5, learning_rate=0.06, subsample=0.9, colsample_bytree=0.9, random_state=42, objective="reg:squarederror", n_jobs=2)
    reg.fit(train.loc[reg_mask, cols], reg_target.loc[reg_mask])

    event_train = train.dropna(subset=["event"]).copy()
    pos = max(1, int(event_train.event.sum())); neg = max(1, len(event_train) - pos)
    clf = XGBClassifier(n_estimators=140, max_depth=4, learning_rate=0.06, subsample=0.9, colsample_bytree=0.9, random_state=42, eval_metric="logloss", n_jobs=2, scale_pos_weight=neg / pos)
    clf.fit(event_train[cols], event_train.event.astype(int))

    failure_model, calibrator, caution_t, abstain_t, _ = build_oof_failure_detector(train, cols, horizon=H, n_splits=3, seed=42)
    return x, ehr, cols, reg, clf, failure_model, calibrator, caution_t, abstain_t


def _enriched_ehr(row):
    r = row.to_dict(); age, bmi, hba1c = float(r["age"]), float(r["bmi"]), float(r["hba1c"])
    r["hypertension"] = bool(age > 55 or bmi > 31)
    r["dyslipidemia"] = bool(bmi > 29 or hba1c > 6.5)
    r["ldl"] = round(float(np.clip(115 + (bmi - 26) * 3 + (hba1c - 5.5) * 5, 50, 220)), 1)
    r["hdl"] = round(float(np.clip(54 - (bmi - 26) * 0.7, 25, 90)), 1)
    r["triglycerides"] = round(float(np.clip(130 + (bmi - 26) * 5 + (hba1c - 5.5) * 12, 50, 400)), 1)
    r["genetic_risk_score"] = round(float((age - 50) / 25 + (hba1c - 5.8) / 2), 2)
    r["ehr_extension_status"] = "simulated extension"
    return r


def _trust_state(risk, caution_t, abstain_t):
    if risk >= abstain_t: return "Abstain"
    if risk >= caution_t: return "Caution"
    return "Forecast"


def _patient_payload(pid):
    x, ehr, cols, reg, clf, failure_model, calibrator, caution_t, abstain_t = build_models()
    rows = x[(x.patient_id == pid) & (x.day >= 7)].copy()
    if rows.empty: raise HTTPException(404, "Patient not found")
    row = rows.iloc[min(100, max(0, len(rows) - H - 1))]
    feats = row[cols].to_frame().T.copy()
    for c in cols: feats[c] = pd.to_numeric(feats[c], errors="coerce")
    if feats[cols].isna().any().any(): raise HTTPException(500, "Model features contain missing or non-numeric values")

    forecast = float(reg.predict(feats[cols])[0])
    event_prob = float(clf.predict_proba(feats[cols])[0, 1])
    failure_risk = float(calibrated_failure_probability(failure_model, calibrator, feats, cols)[0])
    current = float(row.glucose); slope = float(row.glucose_slope if pd.notna(row.glucose_slope) else 0)
    points = [current + (forecast - current) * (i / H) for i in range(H + 1)]
    future_max = float(row.future_max); future_min = float(row.future_min)

    if forecast >= 180: event_type = "Hyperglycemic event risk"
    elif forecast < 70: event_type = "Hypoglycemic event risk"
    elif event_prob >= 0.5: event_type = "Threshold event risk"
    else: event_type = "No threshold event predicted"
    trust = _trust_state(failure_risk, caution_t, abstain_t)
    gated = event_prob >= 0.5 and trust != "Abstain"

    reasons = []
    volatility = float(row.glucose_std_30)
    if volatility > 8: reasons.append("High recent CGM volatility")
    if abs(slope) > 0.7: reasons.append("Rapid glucose trend")
    if float(row.meal_carbs) > 0: reasons.append("Recent meal signal")
    if any(pd.isna(row.get(c)) for c in ["heart_rate", "mets", "meal_carbs"]): reasons.append("Missing wearable/meal signal")
    if not reasons: reasons.append("Stable recent signal pattern")

    scenarios = []
    if trust != "Abstain":
        for sid, label, carb_scale, mets_delta in [("baseline", "Baseline", 1.0, 0.0), ("walk_20", "20-minute activity scenario", 1.0, 2.0), ("reduced_carbs", "20% lower carbohydrate scenario", 0.8, 0.0)]:
            sf = feats.copy(); sf["meal_carbs"] *= carb_scale; sf["mets"] += mets_delta
            sf["glucose_slope"] -= 0.05 if mets_delta else 0
            sf["glucose_mean_30"] -= 2.0 if mets_delta else 0
            scenarios.append({"id": sid, "label": label, "forecast_endpoint": round(float(reg.predict(sf[cols])[0]), 1), "event_probability": round(float(clf.predict_proba(sf[cols])[0, 1]), 4)})
    else:
        scenarios = [{"id": "blocked", "label": "What-if disabled while Abstain", "enabled": False}]

    profile = _enriched_ehr(ehr[ehr.patient_id == pid].iloc[0])
    return {
        "patient": profile,
        "data_streams": {"dynamic": ["CGM", "heart rate", "METs", "meal macros"], "static": ["age", "BMI", "HbA1c", "diagnoses", "labs", "genetic risk (simulated)"]},
        "current": {"timestamp": row.timestamp.isoformat(), "glucose": current, "heart_rate": float(row.heart_rate), "mets": float(row.mets), "carbs_recent": float(row.meal_carbs), "trend": slope},
        "forecast": {"horizon_minutes": 60, "endpoint_glucose": forecast, "points": [round(v, 1) for v in points]},
        "event": {"type": event_type, "probability": event_prob, "thresholds": {"high": 180, "low": 70}, "window_minutes": 60, "alert_candidate": gated, "gated_by_reliability": True, "actual_future_max": future_max, "actual_future_min": future_min},
        "reliability": {"state": trust, "failure_risk": failure_risk, "thresholds": {"caution": caution_t, "abstain": abstain_t}, "reason_codes": reasons, "method": "OOF patient-grouped residual detector + isotonic calibration"},
        "what_if": scenarios,
        "research_note": "Research prototype. Reliability labels use OOF forecasting residuals; final headline evaluation should be generated by scripts/evaluate.py with patient-level bootstrap confidence intervals.",
    }


def _event_metrics():
    x, _, cols, _, clf, failure_model, calibrator, caution_t, abstain_t = build_models()
    hold = x[x.day >= 7].dropna(subset=cols + ["event"]).copy()
    hold["event_probability"] = clf.predict_proba(hold[cols])[:, 1]
    hold["failure_probability"] = calibrated_failure_probability(failure_model, calibrator, hold, cols)
    hold["event_prediction"] = (hold.event_probability >= 0.5).astype(int)
    hold["gated_prediction"] = ((hold.event_prediction == 1) & (hold.failure_probability < abstain_t)).astype(int)
    y = hold.event.astype(int).to_numpy()

    def stats(p):
        _, fp, _, tp = confusion_matrix(y, p, labels=[0, 1]).ravel(); days = max(1e-9, len(hold) / 288)
        return {"precision": float(precision_score(y, p, zero_division=0)), "recall": float(recall_score(y, p, zero_division=0)), "f1": float(f1_score(y, p, zero_division=0)), "false_alarms": int(fp), "false_alarms_per_day": float(fp / days), "true_positives": int(tp), "events": int(y.sum()), "samples": int(len(y))}

    return {"window_minutes": 60, "event_definition": "glucose >180 mg/dL or <70 mg/dL within 60 minutes", "without_gating": stats(hold.event_prediction.to_numpy()), "with_reliability_gating": stats(hold.gated_prediction.to_numpy()), "reliability_thresholds": {"caution": caution_t, "abstain": abstain_t}, "validation": "temporal holdout; reliability detector trained from patient-grouped OOF residuals"}


@app.get("/")
def root(): return {"name": "MetaTwin API", "description": "Glucose digital twin — forecast, adverse event and reliability", "endpoints": ["/api/all", "/api/patient/1", "/api/cohort", "/api/event-metrics", "/api/summary", "/docs"]}

@app.get("/api/summary")
def summary(): return _summary()

@app.get("/api/abstention")
def abstention(): return pd.read_csv(TABLES / "abstention_curve.csv").to_dict(orient="records")

@app.get("/api/calibration")
def calibration(): return pd.read_csv(TABLES / "calibration.csv").to_dict(orient="records")

@app.get("/api/patient/{patient_id}")
def patient(patient_id: int): return _patient_payload(patient_id)

@app.get("/api/cohort")
def cohort():
    _, ehr, _, _, _, _, _, _, _ = build_models(); out = []
    for pid in sorted(ehr.patient_id.unique())[:20]:
        p = _patient_payload(int(pid)); out.append({"patient_id": int(pid), "status": p["patient"]["diabetes_status"], "event_probability": p["event"]["probability"], "reliability": p["reliability"]["state"], "event": p["event"]["type"], "alert": p["event"]["alert_candidate"]})
    return out

@app.get("/api/event-metrics")
def event_metrics(): return _event_metrics()

@app.get("/api/all")
def all_results():
    s = _summary(); a = pd.read_csv(TABLES / "abstention_curve.csv"); c = pd.read_csv(TABLES / "calibration.csv")
    return {"summary": s, "abstention": a.to_dict(orient="records"), "calibration": c.to_dict(orient="records"), "patient": _patient_payload(1), "event_metrics": _event_metrics()}
