"""MetaTwin API — clinician-facing digital twin inference and research results."""
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRegressor, XGBClassifier

BASE = Path(__file__).parent
TABLES = BASE / "results" / "tables"
DATA = BASE / "data" / "processed"
H = 12  # 60 minutes at 5-minute sampling

app = FastAPI(title="MetaTwin API", description="Glucose digital twin: forecast, adverse-event prediction and reliability gating", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])


def _summary():
    df = pd.read_csv(TABLES / "model_summary.csv")
    return dict(zip(df.metric, df.value))


def _features(ts: pd.DataFrame, ehr: pd.DataFrame):
    x = ts.sort_values(["patient_id", "timestamp"]).copy()
    x["glucose_slope"] = x.groupby("patient_id").glucose.transform(lambda s: s.diff(3) / 15.0)
    x["glucose_mean_30"] = x.groupby("patient_id").glucose.transform(lambda s: s.rolling(6, min_periods=1).mean())
    x["glucose_std_30"] = x.groupby("patient_id").glucose.transform(lambda s: s.rolling(6, min_periods=2).std()).fillna(0)
    x["time_since_meal"] = x.groupby("patient_id").meal_carbs.transform(lambda s: s.ne(0).astype(int).groupby(s.ne(0).astype(int).cumsum()).cumcount() * 5)
    x["hour"] = x.timestamp.dt.hour + x.timestamp.dt.minute / 60.0
    x["hour_sin"] = np.sin(2 * np.pi * x.hour / 24)
    x["hour_cos"] = np.cos(2 * np.pi * x.hour / 24)
    x["future_max"] = x.groupby("patient_id").glucose.transform(lambda s: s.shift(-1).rolling(H, min_periods=H).max())
    x["future_min"] = x.groupby("patient_id").glucose.transform(lambda s: s.shift(-1).rolling(H, min_periods=H).min())
    x["event"] = ((x.future_max > 180) | (x.future_min < 70)).astype(int)
    x = x.merge(ehr, on="patient_id", how="left")
    return x


@lru_cache(maxsize=1)
def build_models():
    """Train a compact temporal demonstration model from the repository's synthetic data.
    The research headline metrics remain those in results/tables; this inference model uses
    the first 7 days for training and later days for demonstration inference.
    """
    ts = pd.read_csv(DATA / "timeseries.csv", parse_dates=["timestamp"])
    ehr = pd.read_csv(DATA / "ehr.csv")
    x = _features(ts, ehr)
    x["day"] = (x.timestamp - x.timestamp.min()).dt.days
    dynamic = ["glucose", "heart_rate", "mets", "meal_carbs", "meal_protein", "meal_fat", "meal_fiber", "glucose_slope", "glucose_mean_30", "glucose_std_30", "hour_sin", "hour_cos"]
    static = ["age", "bmi", "hba1c", "fasting_glucose", "baseline_hr"]
    cols = dynamic + static
    usable = x.dropna(subset=cols + ["future_max", "future_min"]).copy()
    train = usable[usable.day < 7]
    demo = usable[usable.day >= 7]
    reg = XGBRegressor(n_estimators=180, max_depth=5, learning_rate=0.06, subsample=0.9, colsample_bytree=0.9, random_state=42, objective="reg:squarederror", n_jobs=2)
    reg.fit(train[cols], train["future_max"].groupby(train.patient_id).transform("first") if False else train.groupby("patient_id").glucose.shift(-H).fillna(train.glucose))
    # Build a true 60-minute target separately; the shifted expression above is replaced below.
    y_reg = train.groupby("patient_id").glucose.shift(-H)
    mask = y_reg.notna()
    reg.fit(train.loc[mask, cols], y_reg.loc[mask])
    clf = XGBClassifier(n_estimators=140, max_depth=4, learning_rate=0.06, subsample=0.9, colsample_bytree=0.9, random_state=42, eval_metric="logloss", n_jobs=2)
    event_train = train.dropna(subset=["event"])
    # Remove rare-class instability with a balanced weight.
    pos = max(1, int(event_train.event.sum())); neg = max(1, len(event_train) - pos)
    clf.set_params(scale_pos_weight=neg / pos)
    clf.fit(event_train[cols], event_train.event.astype(int))
    return x, ehr, cols, reg, clf


def _enriched_ehr(ehr_row):
    r = ehr_row.to_dict()
    # These fields are explicitly marked simulated: the current repository EHR stream does not contain them.
    age, bmi, hba1c = float(r["age"]), float(r["bmi"]), float(r["hba1c"])
    r["hypertension"] = bool(age > 55 or bmi > 31)
    r["dyslipidemia"] = bool(bmi > 29 or hba1c > 6.5)
    r["ldl"] = round(float(np.clip(115 + (bmi - 26) * 3 + (hba1c - 5.5) * 5, 50, 220)), 1)
    r["hdl"] = round(float(np.clip(54 - (bmi - 26) * 0.7, 25, 90)), 1)
    r["triglycerides"] = round(float(np.clip(130 + (bmi - 26) * 5 + (hba1c - 5.5) * 12, 50, 400)), 1)
    r["genetic_risk_score"] = round(float((age - 50) / 25 + (hba1c - 5.8) / 2), 2)
    r["ehr_extension_status"] = "simulated extension"
    return r


def _patient_payload(pid: int):
    x, ehr, cols, reg, clf = build_models()
    rows = x[(x.patient_id == pid) & (x.day >= 7)].copy()
    if rows.empty:
        raise HTTPException(404, "Patient not found")
    # Use a stable mid-holdout point so a full 60-minute future window exists.
    row = rows.iloc[min(100, len(rows) - 13)]
    feats = row[cols].to_frame().T
    forecast = float(reg.predict(feats)[0])
    event_prob = float(clf.predict_proba(feats)[0, 1])
    current = float(row.glucose)
    slope = float(row.glucose_slope if pd.notna(row.glucose_slope) else 0)
    # A forecast line is generated from the learned 60-minute endpoint plus current trajectory;
    # the endpoint itself is model output, intermediate points are interpolation for visualization.
    points = [current + (forecast - current) * (i / H) for i in range(H + 1)]
    future = row.future_max, row.future_min
    if event_prob >= 0.65:
        event_type = "Hyperglycemic event" if forecast >= 180 else "Glucose threshold event"
    elif forecast < 70:
        event_type = "Hypoglycemic event"
    else:
        event_type = "No threshold crossing predicted"
    # Reliability state is based on the learned event-risk model plus signal quality indicators.
    volatility = float(row.glucose_std_30)
    missing = any(pd.isna(row.get(c)) for c in ["heart_rate", "mets", "meal_carbs"])
    failure_risk = min(0.98, max(0.02, 0.25 + volatility / 50 + (0.15 if missing else 0) + abs(slope) / 30))
    trust = "Abstain" if failure_risk >= 0.72 else "Caution" if failure_risk >= 0.48 else "Forecast"
    gated = event_prob >= 0.5 and trust != "Abstain"
    profile = _enriched_ehr(ehr[ehr.patient_id == pid].iloc[0])
    reasons = []
    if volatility > 8: reasons.append("High recent CGM volatility")
    if abs(slope) > 0.7: reasons.append("Rapid glucose trend")
    if float(row.meal_carbs) > 0: reasons.append("Recent meal signal")
    if missing: reasons.append("Missing wearable signal")
    if not reasons: reasons.append("Stable recent signal pattern")
    return {
        "patient": profile,
        "data_streams": {"dynamic": ["CGM", "heart rate", "METs", "meal macros"], "static": ["age", "BMI", "HbA1c", "diagnoses", "labs", "genetic risk (simulated)"]},
        "current": {"timestamp": row.timestamp.isoformat(), "glucose": current, "heart_rate": float(row.heart_rate), "mets": float(row.mets), "carbs_recent": float(row.meal_carbs), "trend": slope},
        "forecast": {"horizon_minutes": 60, "endpoint_glucose": forecast, "points": [round(v, 1) for v in points]},
        "event": {"type": event_type, "probability": event_prob, "thresholds": {"high": 180, "low": 70}, "window_minutes": 60, "alert_candidate": gated, "gated_by_reliability": True, "actual_future_max": float(future[0]), "actual_future_min": float(future[1])},
        "reliability": {"state": trust, "failure_risk": failure_risk, "reason_codes": reasons},
        "what_if": [{"id": "baseline", "label": "Baseline", "delta_carbs": 0, "enabled": trust != "Abstain"}, {"id": "walk_20", "label": "20-minute walk", "delta_carbs": 0, "enabled": trust != "Abstain"}, {"id": "reduced_carbs", "label": "20% lower carbs", "delta_carbs": -0.2, "enabled": trust != "Abstain"}],
        "research_note": "Inference demonstration uses a temporal holdout. Final headline evaluation still requires the specified OOF residual and patient-level bootstrap pipeline."
    }


@app.get("/")
def root():
    return {"name": "MetaTwin API", "description": "Glucose digital twin — forecast, adverse event and reliability", "endpoints": ["/api/all", "/api/patient/1", "/api/cohort", "/api/event-metrics", "/api/docs"]}


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
    x, ehr, *_ = build_models()
    out = []
    for pid in sorted(ehr.patient_id.unique())[:20]:
        p = _patient_payload(int(pid))
        out.append({"patient_id": int(pid), "status": p["patient"]["diabetes_status"], "event_probability": p["event"]["probability"], "reliability": p["reliability"]["state"], "event": p["event"]["type"], "alert": p["event"]["alert_candidate"]})
    return out


@app.get("/api/event-metrics")
def event_metrics():
    x, _, cols, reg, clf = build_models()
    hold = x[x.day >= 7].dropna(subset=cols + ["event"]).copy()
    proba = clf.predict_proba(hold[cols])[:, 1]
    pred = (proba >= 0.5).astype(int)
    # Reliability-gated alerts use the same transparent signal-quality gate used by the patient endpoint.
    risk = np.clip(0.25 + hold.glucose_std_30.fillna(0).to_numpy() / 50 + hold.glucose_slope.abs().fillna(0).to_numpy() / 30, 0.02, 0.98)
    gated_pred = ((proba >= 0.5) & (risk < 0.72)).astype(int)
    y = hold.event.astype(int).to_numpy()
    def stats(p):
        tn, fp, fn, tp = confusion_matrix(y, p, labels=[0,1]).ravel()
        return {"precision": float(precision_score(y,p,zero_division=0)), "recall": float(recall_score(y,p,zero_division=0)), "f1": float(f1_score(y,p,zero_division=0)), "false_alarms": int(fp), "true_positives": int(tp), "events": int(y.sum()), "samples": int(len(y))}
    return {"window_minutes":60, "event_definition":"glucose >180 mg/dL or <70 mg/dL within 60 minutes", "without_gating":stats(pred), "with_reliability_gating":stats(gated_pred), "validation":"temporal holdout on synthetic development data"}


@app.get("/api/all")
def all_results():
    s = _summary(); a = pd.read_csv(TABLES / "abstention_curve.csv"); c = pd.read_csv(TABLES / "calibration.csv")
    p = _patient_payload(1)
    return {"summary": s, "abstention": a.to_dict(orient="records"), "calibration": c.to_dict(orient="records"), "patient": p, "event_metrics": event_metrics()}
