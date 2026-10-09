"""Leakage-safe MetaTwin research evaluation.

Default: evaluate the lightweight repository dataset.
Real-data run: METATWIN_DATA_ROOT=data/processed_real python scripts/evaluate.py

The evaluation uses a seven-day training period and a three-day temporal
holdout. Reliability labels come from patient-grouped OOF forecast residuals;
classifier calibration comes from a second patient-grouped OOF pass. Bootstrap
confidence intervals resample whole patients.
"""
from __future__ import annotations

from pathlib import Path
import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             roc_auc_score, average_precision_score,
                             precision_score, recall_score, f1_score,
                             confusion_matrix)
from xgboost import XGBRegressor, XGBClassifier
from src.reliability import build_oof_failure_detector, calibrated_failure_probability

BASE = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("METATWIN_DATA_ROOT", str(BASE / "data" / "processed")))
TABLES = BASE / "results" / "tables"
H = 12
SEED = 42
DYNAMIC = ["glucose", "heart_rate", "mets", "meal_carbs", "meal_protein", "meal_fat", "meal_fiber", "glucose_slope", "glucose_mean_30", "glucose_std_30", "hour_sin", "hour_cos"]
STATIC = ["age", "bmi", "hba1c", "fasting_glucose", "baseline_hr"]
COLS = DYNAMIC + STATIC


def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


def future_max(s):
    return s.shift(-1).iloc[::-1].rolling(H, min_periods=H).max().iloc[::-1]


def future_min(s):
    return s.shift(-1).iloc[::-1].rolling(H, min_periods=H).min().iloc[::-1]


def load_data():
    ts = pd.read_csv(DATA / "timeseries.csv", parse_dates=["timestamp"])
    ehr = pd.read_csv(DATA / "ehr.csv")
    ts = ts.sort_values(["patient_id", "timestamp"]).copy()
    g = ts.groupby("patient_id").glucose
    ts["glucose_slope"] = g.transform(lambda s: s.diff(3) / 15.0)
    ts["glucose_mean_30"] = g.transform(lambda s: s.rolling(6, min_periods=1).mean())
    ts["glucose_std_30"] = g.transform(lambda s: s.rolling(6, min_periods=2).std()).fillna(0)
    ts["hour"] = ts.timestamp.dt.hour + ts.timestamp.dt.minute / 60.0
    ts["hour_sin"] = np.sin(2 * np.pi * ts.hour / 24)
    ts["hour_cos"] = np.cos(2 * np.pi * ts.hour / 24)
    ts["future_max"] = ts.groupby("patient_id").glucose.transform(future_max)
    ts["future_min"] = ts.groupby("patient_id").glucose.transform(future_min)
    ts["event"] = ((ts.future_max > 180) | (ts.future_min < 70)).astype(int)
    first = ts.groupby("patient_id").timestamp.transform("min")
    ts["day"] = (ts.timestamp - first).dt.total_seconds() / 86400.0
    x = ts.merge(ehr, on="patient_id", how="left")
    for c in COLS:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    return x


def train_forecaster(train):
    d = train.dropna(subset=COLS).copy()
    d["target_60"] = d.groupby("patient_id").glucose.shift(-H)
    d = d.dropna(subset=["target_60"])
    model = XGBRegressor(n_estimators=180, max_depth=5, learning_rate=0.06,
                         subsample=0.9, colsample_bytree=0.9, random_state=SEED,
                         objective="reg:squarederror", n_jobs=2)
    model.fit(d[COLS], d.target_60)
    return model


def train_event_model(train):
    d = train.dropna(subset=COLS + ["event"]).copy()
    pos = max(1, int(d.event.sum())); neg = max(1, len(d) - pos)
    model = XGBClassifier(n_estimators=140, max_depth=4, learning_rate=0.06,
                          subsample=0.9, colsample_bytree=0.9, random_state=SEED,
                          eval_metric="logloss", n_jobs=2, scale_pos_weight=neg / pos)
    model.fit(d[COLS], d.event.astype(int))
    return model


def patient_bootstrap(df, pred_col, target_col, metric_fn, n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    pids = df.patient_id.unique(); values = []
    for _ in range(n):
        sampled = rng.choice(pids, size=len(pids), replace=True)
        b = pd.concat([df[df.patient_id == pid] for pid in sampled], ignore_index=True)
        values.append(metric_fn(b[pred_col].to_numpy(), b[target_col].to_numpy()))
    return float(np.quantile(values, .025)), float(np.quantile(values, .975))


def event_stats(y, pred, rows_per_day=288):
    _, fp, _, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    days = max(1e-9, len(y) / rows_per_day)
    return {"precision": float(precision_score(y, pred, zero_division=0)), "recall": float(recall_score(y, pred, zero_division=0)), "f1": float(f1_score(y, pred, zero_division=0)), "false_alarms": int(fp), "false_alarms_per_day": float(fp / days), "true_positives": int(tp), "events": int(y.sum()), "samples": int(len(y))}


def main():
    x = load_data(); train = x[x.day < 7].copy(); hold = x[x.day >= 7].copy()
    if x.patient_id.nunique() < 10:
        raise ValueError("Evaluation requires at least 10 patients.")

    reg = train_forecaster(train)
    hold_reg = hold.dropna(subset=COLS).copy()
    hold_reg["actual_60"] = hold_reg.groupby("patient_id").glucose.shift(-H)
    hold_reg = hold_reg.dropna(subset=["actual_60"])
    hold_reg["prediction"] = reg.predict(hold_reg[COLS]); hold_reg["persistence"] = hold_reg.glucose

    event_model = train_event_model(train)
    hold_event = hold.dropna(subset=COLS + ["event"]).copy()
    hold_event["event_probability"] = event_model.predict_proba(hold_event[COLS])[:, 1]
    hold_event["event_prediction"] = (hold_event.event_probability >= 0.5).astype(int)

    failure_model, calibrator, caution_t, abstain_t, _ = build_oof_failure_detector(train, COLS, horizon=H, n_splits=3, seed=SEED)
    hold_event["failure_probability"] = calibrated_failure_probability(failure_model, calibrator, hold_event, COLS)
    hold_event["trust_state"] = np.select([hold_event.failure_probability >= abstain_t, hold_event.failure_probability >= caution_t], ["Abstain", "Caution"], default="Forecast")
    hold_event["gated_prediction"] = ((hold_event.event_prediction == 1) & (hold_event.trust_state != "Abstain")).astype(int)

    mae = mean_absolute_error(hold_reg.actual_60, hold_reg.prediction); rmse_value = rmse(hold_reg.actual_60, hold_reg.prediction)
    p_mae = mean_absolute_error(hold_reg.actual_60, hold_reg.persistence); p_rmse = rmse(hold_reg.actual_60, hold_reg.persistence)
    mae_ci = patient_bootstrap(hold_reg, "prediction", "actual_60", mean_absolute_error)
    rmse_ci = patient_bootstrap(hold_reg, "prediction", "actual_60", rmse)

    hold_reg["failure"] = (np.abs(hold_reg.actual_60 - hold_reg.prediction) > 30).astype(int)
    common = hold_event.merge(hold_reg[["patient_id", "timestamp", "failure"]], on=["patient_id", "timestamp"], how="inner")
    y_fail = common.failure.to_numpy(); p_fail = common.failure_probability.to_numpy()
    auroc = roc_auc_score(y_fail, p_fail) if len(np.unique(y_fail)) == 2 else float("nan")
    auprc = average_precision_score(y_fail, p_fail) if y_fail.sum() else float("nan")

    y = hold_event.event.to_numpy().astype(int); ungated = hold_event.event_prediction.to_numpy().astype(int); gated = hold_event.gated_prediction.to_numpy().astype(int)
    event_metrics = {"window_minutes": 60, "event_definition": "glucose >180 mg/dL or <70 mg/dL within 60 minutes", "without_gating": event_stats(y, ungated), "with_reliability_gating": event_stats(y, gated), "failure_detector_auroc": float(auroc), "failure_detector_auprc": float(auprc), "reliability_thresholds": {"caution": caution_t, "abstain": abstain_t}, "trust_coverage": {s: float((hold_event.trust_state == s).mean() * 100) for s in ["Forecast", "Caution", "Abstain"]}, "validation": "temporal holdout; patient-grouped OOF reliability training and calibration", "dataset_root": str(DATA)}

    rng = np.random.default_rng(SEED); pids = hold_event.patient_id.unique(); boot = {"precision": [], "recall": [], "f1": []}
    for _ in range(1000):
        sampled = rng.choice(pids, size=len(pids), replace=True)
        b = pd.concat([hold_event[hold_event.patient_id == p] for p in sampled], ignore_index=True)
        by = b.event.to_numpy().astype(int); bp = b.event_prediction.to_numpy().astype(int)
        boot["precision"].append(precision_score(by, bp, zero_division=0)); boot["recall"].append(recall_score(by, bp, zero_division=0)); boot["f1"].append(f1_score(by, bp, zero_division=0))
    event_ci = {k: [float(np.quantile(v, .025)), float(np.quantile(v, .975))] for k, v in boot.items()}

    merged = hold_reg.merge(hold_event[["patient_id", "timestamp", "failure_probability"]], on=["patient_id", "timestamp"], how="inner")
    abst = []
    for coverage in np.arange(1.0, 0.49, -0.05):
        threshold = float(np.quantile(merged.failure_probability, coverage)); keep = merged.failure_probability <= threshold
        if keep.sum(): abst.append({"coverage": float(keep.mean() * 100), "mae": float(mean_absolute_error(merged.loc[keep, "actual_60"], merged.loc[keep, "prediction"]))})

    cal = hold_event.copy(); cal["bin"] = pd.qcut(cal.event_probability.rank(method="first"), 5, labels=False)
    calibration = cal.groupby("bin").agg(predicted=("event_probability", "mean"), observed=("event", "mean")).reset_index(drop=True)

    TABLES.mkdir(parents=True, exist_ok=True)
    dataset_label = "CGMacros v1.0.0" if DATA.name == "processed_real" else "synthetic development data"
    rows = [
        {"metric": "dataset", "value": dataset_label}, {"metric": "data_patients", "value": int(x.patient_id.nunique())}, {"metric": "data_observations", "value": int(len(x))},
        {"metric": "mae_persistence", "value": p_mae}, {"metric": "mae_xgboost", "value": mae, "ci_low": mae_ci[0], "ci_high": mae_ci[1]},
        {"metric": "rmse_persistence", "value": p_rmse}, {"metric": "rmse_xgboost", "value": rmse_value, "ci_low": rmse_ci[0], "ci_high": rmse_ci[1]},
        {"metric": "auroc_failure", "value": auroc}, {"metric": "auprc_failure", "value": auprc}, {"metric": "failure_prevalence", "value": float(y_fail.mean())},
        {"metric": "event_precision", "value": event_metrics["without_gating"]["precision"], "ci_low": event_ci["precision"][0], "ci_high": event_ci["precision"][1]},
        {"metric": "event_recall", "value": event_metrics["without_gating"]["recall"], "ci_low": event_ci["recall"][0], "ci_high": event_ci["recall"][1]},
        {"metric": "event_f1", "value": event_metrics["without_gating"]["f1"], "ci_low": event_ci["f1"][0], "ci_high": event_ci["f1"][1]},
        {"metric": "gated_precision", "value": event_metrics["with_reliability_gating"]["precision"]}, {"metric": "gated_recall", "value": event_metrics["with_reliability_gating"]["recall"]}, {"metric": "gated_f1", "value": event_metrics["with_reliability_gating"]["f1"]}, {"metric": "gated_false_alarms_per_day", "value": event_metrics["with_reliability_gating"]["false_alarms_per_day"]},
    ]
    pd.DataFrame(rows).to_csv(TABLES / "model_summary.csv", index=False); pd.DataFrame(abst).to_csv(TABLES / "abstention_curve.csv", index=False); calibration.to_csv(TABLES / "calibration.csv", index=False)
    (TABLES / "event_metrics.json").write_text(json.dumps(event_metrics, indent=2)); (TABLES / "event_metrics_ci.json").write_text(json.dumps(event_ci, indent=2))
    (TABLES / "data_provenance.json").write_text(json.dumps({"dataset": dataset_label, "source": "PhysioNet CGMacros v1.0.0" if DATA.name == "processed_real" else "repository synthetic development generator", "patients": int(x.patient_id.nunique()), "observations": int(len(x)), "training_days": 7, "holdout_days": 3, "forecast_horizon_minutes": 60, "event_definition": ">180 mg/dL or <70 mg/dL within 60 minutes", "failure_definition": "absolute 60-minute forecast error >30 mg/dL", "reliability": "patient-grouped OOF residuals + second grouped OOF classifier calibration + isotonic regression", "bootstrap": "1000 patient-level resamples, 95% percentile intervals", "raw_data_committed": False}, indent=2))
    (TABLES / "reliability_thresholds.json").write_text(json.dumps({"caution": caution_t, "abstain": abstain_t, "method": "training-only 80th/95th quantiles of calibrated OOF failure risk"}, indent=2))
    print(json.dumps({"forecast": {"mae": mae, "rmse": rmse_value, "mae_ci": mae_ci, "rmse_ci": rmse_ci}, "event": event_metrics, "event_ci": event_ci}, indent=2))


if __name__ == "__main__": main()
