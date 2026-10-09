"""Leakage-safe reliability detector for MetaTwin.

The forecaster is trained on the first seven patient-days. Failure labels are
created from out-of-fold (OOF) forecasts inside those training days, so the
failure detector never sees in-sample forecast residuals. Patient groups are
kept intact in GroupKFold. An isotonic calibrator maps raw failure scores to
estimated failure probability.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor, XGBClassifier


def _new_reg(seed: int = 42):
    return XGBRegressor(
        n_estimators=140, max_depth=5, learning_rate=0.06,
        subsample=0.9, colsample_bytree=0.9, random_state=seed,
        objective="reg:squarederror", n_jobs=2,
    )


def _new_failure_clf(pos: int, neg: int, seed: int = 42):
    return XGBClassifier(
        n_estimators=120, max_depth=4, learning_rate=0.06,
        subsample=0.9, colsample_bytree=0.9, random_state=seed,
        eval_metric="logloss", n_jobs=2,
        scale_pos_weight=max(1.0, neg / max(1, pos)),
    )


def build_oof_failure_detector(train: pd.DataFrame, cols: list[str], horizon: int = 12,
                                n_splits: int = 3, seed: int = 42):
    """Return failure model, isotonic calibration and training-only thresholds.

    Failure definition: absolute 60-minute forecast error > 30 mg/dL.
    Every residual used as a failure label comes from a patient-held-out OOF
    forecast. Future glucose is never an input feature to the detector.
    """
    d = train.dropna(subset=cols + ["glucose"]).copy()
    d["target_60"] = d.groupby("patient_id").glucose.shift(-horizon)
    d = d.dropna(subset=["target_60"]).copy()

    groups = d["patient_id"].to_numpy()
    unique_groups = np.unique(groups)
    n_splits = min(n_splits, len(unique_groups))
    if n_splits < 2:
        raise ValueError("Need at least two patient groups for OOF reliability training.")

    oof = np.full(len(d), np.nan, dtype=float)
    gkf = GroupKFold(n_splits=n_splits)
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(d, groups=groups)):
        model = _new_reg(seed + fold)
        model.fit(d.iloc[tr_idx][cols], d.iloc[tr_idx]["target_60"])
        oof[va_idx] = model.predict(d.iloc[va_idx][cols])

    d["oof_prediction"] = oof
    d["failure"] = (np.abs(d["target_60"] - d["oof_prediction"]) > 30.0).astype(int)

    pos = int(d.failure.sum())
    neg = int(len(d) - pos)
    if pos < 5:
        raise ValueError("Too few OOF failure examples to train a reliability detector.")

    raw = _new_failure_clf(pos, neg, seed)
    raw.fit(d[cols], d.failure)
    raw_p = raw.predict_proba(d[cols])[:, 1]

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_p, d.failure)
    calibrated = np.asarray(calibrator.predict(raw_p), dtype=float)

    # Thresholds are selected from training-only calibrated risk. Holdout data
    # is never used to choose the Forecast/Caution/Abstain boundaries.
    caution_threshold = float(np.quantile(calibrated, 0.80))
    abstain_threshold = float(np.quantile(calibrated, 0.95))
    abstain_threshold = max(abstain_threshold, caution_threshold + 1e-6)

    return raw, calibrator, caution_threshold, abstain_threshold, d


def calibrated_failure_probability(model, calibrator, frame: pd.DataFrame, cols: list[str]):
    raw = model.predict_proba(frame[cols])[:, 1]
    return np.asarray(calibrator.predict(raw), dtype=float)
