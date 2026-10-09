"""Prepare the public CGMacros dataset for MetaTwin.

Raw CGMacros is intentionally NOT committed to GitHub. It is ~627 MB and is
licensed separately by PhysioNet. Download it from the official source, extract
it locally, then run this script.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def _find_bio(root: Path) -> Path:
    matches = list(root.rglob("bio.csv")) + list(root.rglob("Bio.csv"))
    if not matches:
        raise FileNotFoundError("Could not find bio.csv in the extracted CGMacros directory.")
    return matches[0]


def _find_subject_files(root: Path):
    return sorted(p for p in root.rglob("*.csv")
                  if "cgm" in p.name.lower() and p.name.lower() != "bio.csv")


def _first_col(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None


def _numeric(s):
    return pd.to_numeric(s, errors="coerce")


def _read_subject(path: Path, pid: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    ts_col = _first_col(df, ["Timestamp", "timestamp"])
    dex_col = _first_col(df, ["Dexcom GL"])
    lib_col = _first_col(df, ["Libre GL"])
    hr_col = _first_col(df, ["HR"])
    mets_col = _first_col(df, ["Mets", "METs"])
    if not ts_col or not (dex_col or lib_col):
        raise ValueError(f"Skipping {path}: missing Timestamp or CGM columns")

    out = pd.DataFrame({"timestamp": pd.to_datetime(df[ts_col], errors="coerce")})
    dex = _numeric(df[dex_col]) if dex_col else pd.Series(np.nan, index=df.index)
    lib = _numeric(df[lib_col]) if lib_col else pd.Series(np.nan, index=df.index)
    out["glucose"] = dex.combine_first(lib)
    out["heart_rate"] = _numeric(df[hr_col]) if hr_col else np.nan
    out["mets"] = (_numeric(df[mets_col]) / 10.0) if mets_col else np.nan
    for src, dst in [("Carbs", "meal_carbs"), ("Protein", "meal_protein"),
                     ("Fat", "meal_fat"), ("Fiber", "meal_fiber")]:
        out[dst] = _numeric(df[src]) if src in df.columns else 0.0
    out["patient_id"] = pid
    out = out.dropna(subset=["timestamp", "glucose"])
    out = out.sort_values("timestamp").drop_duplicates("timestamp")
    return (out.set_index("timestamp")
            .resample("5min")
            .agg({"glucose": "mean", "heart_rate": "mean", "mets": "mean",
                  "meal_carbs": "sum", "meal_protein": "sum", "meal_fat": "sum",
                  "meal_fiber": "sum", "patient_id": "first"})
            .reset_index())


def _read_ehr(bio_path: Path) -> pd.DataFrame:
    bio = pd.read_csv(bio_path)
    # The published CGMacros bio table uses `subject` as the participant ID.
    pid_col = _first_col(bio, ["subject", "Subject", "Participant", "Participant ID", "ID"])
    if pid_col is None:
        raise ValueError("CGMacros bio.csv must contain its subject identifier column.")
    out = pd.DataFrame({"patient_id": _numeric(bio[pid_col])})
    for src, dst in {"Age": "age", "Gender": "sex", "BMI": "bmi",
                     "A1c PDL (Lab)": "hba1c",
                     "Fasting GLU - PDL (Lab)": "fasting_glucose"}.items():
        out[dst] = bio[src] if src in bio.columns else np.nan
    out["baseline_hr"] = np.nan
    out["patient_id"] = out.patient_id.astype(int)
    for c in ["age", "bmi", "hba1c", "fasting_glucose"]:
        out[c] = _numeric(out[c])
    out["diabetes_status"] = np.select(
        [out.hba1c < 5.7, out.hba1c <= 6.4],
        ["healthy", "pre-diabetes"], default="T2D")
    return out


def prepare(raw_root: Path, output_root: Path):
    ehr = _read_ehr(_find_bio(raw_root))
    subject_files = _find_subject_files(raw_root)
    if not subject_files:
        raise FileNotFoundError("No participant CGMacros CSV files were found.")

    frames = []
    for f in subject_files:
        digits = "".join(ch for ch in f.stem if ch.isdigit())
        if not digits:
            continue
        pid = int(digits[-3:])
        try:
            frames.append(_read_subject(f, pid))
        except ValueError as exc:
            print(exc)
    if not frames:
        raise RuntimeError("No usable participant files were parsed.")

    ts = pd.concat(frames, ignore_index=True)
    baseline = ts.dropna(subset=["heart_rate"]).groupby("patient_id").heart_rate.first()
    ehr["baseline_hr"] = ehr.patient_id.map(baseline)
    valid = sorted(set(ts.patient_id.unique()) & set(ehr.patient_id.unique()))
    ts = ts[ts.patient_id.isin(valid)].copy().sort_values(["patient_id", "timestamp"])
    ehr = ehr[ehr.patient_id.isin(valid)].copy().sort_values("patient_id")

    output_root.mkdir(parents=True, exist_ok=True)
    ts.to_csv(output_root / "timeseries.csv", index=False)
    ehr.to_csv(output_root / "ehr.csv", index=False)
    print(f"Prepared {len(ehr)} participants and {len(ts):,} five-minute observations.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--output-root", default=Path("data/processed"), type=Path)
    args = parser.parse_args()
    prepare(args.raw_root, args.output_root)
