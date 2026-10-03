"""
MetaTwin synthetic data generator.
Produces two streams:
  - Static EHR (one row per patient)
  - Dynamic time-series (5-min intervals, 10 days per patient)
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

SEED = 42
N_PATIENTS = 50
DAYS = 10
FREQ_MIN = 5
STEPS_PER_DAY = (24 * 60) // FREQ_MIN
STEPS_PER_PATIENT = DAYS * STEPS_PER_DAY


def generate_static_ehr(n=N_PATIENTS, seed=SEED):
    rng = np.random.default_rng(seed)
    ages = rng.integers(25, 76, n)
    sexes = rng.choice(['M', 'F'], n)
    bmis = np.round(rng.normal(26, 4, n).clip(18, 40), 1)
    status = rng.choice(['healthy', 'pre-diabetes', 'T2D'], n, p=[0.35, 0.35, 0.30])

    hba1c = np.where(status == 'healthy', rng.normal(5.3, 0.3, n),
             np.where(status == 'pre-diabetes', rng.normal(6.0, 0.3, n),
                      rng.normal(7.5, 0.9, n))).clip(4.5, 11.0).round(1)

    fasting = (hba1c * 18 + rng.normal(0, 10, n)).clip(70, 220).round(0)
    baseline_hr = rng.normal(72, 8, n).clip(55, 95).round(0).astype(int)

    return pd.DataFrame({
        'patient_id': np.arange(1, n + 1),
        'age': ages,
        'sex': sexes,
        'bmi': bmis,
        'hba1c': hba1c,
        'fasting_glucose': fasting,
        'diabetes_status': status,
        'baseline_hr': baseline_hr,
    })


def generate_patient_timeseries(patient_row, seed=SEED):
    pid = int(patient_row['patient_id'])
    rng = np.random.default_rng(seed + pid)
    hba1c = patient_row['hba1c']
    baseline_hr = patient_row['baseline_hr']

    baseline_glucose = 85 + (hba1c - 5.0) * 18
    meal_sensitivity = 0.5 + (hba1c - 5.0) * 0.15
    decay_rate = 0.985 + (hba1c - 5.0) * 0.001
    dawn_amplitude = 5 + (hba1c - 5.0) * 1.5

    n = STEPS_PER_PATIENT
    start = datetime(2026, 1, 1) + timedelta(days=pid % 10)

    glucose = np.zeros(n)
    hr = np.zeros(n)
    mets = np.zeros(n)
    carbs = np.zeros(n)
    protein = np.zeros(n)
    fat = np.zeros(n)
    fiber = np.zeros(n)

    g = baseline_glucose
    activity = 0.0
    for i in range(n):
        minute_of_day = (i * FREQ_MIN) % 1440
        hour = minute_of_day / 60

        dawn = dawn_amplitude * np.exp(-((hour - 6) ** 2) / 8)

        meal_carbs = 0.0
        for meal_hour, mean_carb in [(8, 45), (13, 60), (20, 70)]:
            if abs(hour - meal_hour) < (FREQ_MIN / 60):
                meal_carbs = float(np.clip(rng.normal(mean_carb, 15), 20, 120))

        if rng.random() < 0.01:
            activity = float(rng.uniform(1, 8))
        activity = max(0.0, activity * np.exp(-FREQ_MIN / 60))

        g += meal_carbs * meal_sensitivity
        g = baseline_glucose + (g - baseline_glucose) * decay_rate
        g += dawn * (FREQ_MIN / 60)
        g += rng.normal(0, 2)
        g -= activity * 1.5
        g = float(np.clip(g, 50, 400))

        glucose[i] = g
        hr[i] = baseline_hr + activity * 4 + rng.normal(0, 3)
        mets[i] = activity
        carbs[i] = meal_carbs
        protein[i] = meal_carbs * 0.3 if meal_carbs > 0 else 0
        fat[i] = meal_carbs * 0.25 if meal_carbs > 0 else 0
        fiber[i] = meal_carbs * 0.1 if meal_carbs > 0 else 0

    timestamps = pd.date_range(start, periods=n, freq=f'{FREQ_MIN}min')
    return pd.DataFrame({
        'timestamp': timestamps,
        'patient_id': pid,
        'glucose': glucose.round(1),
        'heart_rate': hr.round(1),
        'mets': mets.round(2),
        'meal_carbs': carbs.round(1),
        'meal_protein': protein.round(1),
        'meal_fat': fat.round(1),
        'meal_fiber': fiber.round(1),
    })


def generate_all():
    ehr = generate_static_ehr()
    dfs = [generate_patient_timeseries(row) for _, row in ehr.iterrows()]
    ts = pd.concat(dfs, ignore_index=True)
    return ehr, ts


if __name__ == '__main__':
    ehr, ts = generate_all()
    print(f"EHR rows: {len(ehr)}")
    print(f"Timeseries rows: {len(ts):,}")
    print(f"Glucose: min={ts.glucose.min()}, median={ts.glucose.median()}, max={ts.glucose.max()}")
    print(f"% >180: {(ts.glucose > 180).mean()*100:.1f}%")
    print(f"% >250: {(ts.glucose > 250).mean()*100:.1f}%")

    merged = ts.merge(ehr[['patient_id', 'diabetes_status']], on='patient_id')
    for status, grp in merged.groupby('diabetes_status'):
        print(f"\n{status}:")
        print(f"  median glucose: {grp.glucose.median():.1f}")
        print(f"  % >180: {(grp.glucose > 180).mean()*100:.1f}%")
        print(f"  % >250: {(grp.glucose > 250).mean()*100:.1f}%")
