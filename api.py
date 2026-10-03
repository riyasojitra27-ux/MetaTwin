"""
MetaTwin API — serves model results as JSON for the frontend.
Run: uvicorn api:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
from pathlib import Path
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

BASE = Path(__file__).parent
TABLES = BASE / "results" / "tables"

app = FastAPI(
    title="MetaTwin API",
    description="Self-aware glucose digital twin — results endpoint",
    version="0.1.0",
)

# CORS — allow Lovable preview and local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "name": "MetaTwin API",
        "description": "Self-aware glucose digital twin — results endpoint",
        "endpoints": [
            "/api/summary",
            "/api/abstention",
            "/api/calibration",
            "/api/all",
            "/docs",
        ],
    }


@app.get("/api/summary")
def summary():
    df = pd.read_csv(TABLES / "model_summary.csv")
    return dict(zip(df["metric"], df["value"]))


@app.get("/api/abstention")
def abstention():
    df = pd.read_csv(TABLES / "abstention_curve.csv")
    return df.to_dict(orient="records")


@app.get("/api/calibration")
def calibration():
    df = pd.read_csv(TABLES / "calibration.csv")
    return df.to_dict(orient="records")


@app.get("/api/all")
def all_results():
    """Single fetch for the dashboard — everything at once."""
    s = pd.read_csv(TABLES / "model_summary.csv")
    a = pd.read_csv(TABLES / "abstention_curve.csv")
    c = pd.read_csv(TABLES / "calibration.csv")
    return {
        "summary": dict(zip(s["metric"], s["value"])),
        "abstention": a.to_dict(orient="records"),
        "calibration": c.to_dict(orient="records"),
    }