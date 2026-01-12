from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Reuse prediction utilities
from scripts.predict_match import predict_custom, predict_from_dataset

repo_root = Path(__file__).resolve().parents[1]
MODEL_PATH = repo_root / "models" / "rf_model.joblib"
DATA_PATH = repo_root / "data" / "atp_preprocessed.csv"
ELO_PATH = repo_root / "data" / "elo_ratings.json"
TOUR_ELO_PATH = repo_root / "data" / "tournament_elo_ratings.json"
TOUR_STATS_PATH = repo_root / "data" / "tournament_stats.json"

app = FastAPI(title="Tennis Predictor API", version="1.0.0")

# CORS: permitir acceso desde el frontend en desarrollo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
model = None
preprocessed_df = None
elo_ratings: Optional[Dict[str, Dict[str, float]]] = None


class CustomPredictRequest(BaseModel):
    Player_1: str
    Player_2: str
    Date: str  # ISO date string
    Surface: Optional[str] = None
    Tournament: Optional[str] = None
    Rank_1: Optional[float] = None
    Rank_2: Optional[float] = None
    Pts_1: Optional[float] = None
    Pts_2: Optional[float] = None
    Round: Optional[str] = None
    Best_of: Optional[int] = 3
    Series: Optional[str] = None
    Court: Optional[str] = None


class DatasetPredictRequest(BaseModel):
    Player_1: str
    Player_2: str
    Date: str  # must match the dataset date


class PredictResponse(BaseModel):
    predicted_winner_flag: int
    proba_player1_win: Optional[float]
    details: Dict[str, Any]


@app.on_event("startup")
def load_resources():
    global model, preprocessed_df, elo_ratings

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    if not DATA_PATH.exists():
        raise RuntimeError(f"Preprocessed data not found at {DATA_PATH}")
    # Load CSV and parse date column
    preprocessed_df = pd.read_csv(DATA_PATH)
    # Ensure Date column exists and is datetime
    if "Date" in preprocessed_df.columns:
        preprocessed_df["Date"] = pd.to_datetime(preprocessed_df["Date"], errors="coerce")
    else:
        raise RuntimeError("Preprocessed data must contain a 'Date' column")

    # Optional ELO ratings
    if ELO_PATH.exists():
        import json
        with open(ELO_PATH, "r", encoding="utf-8") as f:
            elo_ratings = json.load(f)
    # Optional tournament ELO and stats
    if TOUR_ELO_PATH.exists():
        import json as _json
        with open(TOUR_ELO_PATH, "r", encoding="utf-8") as f:
            tour_elo_ratings = _json.load(f)
    else:
        tour_elo_ratings = None

    if TOUR_STATS_PATH.exists():
        import json as _json2
        with open(TOUR_STATS_PATH, "r", encoding="utf-8") as f:
            tour_stats = _json2.load(f)
    else:
        tour_stats = None
    # expose to module-level
    globals()['tour_elo_ratings'] = tour_elo_ratings
    globals()['tour_stats'] = tour_stats


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict/custom", response_model=PredictResponse)
def predict_custom_endpoint(req: CustomPredictRequest):
    if model is None or preprocessed_df is None:
        raise HTTPException(status_code=500, detail="Resources not loaded")

    match_info = {
        "Player_1": req.Player_1,
        "Player_2": req.Player_2,
        "Date": pd.to_datetime(req.Date, errors="coerce"),
        "Surface": req.Surface,
        "Tournament": req.Tournament,
        "Rank_1": req.Rank_1,
        "Rank_2": req.Rank_2,
        "Pts_1": req.Pts_1,
        "Pts_2": req.Pts_2,
        "Round": req.Round,
        "Best of": req.Best_of,
        "Series": req.Series,
        "Court": req.Court,
    }
    if pd.isna(match_info["Date"]):
        raise HTTPException(status_code=400, detail="Invalid Date format")

    try:
        pred, proba, X = predict_custom(preprocessed_df, model, match_info, elo_ratings, globals().get('tour_elo_ratings'), globals().get('tour_stats'))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return PredictResponse(
        predicted_winner_flag=int(pred),
        proba_player1_win=float(proba) if proba is not None else None,
        details={"features": X.to_dict(orient="records")[0]}
    )


@app.post("/predict/dataset", response_model=PredictResponse)
def predict_dataset_endpoint(req: DatasetPredictRequest):
    if model is None or preprocessed_df is None:
        raise HTTPException(status_code=500, detail="Resources not loaded")

    filt = {
        "Player_1": req.Player_1,
        "Player_2": req.Player_2,
        "Date": pd.to_datetime(req.Date, errors="coerce"),
    }
    if pd.isna(filt["Date"]):
        raise HTTPException(status_code=400, detail="Invalid Date format")

    try:
        pred, proba, row = predict_from_dataset(preprocessed_df, model, filt)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

    details = {
        "matched_row": row[[c for c in ["Date", "Tournament", "Player_1", "Player_2", "Winner"] if c in row.index]].to_dict()
    }

    return PredictResponse(
        predicted_winner_flag=int(pred),
        proba_player1_win=float(proba) if proba is not None else None,
        details=details,
    )
