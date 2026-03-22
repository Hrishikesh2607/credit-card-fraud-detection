from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import joblib, json, numpy as np
from pathlib import Path
from schemas import (Transaction, PredictionResponse, BatchRequest, BatchResponse, RiskTier)
from preprocess import preprocess

MODELS_DIR = Path("../models")
artifacts  = {}   

def risk_tier(prob: float) -> RiskTier:
    if   prob < 0.30: return RiskTier.low
    elif prob < 0.60: return RiskTier.medium
    else:             return RiskTier.high

def make_prediction(txn_dict: dict) -> PredictionResponse:
    features  = preprocess(txn_dict, artifacts["scaler"])
    prob      = float(artifacts["model"].predict_proba(features)[0][1])
    threshold = artifacts["threshold"]
    return PredictionResponse(
        fraud_probability = round(prob, 4),
        flagged           = prob >= threshold,
        risk_tier         = risk_tier(prob),
        threshold_used    = threshold,
    )

@asynccontextmanager
async def lifespan(app: FastAPI):
    artifacts["model"]     = joblib.load(MODELS_DIR / "best_model.pkl")
    artifacts["scaler"]    = joblib.load(MODELS_DIR / "scaler.pkl")
    cfg = json.loads((MODELS_DIR / "threshold_config.json").read_text())
    artifacts["threshold"] = cfg["threshold"]
    artifacts["config"]    = cfg
    print(f"Model loaded. Threshold={artifacts['threshold']}")
    yield
    artifacts.clear()

app = FastAPI(
    title="Fraud Detection API",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
def health():
    return {
        "status"   : "ok",
        "model"    : "XGBoost",
        "threshold": artifacts["threshold"],
        "version"  : "1.0.0"
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(txn: Transaction):
    try:
        return make_prediction(txn.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    if len(req.transactions) > 1000:
        raise HTTPException(400, "Max 1000 transactions per batch")
    preds = [make_prediction(t.model_dump()) for t in req.transactions]
    return BatchResponse(
        predictions   = preds,
        total         = len(preds),
        flagged_count = sum(p.flagged for p in preds)
    )
    
    