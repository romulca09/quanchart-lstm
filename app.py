
import os
import json
import joblib
import numpy as np
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model

MODELS_DIR = os.environ.get("MODELS_DIR", ".")
THRESHOLDS_PATH = os.environ.get("THRESHOLDS_PATH", "config_umbrales_forex.json")
API_KEY = os.environ.get("LSTM_INFERENCE_KEY", "")

ASSET_FILES = {
    "EURUSD": ("lstm_eurusd_classification_model.keras", "scaler_eurusd.joblib"),
    "USDJPY": ("lstm_usdjpy_classification_model.keras", "scaler_usdjpy.joblib"),
    "XAUUSD": ("lstm_gold_classification_model.keras", "scaler_gold.joblib"),
}

app = FastAPI(title="QuantChart LSTM Inference")

models = {}
scalers = {}
thresholds = {}


def load_all():
    global thresholds
    with open(THRESHOLDS_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "umbrales" in cfg and isinstance(cfg["umbrales"], dict):
        thresholds = cfg["umbrales"]
    else:
        thresholds = cfg
    for asset, (m_file, s_file) in ASSET_FILES.items():
        models[asset] = load_model(os.path.join(MODELS_DIR, m_file), compile=False)
        scalers[asset] = joblib.load(os.path.join(MODELS_DIR, s_file))


@app.on_event("startup")
def _startup():
    load_all()


def check_auth(authorization: str | None):
    if API_KEY:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Falta autorización")
        if authorization.split(" ", 1)[1] != API_KEY:
            raise HTTPException(status_code=403, detail="Clave inválida")


class Candle(BaseModel):
    t: int
    o: float
    h: float
    l: float
    c: float
    v: float


class PredictRequest(BaseModel):
    asset: str
    tf: str | None = None
    window: int | None = None
    candles: list[Candle]


def build_features(candles: list[Candle], asset: str) -> np.ndarray:
    rows = [[c.o, c.h, c.l, c.c, c.v] for c in candles]
    X = np.array(rows, dtype="float32")
    timesteps = models[asset].input_shape[1] or X.shape[0]
    if X.shape[0] >= timesteps:
        X = X[-timesteps:]
    else:
        pad = np.repeat(X[:1], timesteps - X.shape[0], axis=0)
        X = np.vstack([pad, X])
    X_scaled = scalers[asset].transform(X)
    return X_scaled.reshape(1, timesteps, X_scaled.shape[1])


@app.get("/health")
def health():
    return {"status": "ok", "models": list(models.keys())}


@app.post("/predict")
def predict(req: PredictRequest, authorization: str | None = Header(default=None)):
    check_auth(authorization)
    asset = req.asset.upper()
    if asset not in models:
        raise HTTPException(status_code=400, detail=f"Activo no soportado: {asset}")
    if not req.candles:
        raise HTTPException(status_code=400, detail="Sin velas")
    X = build_features(req.candles, asset)
    prob = float(models[asset].predict(X, verbose=0)[0][0])
    threshold = float(thresholds.get(asset, 0.5))
    return {"prob_up": prob, "threshold": threshold, "asset": asset, "tf": req.tf, "window": req.window}
