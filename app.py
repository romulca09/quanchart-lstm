"""
Servicio de inferencia LSTM para QuantChart.
Carga los tres modelos Keras + escaladores entrenados en Google Colab
(EUR/USD, USD/JPY, XAU/USD) y expone un endpoint /predict compatible con
base44/shared/lstmInference.ts.

Contrato:
  POST /predict  { asset, tf, window, candles: [{t,o,h,l,c,v}] }
  -> { prob_up: float, threshold: float, asset, tf, window }

Despliegue: Hugging Face Spaces (Docker) o Render Web Service.
"""

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

# Archivos por activo. Ajusta los nombres si difieren de los de tu Colab.
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
    # config_umbrales_forex.json puede ser {"EURUSD": 0.485, ...} o
    # {"umbrales": {"EURUSD": 0.485, ...}} — se normaliza a dict plano.
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
    """
    Construye la matriz de características a partir de la ventana de velas.
    IMPORTANTE: esta sección debe coincidir con la ingeniería de features
    que usaste al entrenar. Por defecto usa OHLCV escalado, que es lo más
    común para modelos LSTM de clasificación.

    Si entrenaste con otra lógica (retornos, medias móviles, RSI, etc.),
    reemplaza este bloque con tu propio preprocesamiento.
    """
    rows = [[c.o, c.h, c.l, c.c, c.v] for c in candles]
    X = np.array(rows, dtype="float32")

    # El modelo espera un número fijo de timesteps (ventana de entrenamiento).
    # Se lee del propio modelo para adaptar la ventana recibida.
    timesteps = models[asset].input_shape[1] or X.shape[0]

    # Ajustar la ventana: tomar las últimas `timesteps` velas; si faltan,
    # rellenar al inicio repitiendo la primera.
    if X.shape[0] >= timesteps:
        X = X[-timesteps:]
    else:
        pad = np.repeat(X[:1], timesteps - X.shape[0], axis=0)
        X = np.vstack([pad, X])

    # Escalado con el escalador entrenado para este activo.
    X_scaled = scalers[asset].transform(X)
    # Forma (1, timesteps, n_features) para la predicción.
    return X_scaled.reshape(1, timesteps, X_scaled.shape[1])


@app.get("/health")
def health():
    info = {}
    for asset in models:
        try:
            info[asset] = {
                "input_shape": list(models[asset].input_shape),
                "scaler_n_features": getattr(scalers[asset], "n_features_in_", None),
            }
        except Exception as e:
            info[asset] = {"error": str(e)}
    return {"status": "ok", "models": list(models.keys()), "details": info}


@app.post("/predict")
def predict(req: PredictRequest, authorization: str | None = Header(default=None)):
    check_auth(authorization)
    asset = req.asset.upper()
    if asset not in models:
        raise HTTPException(status_code=400, detail=f"Activo no soportado: {asset}")
    if not req.candles:
        raise HTTPException(status_code=400, detail="Sin velas")

    try:
        X = build_features(req.candles, asset)
        prob = float(models[asset].predict(X, verbose=0)[0][0])
    except Exception as e:
        import traceback
        raise HTTPException(
            status_code=500,
            detail=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        )

    threshold = float(thresholds.get(asset, 0.5))

    return {
        "prob_up": prob,
        "threshold": threshold,
        "asset": asset,
        "tf": req.tf,
        "window": req.window,
    }

    
