"""
Servicio de inferencia LSTM para QuantChart — V2 (14 features robustos).

Carga 1 modelo Keras por activo, escalador y calibrador isotónico
entrenados con el pipeline V2 (14 variables técnicas robustas).

Contrato:
  POST /predict  { asset, tf, window, candles: [{t,o,h,l,c,v}] }
  -> { prob_up, umbral_compra, umbral_venta, signal, predicted_magnitude, ... }

Despliegue: Render Web Service.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model

MODELS_DIR = os.environ.get("MODELS_DIR", ".")
THRESHOLDS_PATH = os.environ.get("THRESHOLDS_PATH", "config_umbrales_forex.json")
API_KEY = os.environ.get("LSTM_INFERENCE_KEY", "")

# Configuración por activo: archivos V2 (1 modelo por activo)
ASSET_CONFIG = {
    "EURUSD": {
        "model": "lstm_eurusd_v2_model_0.keras",
        "scaler": "escalador_eurusd_v2.joblib",
        "calibrator": "calibrador_eurusd_v2.joblib",
        "min_trade_magnitude": 0.0015,
    },
    "USDJPY": {
        "model": "lstm_usdjpy_v2_model_0.keras",
        "scaler": "escalador_usdjpy_v2.joblib",
        "calibrator": "calibrador_usdjpy_v2.joblib",
        "min_trade_magnitude": 0.0012,
    },
    "XAUUSD": {
        "model": "lstm_xausd_v2_model_0.keras",
        "scaler": "scaler_xausd_v2.joblib",
        "calibrator": "calibrador_xausd_v2.joblib",
        "min_trade_magnitude": 0.0020,
    },
}

app = FastAPI(title="QuantChart LSTM Inference V2")

models = {}
scalers = {}
calibrators = {}
thresholds = {}


def load_all():
    global thresholds
    with open(THRESHOLDS_PATH, "r", encoding="utf-8") as f:
        thresholds = json.load(f)

    for asset, ac in ASSET_CONFIG.items():
        model_path = os.path.join(MODELS_DIR, ac["model"])
        if not os.path.exists(model_path):
            continue
        models[asset] = load_model(model_path, compile=False)
        scalers[asset] = joblib.load(os.path.join(MODELS_DIR, ac["scaler"]))
        cal_path = os.path.join(MODELS_DIR, ac["calibrator"])
        calibrators[asset] = joblib.load(cal_path) if os.path.exists(cal_path) else None


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


# =============================================================================
# FEATURE ENGINEERING — 14 features robustos (idéntico al entrenamiento V2)
# =============================================================================

FEATURES_LIST = [
    "returns", "volatility_5", "volatility_10", "hl_spread",
    "hl_spread_ma", "ema_diff", "rsi_14", "close_to_ema5",
    "close_to_ema15", "co_spread", "gap_open_close", "log_volume",
    "high", "low",
]


def build_features(candles):
    """
    Genera exactamente las mismas 14 variables técnicas robustas que el
    pipeline V2 requiere para la inferencia.
    """
    df = pd.DataFrame({
        "open": [c.o for c in candles],
        "high": [c.h for c in candles],
        "low": [c.l for c in candles],
        "close": [c.c for c in candles],
        "volume": [c.v for c in candles],
    })

    # 1. Retornos y Volatilidad Histórica
    df["returns"] = df["close"].pct_change().fillna(0)
    df["volatility_5"] = df["returns"].rolling(5).std().fillna(0)
    df["volatility_10"] = df["returns"].rolling(10).std().fillna(0)

    # 2. Rangos de Precio (Spread Alto-Bajo)
    df["hl_spread"] = (df["high"] - df["low"]).fillna(0)
    df["hl_spread_ma"] = df["hl_spread"].rolling(5).mean().fillna(0)

    # 3. Tendencia con Medias Móviles Exponenciales (EMA)
    df["ema_5"] = df["close"].ewm(span=5, adjust=False).mean()
    df["ema_15"] = df["close"].ewm(span=15, adjust=False).mean()
    df["ema_diff"] = df["ema_5"] - df["ema_15"]

    # 4. Momentum (RSI de 14 períodos — simple rolling mean, no Wilder)
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean().fillna(0)
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean().fillna(0)
    rs = gain / (loss + 1e-9)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    df["rsi_14"] = df["rsi_14"].fillna(50)

    # 5. Desviación de Tendencia Corta/Larga
    df["close_to_ema5"] = df["close"] / (df["ema_5"] + 1e-9) - 1.0
    df["close_to_ema15"] = df["close"] / (df["ema_15"] + 1e-9) - 1.0

    # 6. Fuerza del Rango y Gap de Apertura
    df["co_spread"] = df["close"] - df["open"]
    df["gap_open_close"] = df["open"] - df["close"].shift(1).fillna(df["open"])
    df["log_volume"] = np.log1p(df["volume"])

    result = df[FEATURES_LIST].bfill().fillna(0)
    return result.values  # numpy array (n, 14)


def build_model_input(candles, asset):
    """Construye la matriz de entrada (1, 24, 14) para el modelo."""
    feats = build_features(candles)  # (n, 14)

    timesteps = models[asset].input_shape[1] or 24
    if feats.shape[0] >= timesteps:
        X = feats[-timesteps:]
    else:
        pad = np.repeat(feats[:1], timesteps - feats.shape[0], axis=0)
        X = np.vstack([pad, feats])

    X_scaled = scalers[asset].transform(X)
    return X_scaled.reshape(1, timesteps, X_scaled.shape[1])


@app.get("/health")
def health():
    info = {}
    for asset in ASSET_CONFIG:
        if asset not in models:
            info[asset] = {"loaded": False}
            continue
        try:
            info[asset] = {
                "loaded": True,
                "input_shape": list(models[asset].input_shape),
                "scaler_n_features": getattr(scalers[asset], "n_features_in_", None),
                "has_calibrator": calibrators.get(asset) is not None,
                "thresholds": thresholds.get(asset, {}),
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
        X = build_model_input(req.candles, asset)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error construyendo features: {e}")

    try:
        out = models[asset].predict(X, verbose=0)
        if isinstance(out, list):
            # Modelo con cabeza dual (dir + mag)
            prob_up = float(out[0][0][0])
            predicted_magnitude = float(out[1][0][0]) if len(out) > 1 else 0.0
        else:
            # Modelo con salida única
            prob_up = float(out[0][0])
            predicted_magnitude = 0.0

        cal = calibrators.get(asset)
        if cal is not None:
            prob_up = float(cal.predict([prob_up])[0])
    except Exception as e:
        import traceback
        raise HTTPException(
            status_code=500,
            detail=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        )

    # Umbrales duales: LONG si prob >= umbral_compra, SHORT si prob < umbral_venta,
    # NEUTRAL en el rango intermedio.
    thr = thresholds.get(asset, {})
    umbral_compra = float(thr.get("umbral_compra", 0.5))
    umbral_venta = float(thr.get("umbral_venta", 0.5))

    if prob_up >= umbral_compra:
        signal = "LONG"
    elif prob_up < umbral_venta:
        signal = "SHORT"
    else:
        signal = "NEUTRAL"

    min_mag = ASSET_CONFIG[asset].get("min_trade_magnitude", 0.0)
    actionable = predicted_magnitude >= min_mag if predicted_magnitude > 0 else True

    return {
        "prob_up": prob_up,
        "umbral_compra": umbral_compra,
        "umbral_venta": umbral_venta,
        "threshold": umbral_compra,  # compatibilidad hacia atrás
        "predicted_magnitude": predicted_magnitude,
        "min_magnitude": min_mag,
        "actionable": actionable,
        "signal": signal,
        "asset": asset,
        "tf": req.tf,
        "window": req.window,
    }
