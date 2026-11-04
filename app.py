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

# Alias aceptados por activo. Los archivos se buscan por coincidencia de nombre
# en lugar de exigir un nombre exacto: a lo largo del proyecto cambiaron entre
# español e inglés (calibrador/calibrator, escalador/scaler) y entre xausd/xauusd.
ASSET_ALIASES = {
    "EURUSD": ["eurusd"],
    "USDJPY": ["usdjpy"],
    "XAUUSD": ["xauusd", "xausd", "gold", "oro"],
}

MIN_TRADE_MAGNITUDE = {"EURUSD": 0.0015, "USDJPY": 0.0012, "XAUUSD": 0.0020}

ASSET_CONFIG = {a: {"min_trade_magnitude": m} for a, m in MIN_TRADE_MAGNITUDE.items()}

app = FastAPI(title="QuantChart LSTM Inference V2")

models = {}
scalers = {}
calibrators = {}
thresholds = {}
resolved = {}  # archivos realmente cargados por activo (diagnóstico)
dir_files = []  # contenido del directorio de modelos (diagnóstico)


def _pick(files, aliases, ext, required=(), prefer=None):
    """Elige el primer archivo con la extensión, un alias del activo y los
    tokens obligatorios. `prefer` desempata (p.ej. el modelo _0)."""
    cands = [
        f
        for f in files
        if f.lower().endswith(ext)
        and any(a in f.lower() for a in aliases)
        and all(t in f.lower() for t in required)
    ]
    if prefer:
        for f in cands:
            if prefer in f.lower():
                return f
    return cands[0] if cands else None


def _threshold_pair(thr):
    """Acepta umbral_compra/umbral_venta y también compra/venta, buy/sell."""
    if not isinstance(thr, dict):
        return 0.5, 0.5
    buy = thr.get("umbral_compra", thr.get("compra", thr.get("buy", 0.5)))
    sell = thr.get("umbral_venta", thr.get("venta", thr.get("sell", 0.5)))
    return float(buy), float(sell)


def load_all():
    global thresholds, dir_files
    try:
        dir_files = sorted(
            f for f in os.listdir(MODELS_DIR) if os.path.isfile(os.path.join(MODELS_DIR, f))
        )
    except Exception:
        dir_files = []

    try:
        with open(THRESHOLDS_PATH, "r", encoding="utf-8") as f:
            thresholds = json.load(f)
    except Exception:
        thresholds = {}

    for asset, aliases in ASSET_ALIASES.items():
        model_file = _pick(dir_files, aliases, ".keras", required=("v2",), prefer="_model_0")
        if not model_file:
            resolved[asset] = {"error": "modelo .keras no encontrado"}
            continue
        scaler_file = _pick(dir_files, aliases, ".joblib", required=("scal",))
        if not scaler_file:
            resolved[asset] = {"error": "escalador .joblib no encontrado", "model": model_file}
            continue
        cal_file = _pick(dir_files, aliases, ".joblib", required=("calib",))

        try:
            models[asset] = load_model(os.path.join(MODELS_DIR, model_file), compile=False)
            scalers[asset] = joblib.load(os.path.join(MODELS_DIR, scaler_file))
            calibrators[asset] = (
                joblib.load(os.path.join(MODELS_DIR, cal_file)) if cal_file else None
            )
        except Exception as e:
            models.pop(asset, None)
            scalers.pop(asset, None)
            resolved[asset] = {"error": f"{type(e).__name__}: {e}"}
            continue

        resolved[asset] = {"model": model_file, "scaler": scaler_file, "calibrator": cal_file}


@app.on_event("startup")
def _startup():
    load_all()


@app.get("/")
def root():
    return {
        "service": "QuantChart LSTM Inference V2",
        "assets_loaded": list(models.keys()),
        "endpoints": ["/health", "/predict"],
    }


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
            info[asset] = {"loaded": False, **resolved.get(asset, {})}
            continue
        try:
            buy, sell = _threshold_pair(thresholds.get(asset, {}))
            info[asset] = {
                "loaded": True,
                "input_shape": list(models[asset].input_shape),
                "scaler_n_features": getattr(scalers[asset], "n_features_in_", None),
                "has_calibrator": calibrators.get(asset) is not None,
                "thresholds": {"umbral_compra": buy, "umbral_venta": sell},
                **resolved.get(asset, {}),
            }
        except Exception as e:
            info[asset] = {"error": str(e)}
    return {
        "status": "ok",
        "models": list(models.keys()),
        "details": info,
        "dir_files": dir_files,
    }


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
    umbral_compra, umbral_venta = _threshold_pair(thresholds.get(asset, {}))

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
