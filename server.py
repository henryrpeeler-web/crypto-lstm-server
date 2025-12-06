from fastapi import FastAPI, HTTPException
import tensorflow as tf
import joblib
import numpy as np
import requests

app = FastAPI()

# ----------------------
# Load model + scaler
# ----------------------
try:
    model = tf.keras.models.load_model("crypto_lstm_model_v2.keras")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    print("MODEL LOAD ERROR:", str(e))
    model = None
    scaler = None

# ----------------------
# Root endpoint
# ----------------------
@app.get("/")
def root():
    return {"message": "Server is live!"}

# ----------------------
# Helper: fetch last 30 Binance candles
# ----------------------
def fetch_binance_candles():
    url = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=5m&limit=30"
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Binance API error: {e}")

    if not data or len(data) < 30:
        raise HTTPException(status_code=500, detail="Not enough candle data returned from Binance")

    # Each candle: [open_time, open, high, low, close, volume, ...]
    sequence = []
    for c in data[-30:]:
        open_p = float(c[1])
        high_p = float(c[2])
        low_p = float(c[3])
        close_p = float(c[4])
        volume = float(c[5])
        sequence.append([open_p, high_p, low_p, close_p, volume])
    return sequence

# ----------------------
# Predict live endpoint
# ----------------------
@app.get("/predict-live")
def predict_live():
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model or scaler not loaded.")

    sequence = fetch_binance_candles()
    sequence_scaled = scaler.transform(sequence)
    sequence_scaled = sequence_scaled[np.newaxis, :, :]  # shape (1,30,5)

    prob = model.predict(sequence_scaled)[0][0]
    signal = "BUY" if prob >= 0.5 else "SELL"

    last_candle = sequence[-1]
    return {
        "candle": {
            "open": last_candle[0],
            "high": last_candle[1],
            "low": last_candle[2],
            "close": last_candle[3],
            "volume": last_candle[4],
        },
        "probability": float(prob),
        "signal": signal
    }

# ----------------------
# Predict from user input
# ----------------------
@app.post("/predict")
def predict_from_input(candle: dict):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model or scaler not loaded.")

    try:
        open_p = float(candle["open"])
        high_p = float(candle["high"])
        low_p = float(candle["low"])
        close_p = float(candle["close"])
        volume = float(candle["volume"])
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid input. Required keys: open, high, low, close, volume"
        )

    X = np.array([[open_p, high_p, low_p, close_p, volume]])
    X_scaled = scaler.transform(X)
    prob = model.predict(X_scaled)[0][0]
    signal = "BUY" if prob >= 0.5 else "SELL"

    return {
        "probability": float(prob),
        "signal": signal
    }
