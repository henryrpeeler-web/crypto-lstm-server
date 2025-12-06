from fastapi import FastAPI, HTTPException
import tensorflow as tf
import joblib
import requests
import numpy as np

app = FastAPI()

# ----------------------
# Load stateless model + scaler
# ----------------------
try:
    model = tf.keras.models.load_model("crypto_lstm_model_stateless.keras")
except Exception as e:
    print("MODEL LOAD ERROR:", str(e))
    model = None

try:
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    print("SCALER LOAD ERROR:", str(e))
    scaler = None

# After loading model and scaler
if model is not None:
    try:
        dummy = np.zeros((1,30,5), dtype=np.float32)
        _ = model.predict(dummy, verbose=0)
        print("Model pre-warmed for fast first inference")
    except Exception as e:
        print("Pre-warm failed:", e)

@app.get("/")
def root():
    return {"message": "Server is live!"}

# ----------------------
# Fetch last 30 candles from Coinbase
# ----------------------
def fetch_last_30_candles():
    url = "https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=300"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Coinbase API error: {e}")

    if not data or len(data) < 30:
        raise HTTPException(status_code=500, detail="Not enough candle data returned from Coinbase")

    data_sorted = sorted(data, key=lambda x: x[0])
    last_30 = data_sorted[-30:]

    sequence = []
    for c in last_30:
        sequence.append([
            float(c[3]),  # open
            float(c[2]),  # high
            float(c[1]),  # low
            float(c[4]),  # close
            float(c[5])   # volume
        ])
    return np.array(sequence)  # shape (30,5)

# ----------------------
# /predict-live endpoint
# ----------------------
import numpy as np
from fastapi import HTTPException

@app.get("/predict-live")
def predict_live():
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model or scaler not loaded.")

    # Fetch the last 30 1-minute candles from Coinbase
    url = "https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=60"
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Coinbase API error: {e}")

    if not data or len(data) < 30:
        raise HTTPException(status_code=500, detail="Not enough candle data returned from Coinbase")

    # Coinbase returns: [time, low, high, open, close, volume]
    data_sorted = sorted(data, key=lambda x: x[0])  # sort ascending by timestamp
    last_30 = data_sorted[-30:]

    sequence = []
    for c in last_30:
        open_p = float(c[3])
        high_p = float(c[2])
        low_p = float(c[1])
        close_p = float(c[4])
        volume = float(c[5])
        sequence.append([open_p, high_p, low_p, close_p, volume])

    # Scale features
    sequence_scaled = scaler.transform(sequence)  # shape (30,5)

    # Add batch dimension for stateless LSTM
    sequence_scaled = np.expand_dims(sequence_scaled, axis=0)  # shape (1,30,5)

    # Predict
    prob = float(model.predict(sequence_scaled)[0][0])
    signal = "BUY" if prob >= 0.5 else "SELL"

    last_candle = last_30[-1]
    return {
        "candle": {
            "open": float(last_candle[3]),
            "high": float(last_candle[2]),
            "low": float(last_candle[1]),
            "close": float(last_candle[4]),
            "volume": float(last_candle[5]),
        },
        "probability": prob,
        "signal": signal
    }
