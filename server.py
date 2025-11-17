from fastapi import FastAPI, HTTPException
import joblib
import tensorflow as tf
import numpy as np
import requests

app = FastAPI()

# ----------------------
# Load model + scaler
# ----------------------
try:
    model = tf.keras.models.load_model("crypto_lstm_model.keras")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    print("MODEL LOAD ERROR:", str(e))
    model = None
    scaler = None


@app.get("/")
def root():
    return {"message": "Server is live!"}


# ----------------------
# LIVE candle fetcher (Coinbase)
# ----------------------
def fetch_live_candle():
    url = "https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=60"

    try:
        r = requests.get(url, timeout=3)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Coinbase API error: {e}")

    if not data or len(data) == 0:
        raise HTTPException(status_code=500, detail="No data returned from Coinbase")

    # Coinbase returns: [time, low, high, open, close, volume]
    c = data[0]
    open_p = float(c[3])
    high_p = float(c[2])
    low_p = float(c[1])
    close_p = float(c[4])
    volume = float(c[5])

    return [open_p, high_p, low_p, close_p, volume]


# ----------------------
# Predict-live endpoint
# ----------------------
@app.get("/predict-live")
def predict_live():
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model or scaler not loaded.")

    # 1. Fetch live candle
    candle = fetch_live_candle()

    # 2. Convert + scale
    X = np.array([candle])
    X_scaled = scaler.transform(X)

    # 3. Predict
    prob = model.predict(X_scaled)[0][0]
    signal = "BUY" if prob >= 0.5 else "SELL"

    return {
        "candle": {
            "open": candle[0],
            "high": candle[1],
            "low": candle[2],
            "close": candle[3],
            "volume": candle[4],
        },
        "probability": float(prob),
        "signal": signal
    }
