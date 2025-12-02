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

    # Step 1 — Fetch
    try:
        candle = fetch_live_candle()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Live candle fetch failed: {e}"
        )

    # Step 2 — Scale
    try:
        X = np.array([candle])
        X_scaled = scaler.transform(X)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Scaling failed: {e}"
        )

    # Step 3 — Predict
    try:
        prob = model.predict(X_scaled)[0][0]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model prediction failed: {e}"
        )

    signal = "BUY" if prob >= 0.5 else "SELL"

    return {
        "candle": candle,
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

    # 1. Create input array
    X = np.array([[open_p, high_p, low_p, close_p, volume]])

    # 2. Scale it
    X_scaled = scaler.transform(X)

    # 3. Predict
    prob = model.predict(X_scaled)[0][0]
    signal = "BUY" if prob >= 0.5 else "SELL"

    return {
        "probability": float(prob),
        "signal": signal
    }
