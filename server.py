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
    model = tf.keras.models.load_model("crypto_lstm_model_stateless.keras")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    print("MODEL LOAD ERROR:", str(e))
    model = None
    scaler = None

@app.get("/")
def root():
    return {"message": "Server is live!"}

# ----------------------
# Fetch candles from Coinbase
# ----------------------
def fetch_candles(n=30, granularity=300):
    url = f"https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity={granularity}"
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Coinbase API error: {e}")

    if not data or len(data) < n:
        raise HTTPException(status_code=500, detail=f"Not enough candle data returned from Coinbase")

    # Coinbase returns [time, low, high, open, close, volume], sort ascending by time
    data_sorted = sorted(data, key=lambda x: x[0])
    return data_sorted[-n:]

# ----------------------
# Predict live
# ----------------------
@app.get("/predict-live")
def predict_live():
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model or scaler not loaded.")

    last_30 = fetch_candles(n=30, granularity=300)
    sequence = [[float(c[3]), float(c[2]), float(c[1]), float(c[4]), float(c[5])] for c in last_30]

    # Scale
    sequence_scaled = scaler.transform(sequence)
    sequence_scaled = sequence_scaled[np.newaxis, :, :]  # Add batch dimension

    # Predict
    prob = model.predict(sequence_scaled)[0][0]
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
        X = np.array([[float(candle["open"]),
                       float(candle["high"]),
                       float(candle["low"]),
                       float(candle["close"]),
                       float(candle["volume"])]])
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid input. Required keys: open, high, low, close, volume"
        )

    X_scaled = scaler.transform(X)
    prob = model.predict(X_scaled)[0][0]
    signal = "BUY" if prob >= 0.5 else "SELL"

    return {"probability": float(prob), "signal": signal}
