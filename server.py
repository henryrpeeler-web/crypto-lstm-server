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
except Exception as e:
    print("MODEL LOAD ERROR:", str(e))
    model = None

try:
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    print("SCALER LOAD ERROR:", str(e))
    scaler = None

# ----------------------
# Root endpoint
# ----------------------
@app.get("/")
def root():
    return {"message": "Server is live!"}

# ----------------------
# Fetch last 30 candles from Coinbase
# ----------------------
def fetch_last_30_candles():
    url = "https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=300"  # 5-min candles
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Coinbase API error: {e}")

    if not data or len(data) < 30:
        raise HTTPException(status_code=500, detail="Not enough candle data returned from Coinbase")

    # Sort by timestamp ascending
    data_sorted = sorted(data, key=lambda x: x[0])
    return data_sorted[-30:]  # last 30 candles

# ----------------------
# Predict live endpoint
# ----------------------
@app.get("/predict-live")
def predict_live():
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model or scaler not loaded.")

    last_30 = fetch_last_30_candles()

    # Build sequence
    sequence = []
    for c in last_30:
        try:
            sequence.append([
                float(c[3]),  # open
                float(c[2]),  # high
                float(c[1]),  # low
                float(c[4]),  # close
                float(c[5])   # volume
            ])
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error parsing candle: {e}")

    if len(sequence) != 30 or len(sequence[0]) != 5:
        raise HTTPException(status_code=500, detail=f"Invalid sequence shape: {np.array(sequence).shape}")

    # Scale and reshape
    try:
        sequence_scaled = scaler.transform(sequence)  # (30,5)
        sequence_scaled = sequence_scaled[np.newaxis, :, :]  # (1,30,5)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scaling sequence: {e}")

    # Predict
    try:
        prob = model.predict(sequence_scaled)[0][0]
        signal = "BUY" if prob >= 0.5 else "SELL"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")

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
        X = np.array([[
            float(candle["open"]),
            float(candle["high"]),
            float(candle["low"]),
            float(candle["close"]),
            float(candle["volume"])
        ]])
        X_scaled = scaler.transform(X)
        X_scaled = X_scaled[np.newaxis, :, :]  # (1,1,5)
        prob = model.predict(X_scaled)[0][0]
        signal = "BUY" if prob >= 0.5 else "SELL"
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input or prediction error: {e}")

    return {"probability": float(prob), "signal": signal}
