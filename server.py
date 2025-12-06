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

@app.get("/")
def root():
    return {"message": "Server is live!"}

# ----------------------
# Fetch last 30 candles from Coinbase
# ----------------------
def fetch_last_30_candles():
    url = "https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=300"
    try:
        r = requests.get(url, timeout=5)
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
@app.get("/predict-live")
def predict_live():
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model or scaler not loaded.")

    seq = fetch_last_30_candles()

    try:
        seq_scaled = scaler.transform(seq)
        seq_scaled = seq_scaled[np.newaxis, :, :]  # shape (1,30,5)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scaling sequence: {e}")

    try:
        prob = model.predict(seq_scaled, verbose=0)[0][0]
        signal = "BUY" if prob >= 0.5 else "SELL"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")

    last = seq[-1]
    return {
        "candle": {
            "open": float(last[0]),
            "high": float(last[1]),
            "low": float(last[2]),
            "close": float(last[3]),
            "volume": float(last[4]),
        },
        "probability": float(prob),
        "signal": signal
    }
