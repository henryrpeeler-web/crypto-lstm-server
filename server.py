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
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    print("MODEL LOAD ERROR:", str(e))
    model = None
    scaler = None

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

    # Coinbase returns: [time, low, high, open, close, volume]
    data_sorted = sorted(data, key=lambda x: x[0])
    last_30 = data_sorted[-30:]

    sequence = []
    for c in last_30:
        open_p = float(c[3])
        high_p = float(c[2])
        low_p = float(c[1])
        close_p = float(c[4])
        volume = float(c[5])
        sequence.append([open_p, high_p, low_p, close_p, volume])

    return np.array(sequence)  # shape (30,5)

# ----------------------
# /predict-live endpoint
# ----------------------
@app.get("/predict-live")
def predict_live():
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model or scaler not loaded.")

    seq = fetch_last_30_candles()
    seq_scaled = scaler.transform(seq)         # scale features
    seq_scaled = seq_scaled[np.newaxis, :, :]  # add batch dimension for stateless LSTM

    prob = model.predict(seq_scaled, verbose=0)[0][0]
    signal = "BUY" if prob >= 0.5 else "SELL"

    last_candle = seq[-1]
    return {
        "candle": {
            "open": float(last_candle[0]),
            "high": float(last_candle[1]),
            "low": float(last_candle[2]),
            "close": float(last_candle[3]),
            "volume": float(last_candle[4]),
        },
        "probability": float(prob),
        "signal": signal
    }

