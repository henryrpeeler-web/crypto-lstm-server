import requests
import numpy as np
import pickle
import tensorflow as tf
import pandas as pd

MODEL_PATH = "crypto_lstm_model_v2.keras"
SCALER_PATH = "scaler.pkl"

# Load model + scaler
model = tf.keras.models.load_model(MODEL_PATH)
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

def get_latest_candle():
    url = "https://api.exchange.coinbase.com/products/BTC-USDT/candles?granularity=300"
    data = requests.get(url).json()
    if isinstance(data, list) and len(data) > 0:
        o, h, l, c, v = data[0][1], data[0][2], data[0][3], data[0][4], data[0][5]
        return {"open": o, "high": h, "low": l, "close": c, "volume": v}
    return None

def preprocess_candle(c):
    arr = np.array([[c["open"], c["high"], c["low"], c["close"], c["volume"]]])
    scaled = scaler.transform(arr)
    sequence = np.zeros((1, 30, 5))
    sequence[0, -1, :] = scaled
    return sequence

def predict_signal():
    candle = get_latest_candle()
    if candle is None:
        return

    X = preprocess_candle(candle)
    p = float(model.predict(X, verbose=0)[0][0])
    signal = "BUY" if p > 0.5 else "SELL"

    payload = {
        "candle": candle,
        "probability": p,
        "signal": signal,
    }

    print(payload)

if __name__ == "__main__":
    predict_signal()
