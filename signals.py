import requests
import numpy as np
import joblib
import tensorflow as tf

MODEL_PATH = "crypto_lstm_model_stateless.keras"
SCALER_PATH = "scaler.pkl"

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    print("MODEL/SCALER LOAD ERROR:", e)
    model = None
    scaler = None

def get_latest_candle():
    url = "https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=300"
    try:
        data = requests.get(url, timeout=5).json()
    except Exception as e:
        print("Coinbase API error:", e)
        return None

    if not data or len(data) == 0:
        print("No candle data returned")
        return None

    c = data[0]
    return {
        "open": float(c[3]),
        "high": float(c[2]),
        "low": float(c[1]),
        "close": float(c[4]),
        "volume": float(c[5])
    }

def preprocess_candle(c):
    arr = np.array([[c["open"], c["high"], c["low"], c["close"], c["volume"]]])
    scaled = scaler.transform(arr)
    seq = np.zeros((1, 30, 5))
    seq[0, -1, :] = scaled
    return seq

def predict_signal():
    if model is None or scaler is None:
        print("Model or scaler not loaded")
        return

    candle = get_latest_candle()
    if candle is None:
        print("No candle for prediction")
        return

    X = preprocess_candle(candle)
    p = float(model.predict(X, verbose=0)[0][0])
    signal = "BUY" if p > 0.5 else "SELL"

    output = {
        "candle": candle,
        "probability": p,
        "signal": signal
    }

    print(output)
    return output

if __name__ == "__main__":
    predict_signal()
