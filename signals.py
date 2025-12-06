import requests
import numpy as np
import joblib
import tensorflow as tf

MODEL_PATH = "crypto_lstm_model_v2.keras"
SCALER_PATH = "scaler.pkl"

# ----------------------
# Load model + scaler
# ----------------------
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    print("MODEL LOAD ERROR:", str(e))
    model = None
    scaler = None

# ----------------------
# Fetch latest candle from Coinbase
# ----------------------
def get_latest_candle():
    url = "https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=300"
    try:
        data = requests.get(url, timeout=5).json()
    except Exception as e:
        print("Coinbase API error:", e)
        return None

    if not data or len(data) == 0:
        print("No candle data returned from Coinbase")
        return None

    # Coinbase returns [time, low, high, open, close, volume]
    latest = data[0]
    return {
        "open": float(latest[3]),
        "high": float(latest[2]),
        "low": float(latest[1]),
        "close": float(latest[4]),
        "volume": float(latest[5])
    }

# ----------------------
# Preprocess a single candle
# ----------------------
def preprocess_candle(candle):
    arr = np.array([[candle["open"],
                     candle["high"],
                     candle["low"],
                     candle["close"],
                     candle["volume"]]])
    scaled = scaler.transform(arr)
    # Shape to (1, 30, 5) expected by LSTM
    sequence = np.zeros((1, 30, 5))
    sequence[0, -1, :] = scaled
    return sequence

# ----------------------
# Predict signal
# ----------------------
def predict_signal():
    if model is None or scaler is None:
        print("Model or scaler not loaded.")
        return

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
    return payload

if __name__ == "__main__":
    predict_signal()
