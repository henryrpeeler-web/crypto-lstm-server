
from fastapi import FastAPI
import joblib
import tensorflow as tf
import numpy as np

app = FastAPI()

# Load model and scaler
model = tf.keras.models.load_model("crypto_lstm_model.keras")
scaler = joblib.load("scaler.pkl")

@app.get("/")
def root():
    return {"message": "Server is live!"}

@app.get("/predict")
def predict(open: float, high: float, low: float, close: float, volume: float):
    # Convert input into numpy array
    X = np.array([[open, high, low, close, volume]])
    X_scaled = scaler.transform(X)
    prob = model.predict(X_scaled)
    signal = "BUY" if prob >= 0.5 else "SELL"
    return {"probability": float(prob[0][0]), "signal": signal}

import requests
import numpy as np

@app.get("/predict-live")
def predict_live():
    # 1. Get 200 candles from Binance (5m)
    url = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=5m&limit=200"
    data = requests.get(url).json()

    closes = [float(k[4]) for k in data]  # close prices only

    # 2. Scale using your saved scaler
    scaled = scaler.transform(np.array(closes).reshape(-1, 1)).reshape(-1)

    # 3. Create the sequence for the model (last 60 closes)
    seq = np.array(scaled[-60:]).reshape(1, 60, 1)

    # 4. Predict
    pred = model.predict(seq)[0][0]

    # 5. Inverse scale prediction
    pred_price = scaler.inverse_transform([[pred]])[0][0]
    last_price = closes[-1]
    direction = "UP" if pred_price > last_price else "DOWN"

    return {
        "current_price": last_price,
        "predicted_price": pred_price,
        "direction": direction
    }
