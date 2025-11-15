from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import requests
import asyncio

# --- Load model and scaler ---
try:
    model = load_model("crypto_lstm_model.keras")  # your .keras model
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model = None
    scaler = None

app = FastAPI(title="Miyamotoan Analyst")

# --- Globals ---
latest_prediction = None
latest_signal = None
PAIR = "BTC-USD"
TIMEFRAME = 300  # 5m candles

# --- Fetch 5m OHLCV from Coinbase ---
async def fetch_candles():
    global latest_prediction, latest_signal
    while True:
        try:
            url = f"https://api.exchange.coinbase.com/products/{PAIR}/candles?granularity={TIMEFRAME}"
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()  # [time, low, high, open, close, volume]

            # Coinbase returns [time, low, high, open, close, volume], oldest first
            data_sorted = sorted(data, key=lambda x: x[0])
            closes = np.array([c[4] for c in data_sorted]).reshape(-1, 1)  # column vector

            # Scale
            if scaler:
                closes_scaled = scaler.transform(closes)
            else:
                closes_scaled = closes

            # Reshape for LSTM: [samples, timesteps, features]
            X = closes_scaled.reshape(1, len(closes_scaled), 1)

            # Predict
            pred = model.predict(X)
            latest_prediction = pred.flatten().tolist()
            latest_signal = ["BUY" if p > 0.5 else "SELL" for p in pred.flatten()]

        except Exception as e:
            print(f"Error fetching or predicting: {e}")

        # Wait 5 minutes
        await asyncio.sleep(TIMEFRAME)

# --- Endpoints ---
@app.get("/")
def root():
    return {"status": "Miyamotoan Analyst server is live!"}

@app.get("/predict-live")
def predict_live():
    if latest_prediction is None or latest_signal is None:
        raise HTTPException(status_code=500, detail="No prediction available yet.")
    return {"prediction": latest_prediction, "signal": latest_signal}

# --- Startup task ---
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(fetch_candles())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=10000, reload=True)
