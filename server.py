# server.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from binance.client import Client
from datetime import datetime

# ---------------- CONFIG ----------------
BINANCE_SYMBOL = "BTCUSDT"
BINANCE_INTERVAL = "5m"  # 5-minute candles
FETCH_INTERVAL_SECONDS = 300  # 5 minutes
MODEL_PATH = "crypto_lstm_model.keras"
SCALER_PATH = "scaler.pkl"

# Binance client (no API keys needed for public data)
client = Client()

# ---------------- LOAD MODEL ----------------
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

try:
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    print(f"Error loading scaler: {e}")
    scaler = None

# ---------------- APP SETUP ----------------
app = FastAPI(title="Miyamotoan Analyst Auto-Fetcher")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store latest prediction
latest_signal = {"timestamp": None, "signal": None, "prediction": None}

# ---------------- BINANCE FETCH & PREDICT ----------------
async def fetch_and_predict():
    global latest_signal
    while True:
        try:
            # Fetch recent candles
            candles = client.get_klines(
                symbol=BINANCE_SYMBOL,
                interval=BINANCE_INTERVAL,
                limit=50  # last 50 candles
            )

            closes = [float(c[4]) for c in candles]  # close prices
            X = np.array(closes).reshape(-1, 1, 1)  # reshape for LSTM

            # scale if scaler exists
            if scaler:
                X = scaler.transform(X)

            # predict
            if model:
                pred = model.predict(X)
                signal = "BUY" if pred[-1][0] > 0.5 else "SELL"
                latest_signal = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "signal": signal,
                    "prediction": float(pred[-1][0])
                }
                print(f"[{latest_signal['timestamp']}] Signal: {signal}")
        except Exception as e:
            print(f"Error during fetch/predict: {e}")

        await asyncio.sleep(FETCH_INTERVAL_SECONDS)

# ---------------- ENDPOINTS ----------------
@app.get("/")
def root():
    return {"status": "Miyamotoan Analyst server is live!"}

@app.get("/latest-signal")
def get_latest_signal():
    if latest_signal["signal"] is None:
        return {"detail": "No signal yet. Wait for first fetch."}
    return latest_signal

# ---------------- START BACKGROUND TASK ----------------
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(fetch_and_predict())

# ---------------- RUN ----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=10000, reload=True)
