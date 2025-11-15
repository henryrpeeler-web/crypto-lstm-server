# server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn
import os
import threading
import time

# Binance client
from binance.client import Client

# --- Load trained model and scaler ---
try:
    model = joblib.load("lstm_model.pkl")  # make sure this exists
    scaler = joblib.load("scaler.pkl")     # your StandardScaler
except Exception as e:
    print(f"Error loading model/scaler: {e}")
    model = None
    scaler = None

# --- Binance API setup ---
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
binance_client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)

# --- FastAPI setup ---
app = FastAPI(title="Miyamotoan Analyst")

# --- Request schema ---
class CandleData(BaseModel):
    data: list  # list of [open, high, low, close, volume]

# --- Root endpoint ---
@app.get("/")
def root():
    return {"status": "Miyamotoan Analyst server is live!"}

# --- Local predict endpoint (send your own candles) ---
@app.post("/predict-live")
async def predict_live(candle_request: CandleData):
    if not model or not scaler:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    data = candle_request.data
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Empty data provided.")

    try:
        closes = [float(c[3]) for c in data]  # close prices only
        X = np.array(closes).reshape(-1, 1, 1)  # reshape for LSTM
        if scaler:
            X = scaler.transform(X)
        pred = model.predict(X)
        signal = ["BUY" if p > 0.5 else "SELL" for p in pred.flatten()]
        return {"prediction": pred.flatten().tolist(), "signal": signal}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

# --- Helper: fetch Binance 5m candles ---
def fetch_binance_5m(symbol="BTCUSDT", limit=50):
    klines = binance_client.get_klines(
        symbol=symbol,
        interval=Client.KLINE_INTERVAL_5MINUTE,
        limit=limit
    )
    data = []
    for k in klines:
        data.append([float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])])
    return data

# --- Predict from Binance 5m candles ---
@app.get("/predict-binance")
def predict_binance(symbol: str = "BTCUSDT"):
    if not model or not scaler:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    try:
        data = fetch_binance_5m(symbol)
        closes = [c[3] for c in data]
        X = np.array(closes).reshape(-1, 1, 1)
        if scaler:
            X = scaler.transform(X)
        pred = model.predict(X)
        signal = ["BUY" if p > 0.5 else "SELL" for p in pred.flatten()]
        return {"symbol": symbol, "prediction": pred.flatten().tolist(), "signal": signal}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

# --- Optional: auto-predict every 5 minutes ---
def schedule_predictions():
    while True:
        try:
            data = fetch_binance_5m("BTCUSDT")
            closes = [c[3] for c in data]
            X = np.array(closes).reshape(-1, 1, 1)
            if scaler:
                X = scaler.transform(X)
            pred = model.predict(X)
            print(f"Auto-prediction BTCUSDT: {pred.flatten()}")
        except Exception as e:
            print(f"Auto-prediction error: {e}")
        time.sleep(300)  # wait 5 minutes

threading.Thread(target=schedule_predictions, daemon=True).start()

# --- Run server ---
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=10000, reload=True)
