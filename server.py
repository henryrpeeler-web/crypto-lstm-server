from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
from binance.client import Client
import os
import uvicorn

# --- Load model & scaler ---
try:
    model = joblib.load("lstm_model.pkl")   # your trained LSTM
    scaler = joblib.load("scaler.pkl")      # your StandardScaler
except Exception as e:
    print(f"Error loading model/scaler: {e}")
    model = None
    scaler = None

# --- Binance setup ---
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)

app = FastAPI(title="Miyamotoan Analyst")

@app.get("/")
def root():
    return {"status": "Server is live!"}

@app.get("/predict-live")
def predict_live(symbol: str = "BTCUSDT", interval: str = "5m", limit: int = 50):
    if not model or not scaler:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    try:
        # Fetch last 'limit' candles from Binance
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        closes = [float(k[4]) for k in klines]  # index 4 = close price
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Binance API fetch failed: {e}")

    # Reshape for LSTM
    X = np.array(closes).reshape(-1, 1, 1)  # [samples, timesteps, features]

    # Apply scaler
    try:
        if scaler:
            X = scaler.transform(X)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scaler error: {e}")

    # Make prediction
    try:
        pred = model.predict(X)
        signal = ["BUY" if p > 0.5 else "SELL" for p in pred.flatten()]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    return {"prediction": pred.flatten().tolist(), "signal": signal}

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=10000, reload=True)
