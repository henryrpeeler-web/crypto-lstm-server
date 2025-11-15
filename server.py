# server.py
from tensorflow.keras.models import load_model
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn

# --- Load trained models and scalers ---
try:
    model = load_model("crypto_lstm_model.keras")  # your trained LSTM
    scaler = joblib.load("scaler.pkl")     # StandardScaler or similar
except Exception as e:
    print(f"Error loading model/scaler: {e}")
    model = None
    scaler = None

app = FastAPI(title="Miyamotoan Analyst")

# --- Request schema ---
class CandleData(BaseModel):
    data: list  # list of [open, high, low, close, volume]

@app.get("/")
def root():
    return {"status": "Miyamotoan Analyst server is live!"}

@app.post("/predict-live")
async def predict_live(candle_request: CandleData):
    if not model or not scaler:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    data = candle_request.data

    # Basic validation
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Empty data provided.")
    
    try:
        # Extract closes (or whatever features your model needs)
        closes = [float(c[3]) for c in data]  # 0=open, 1=high, 2=low, 3=close, 4=volume
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Data format error: {e}")

    # Convert to numpy array and reshape for LSTM [samples, timesteps, features]
    X = np.array(closes).reshape(-1, 1)
    
    # Apply scaling if you used a scaler
    if scaler:
        try:
            X = scaler.transform(X)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Scaler error: {e}")

    X = X.reshape(-1, 1, 1)
    # Make prediction
    try:
        pred = model.predict(X)
        # Example: assume sigmoid output -> buy if >0.5
        signal = ["BUY" if p > 0.5 else "SELL" for p in pred.flatten()]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    return {"prediction": pred.flatten().tolist(), "signal": signal}


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=10000, reload=True)
