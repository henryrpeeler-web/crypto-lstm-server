
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
