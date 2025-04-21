# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("MVS.csv", parse_dates=['Date'])
data.set_index('Date', inplace=True)

# Load models & scalers
model_m = load_model("lstm_model_mastercard.h5")
model_v = load_model("lstm_model_visa.h5")

scaler_m = joblib.load("scaler_mastercard.pkl")
scaler_v = joblib.load("scaler_visa.pkl")
scaler_detrended_m = joblib.load("scaler_detrended_mastercard.pkl")
scaler_detrended_v = joblib.load("scaler_detrended_visa.pkl")
trend_model_m = joblib.load("trend_model_mastercard.pkl")
trend_model_v = joblib.load("trend_model_visa.pkl")

# UI
st.title("📈 Stock Price Prediction (Mastercard & Visa)")
option = st.selectbox("Select Stock", ("Mastercard", "Visa"))

start_date = st.date_input("Select Start Date", value=datetime(2023, 1, 1))
end_date = st.date_input("Select End Date", value=datetime(2024, 1, 1))

seq_length = 60

# Select relevant data
if option == "Mastercard":
    stock_data = data[['Close_M', 'Volume_M']]
    close_col = 'Close_M'
    volume_col = 'Volume_M'
    scaler = scaler_m
    scaler_detrended = scaler_detrended_m
    trend_model = trend_model_m
    model = model_m
else:
    stock_data = data[['Close_V', 'Volume_V']]
    close_col = 'Close_V'
    volume_col = 'Volume_V'
    scaler = scaler_v
    scaler_detrended = scaler_detrended_v
    trend_model = trend_model_v
    model = model_v

# Compute features
stock_data['Returns'] = stock_data[close_col].pct_change()
stock_data['Volatility'] = stock_data['Returns'].rolling(window=20).std().bfill()
stock_data['MA_Close'] = stock_data[close_col].rolling(window=20).mean().bfill()
stock_data['SMA50'] = stock_data[close_col].rolling(window=50).mean().bfill()
stock_data['SMA200'] = stock_data[close_col].rolling(window=200).mean().bfill()

features = stock_data[[close_col, volume_col, 'Volatility', 'MA_Close', 'SMA50', 'SMA200', 'Returns']].dropna()
features['Volatility'] = features['Volatility'].rolling(window=5).mean().bfill()

# Detrend
X_idx = np.arange(len(features)).reshape(-1, 1)
detrended = features[close_col].values - trend_model.predict(X_idx)

# Scale
scaled_features = scaler.transform(features)
scaled_detrended = scaler_detrended.transform(detrended.reshape(-1, 1))

# Sequence
def create_sequences(features, target, seq_length):
    X, y = [], []
    for i in range(len(features) - seq_length):
        X.append(features[i:i + seq_length])
        y.append(target[i + seq_length])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_features, scaled_detrended, seq_length)
predicted_detrended = model.predict(X)
predicted_detrended = scaler_detrended.inverse_transform(predicted_detrended)

# Add trend back
future_X = np.arange(len(features) - len(predicted_detrended), len(features)).reshape(-1, 1)
trend = trend_model.predict(future_X)
predicted_prices = predicted_detrended.flatten() + trend
actual_prices = features[close_col].iloc[seq_length:]

# Filter by date
filtered_index = actual_prices.index[(actual_prices.index >= pd.to_datetime(start_date)) & 
                                     (actual_prices.index <= pd.to_datetime(end_date))]

actual_filtered = actual_prices.loc[filtered_index]
predicted_filtered = pd.Series(predicted_prices, index=actual_prices.index).loc[filtered_index]

# Plot
st.subheader(f"{option} Stock Price Prediction")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(actual_filtered.index, actual_filtered, label='Actual', color='orange')
ax.plot(predicted_filtered.index, predicted_filtered, label='Predicted', color='green')
ax.set_title(f"{option} Price Prediction")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)
