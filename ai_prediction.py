# ai_prediction.py

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
import plotly.graph_objects as go


def run_ai_prediction():
    st.header("🤖 AI-Based Price Prediction")
    st.subheader("Estimate short-term trend and future prices using basic AI")

    ai_stock = st.text_input("Enter NSE Stock Symbol (e.g., INFY)", key="ai_stock_input")

    if ai_stock and st.button("Run AI Prediction"):
        try:
            ticker = ai_stock.strip().upper() + ".NS"
            data = yf.download(ticker, period="6mo", progress=False)

            if len(data) < 30:
                st.error("❌ Not enough data to predict. Try another stock.")
                return

            data.dropna(inplace=True)
            data["EMA20"] = EMAIndicator(data["Close"], window=20).ema_indicator()
            data["RSI"] = RSIIndicator(data["Close"], window=14).rsi()

            # Prepare training data
            lookback = 20
            future_days = 10
            recent_data = data[["Close"]].iloc[-lookback:].reset_index(drop=True)
            
            X = np.arange(lookback).reshape(-1, 1)
            y = recent_data["Close"].to_numpy().flatten()  # Ensure 1D array
            
            # Optional debug
            st.write("Shape of X:", X.shape)
            st.write("Shape of y:", y.shape)
            
            model = LinearRegression()
            model.fit(X, y)

            # Predict next 10 days
            X_future = np.arange(lookback, lookback + future_days).reshape(-1, 1)
            y_future = model.predict(X_future)

            # Summary Table
            entry_price = y[-1]
            exit_price = y_future[-1]
            trend = "📈 Uptrend" if exit_price > entry_price else "📉 Downtrend"

            st.markdown("### 📋 AI Forecast Summary")
            st.table({
                "Entry Price (Today)": [f"₹{entry_price:.2f}"],
                "Predicted Exit (10d)": [f"₹{exit_price:.2f}"],
                "Trend": [trend]
            })

            # Plot
            all_dates = list(data.index[-lookback:]) + [data.index[-1] + pd.Timedelta(days=i) for i in range(1, future_days + 1)]
            fig = go.Figure()

            fig.add_trace(go.Scatter(x=data.index[-lookback:], y=y, mode="lines+markers", name="Historical"))
            fig.add_trace(go.Scatter(x=all_dates[-future_days:], y=y_future, mode="lines+markers", name="Predicted", line=dict(dash="dot")))

            fig.update_layout(title=f"{ai_stock.upper()} – Price Prediction (10 Days)",
                              xaxis_title="Date", yaxis_title="Price (₹)",
                              template="plotly_dark", height=500)
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error during AI prediction: {e}")

