import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import timedelta


def run_ai_prediction():
    st.markdown("---")
    st.subheader("🧠 AI-Based Stock Price Prediction")
    st.caption("Use AI to forecast closing prices for the next few days")

    stock_input = st.text_input("Enter NSE Stock Symbol (e.g., INFY)", key="ai_stock_input")
    forecast_days = st.slider("Select forecast period (days)", 1, 15, 5, key="forecast_slider")

    if stock_input:
        try:
            full_ticker = stock_input.upper().strip() + ".NS"
            data = yf.download(full_ticker, period="6mo", progress=False)

            if data.empty or len(data) < 30:
                st.warning("⚠️ Not enough data for prediction.")
                return

            df = data.copy()
            df.reset_index(inplace=True)
            df = df[["Date", "Open", "High", "Low", "Close"]].dropna()

            df["Date"] = pd.to_datetime(df["Date"])
            df["Days"] = (df["Date"] - df["Date"].min()).dt.days

            # Prepare input for model
            X = df[["Days"]].values
            y = df[["Close"]].values

            st.write(f"\U0001F4CA X shape: {X.shape}")
            st.write(f"\U0001F4C9 y shape: {y.shape}")

            # Train the model
            model = LinearRegression()
            model.fit(X, y)

            last_day = df["Days"].iloc[-1]
            future_days = np.array([last_day + i for i in range(1, forecast_days + 1)]).reshape(-1, 1)
            future_dates = [df["Date"].iloc[-1] + timedelta(days=i) for i in range(1, forecast_days + 1)]

            future_preds = model.predict(future_days).flatten()

            # Create forecast DataFrame
            forecast_df = pd.DataFrame({
                "Date": future_dates,
                "Predicted Close": future_preds
            })

            # Display table
            st.markdown("#### 📊 AI Forecast Table")
            st.dataframe(forecast_df.style.format({"Predicted Close": "₹{:.2f}"}), hide_index=True)

            # Plot candlestick chart + predicted line
            st.markdown("#### 📈 Price Chart with AI Forecast")
            fig = go.Figure()

            # Historical candlestick
            fig.add_trace(go.Candlestick(
                x=df["Date"],
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="Actual"
            ))

            # Forecast line
            fig.add_trace(go.Scatter(
                x=forecast_df["Date"],
                y=forecast_df["Predicted Close"],
                mode="lines+markers",
                name="AI Forecast",
                line=dict(color="orange", width=2, dash="dash")
            ))

            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Price (₹)",
                template="plotly_white",
                showlegend=True,
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error during AI prediction: {str(e)}")
