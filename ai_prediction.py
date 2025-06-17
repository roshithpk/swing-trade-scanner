import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from datetime import timedelta

def run_ai_prediction():
    st.header("🧠 AI-Based Price Prediction")
    st.subheader("📌 Enter Stock Symbol (NSE) for Prediction")

    stock_symbol = st.text_input("Enter Stock Symbol (e.g., INFY):")
    forecast_days = st.slider("Days to Forecast", 1, 14, 7)

    if stock_symbol:
        full_ticker = stock_symbol.upper().strip() + ".NS"
        try:
            data = yf.download(full_ticker, period="6mo", progress=False)
            if data.empty or len(data) < 30:
                st.warning("⚠️ Not enough data for prediction.")
                return

            # Keep only necessary columns
            df = data[["Close", "Open", "High", "Low"]].dropna().reset_index()

            df["Date"] = pd.to_datetime(df["Date"])
            df = df[["Date", "Open", "High", "Low", "Close"]]

            df["Days"] = (df["Date"] - df["Date"].min()).dt.days

            # MODEL
            X = df[["Days"]]
            y = df[["Close"]]

            st.write(f"📊 X shape: {X.shape}")
            st.write(f"📉 y shape: {y.shape}")

            model = LinearRegression()
            model.fit(X, y)

            # Predict future
            last_day = df["Days"].iloc[-1]
            future_days = np.array([last_day + i for i in range(1, forecast_days + 1)]).reshape(-1, 1)
            future_dates = [df["Date"].max() + timedelta(days=i) for i in range(1, forecast_days + 1)]
            predictions = model.predict(future_days).flatten()

            forecast_df = pd.DataFrame({
                "Date": future_dates,
                "Predicted": predictions
            })

            merged = pd.merge(df, forecast_df, on="Date", how="outer")

            # Debug output
            st.write("📊 Shape of df (historical):", df.shape)
            st.write("📊 Shape of forecast_df:", forecast_df.shape)
            st.write("📊 Sample forecast_df:", forecast_df.tail(3))
            st.write("📊 Sample merged:", merged.tail(3))

            pred_range = merged[merged["Predicted"].notnull() & (merged["Date"] > df["Date"].max())]

            st.write("📊 pred_range shape:", pred_range.shape)
            st.write("📊 pred_range.dtypes:", pred_range.dtypes)
            st.write("📊 pred_range sample:", pred_range.tail(3))

            # PLOT
            fig = go.Figure()

            # Add candlestick chart for historical
            fig.add_trace(go.Candlestick(
                x=df["Date"],
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="Historical"
            ))

            # Add prediction line
            x_vals = pred_range["Date"]
            y_vals = pred_range["Predicted"]

            st.write("✅ X values type:", type(x_vals), "Shape:", x_vals.shape)
            st.write("✅ Y values type:", type(y_vals), "Shape:", y_vals.shape)

            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='lines+markers',
                name="Predicted",
                line=dict(color='blue', dash="dot")
            ))

            fig.update_layout(title="📈 AI Price Prediction (Linear Regression)",
                              xaxis_title="Date",
                              yaxis_title="Price",
                              xaxis_rangeslider_visible=False)

            st.plotly_chart(fig, use_container_width=True)

            # Result Table
            st.subheader("📋 Predicted Trend")
            entry_price = df["Close"].iloc[-1]
            exit_price = predictions[-1]
            st.table(pd.DataFrame([{
                "Entry Price (Today)": round(entry_price, 2),
                f"Exit Price (After {forecast_days} Days)": round(exit_price, 2),
                "Trend": "📈 Up" if exit_price > entry_price else "📉 Down"
            }]))

        except Exception as e:
            st.error(f"Error during AI prediction: {e}")
