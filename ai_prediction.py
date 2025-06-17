import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
from datetime import datetime, timedelta

def run_ai_prediction():
    st.subheader("🤖 AI-Based Forecast")
    st.markdown("Use AI (Prophet) to forecast next 1–2 weeks of price movement with candlestick visualization.")

    with st.expander("📈 AI-Based Price Forecast"):
        stock_input = st.text_input("Enter NSE stock symbol (e.g., INFY)", key="ai_stock")

        if stock_input:
            ticker = stock_input.upper().strip() + ".NS"
            try:
                df = yf.download(ticker, period="6mo", interval="1d", progress=False)

                if df.empty:
                    st.error("⚠️ No data found.")
                    return

                df.reset_index(inplace=True)
                df = df[["Date", "Open", "High", "Low", "Close"]]

                # Prepare for Prophet
                prophet_df = df[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})

                model = Prophet()
                model.fit(prophet_df)

                future = model.make_future_dataframe(periods=10)
                forecast = model.predict(future)

                # --- Create Candlestick Chart ---
                fig = go.Figure()

                # Historical candles
                fig.add_trace(go.Candlestick(
                    x=df["Date"],
                    open=df["Open"],
                    high=df["High"],
                    low=df["Low"],
                    close=df["Close"],
                    name="Historical"
                ))

                # Forecast line
                fig.add_trace(go.Scatter(
                    x=forecast["ds"],
                    y=forecast["yhat"],
                    mode="lines",
                    line=dict(color="blue", dash="dash"),
                    name="Predicted Close"
                ))

                fig.update_layout(
                    title=f"{stock_input.upper()} - Price Forecast (Next 10 Days)",
                    xaxis_title="Date",
                    yaxis_title="Price (₹)",
                    xaxis_rangeslider_visible=False,
                    template="plotly_dark",
                    height=600
                )

                # Display
                st.plotly_chart(fig, use_container_width=True)

                # Table for forecast
                forecast_table = forecast[["ds", "yhat"]].tail(10)
                forecast_table.columns = ["Date", "Predicted Close (₹)"]
                forecast_table["Predicted Close (₹)"] = forecast_table["Predicted Close (₹)"].apply(lambda x: f"₹{x:.2f}")

                st.markdown("### 📋 Forecast Table")
                st.dataframe(forecast_table, hide_index=True)

            except Exception as e:
                st.error(f"Error during AI prediction: {str(e)}")
