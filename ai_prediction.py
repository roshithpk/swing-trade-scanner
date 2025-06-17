import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

def run_ai_prediction():
    st.subheader("🧠 AI-Based Forecast")
    st.markdown("Use AI to predict stock trend and visualize future candles for the next 1–2 weeks.")

    with st.expander("📈 Click here to use AI Prediction"):
        stock_input = st.text_input("Enter Stock Symbol (e.g., INFY)")

        if stock_input:
            full_ticker = stock_input.strip().upper() + ".NS"
            try:
                # Download historical data
                df = yf.download(full_ticker, period="6mo", interval="1d", progress=False)
                if df.empty or len(df) < 60:
                    st.warning("Not enough data to make prediction. Need at least 60 days.")
                    return

                df.reset_index(inplace=True)

                # Prepare data for Prophet
                prophet_df = df[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
                prophet_model = Prophet(daily_seasonality=True)
                prophet_model.fit(prophet_df)

                # Future 14 days
                future = prophet_model.make_future_dataframe(periods=14)
                forecast = prophet_model.predict(future)

                # Merge predictions back into original
                forecast_df = forecast[["ds", "yhat"]].rename(columns={"ds": "Date", "yhat": "Predicted"})
                merged = pd.merge(df, forecast_df, on="Date", how="outer")

                # Display table of predictions (last known + forecast)
                pred_range = forecast_df[forecast_df["Date"] > df["Date"].max()].copy()
                pred_range["Predicted"] = pred_range["Predicted"].round(2)
                st.markdown("#### 📋 Forecasted Prices:")
                st.dataframe(pred_range.tail(10), hide_index=True, use_container_width=True)

                # Plot candlestick + forecast
                fig = go.Figure()

                # Candlestick for historical data
                fig.add_trace(go.Candlestick(
                    x=df["Date"],
                    open=df["Open"],
                    high=df["High"],
                    low=df["Low"],
                    close=df["Close"],
                    name="Historical"
                ))

                # Prediction line
                pred_range = merged[merged["Predicted"].notnull() & merged["Date"] > df["Date"].max()]
                fig.add_trace(go.Scatter(
                    x=pred_range["Date"].tolist(),
                    y=pred_range["Predicted"].tolist(),
                    mode='lines+markers',
                    name="Predicted",
                    line=dict(color='blue', dash="dot")
                ))

                fig.update_layout(
                    title=f"{stock_input.upper()} - AI Forecast (Next 2 Weeks)",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    height=600,
                    xaxis_rangeslider_visible=False
                )

                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error during AI prediction: {str(e)}")
