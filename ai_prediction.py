import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def run_ai_prediction():
    st.subheader("🤖 AI-Based Stock Price Forecast")
    with st.expander("ℹ️ About AI Forecast"):
        st.markdown("This uses **Linear Regression** on historical closing prices to predict the stock trend for the next 1–2 weeks.")

    st.write("✅ ai_prediction.py successfully loaded")  # Debug indicator

    stock_input = st.text_input("Enter stock symbol for AI prediction (e.g., INFY)", key="ai_stock_input")

    if st.button("Predict Price", key="ai_predict_button"):
        if not stock_input:
            st.warning("Please enter a stock symbol.")
            return

        try:
            ticker = stock_input.strip().upper() + ".NS"
            data = yf.download(ticker, period="6mo", progress=False)

            if data.empty or len(data) < 30:
                st.warning("Not enough historical data to make a prediction.")
                return

            data = data.dropna()
            data["Date"] = data.index
            data.reset_index(drop=True, inplace=True)
            data["DayIndex"] = np.arange(len(data))

            # Use closing prices for prediction
            X = data["DayIndex"].values.reshape(-1, 1)
            y = data["Close"].values

            st.write(f"📊 X shape: {X.shape}")
            st.write(f"📉 y shape: {y.shape}")
            st.write("✅ X preview:", X[:5])
            st.write("✅ y preview:", y[:5])
            st.write(f"📜 y type: {type(y)}")

            model = LinearRegression()
            model.fit(X, y.flatten())

            # Predict for next 14 days
            future_days = 14
            future_index = np.arange(len(data), len(data) + future_days).reshape(-1, 1)
            future_preds = model.predict(future_index)

            future_dates = pd.date_range(start=data["Date"].iloc[-1] + pd.Timedelta(days=1), periods=future_days, freq='B')  # Business days

            # Merge with existing data
            forecast_df = pd.DataFrame({
                "Date": future_dates,
                "Predicted Close": future_preds
            })

            st.markdown("### 🔍 AI Forecast Table")
            forecast_display = forecast_df.copy()
            forecast_display["Predicted Close"] = forecast_display["Predicted Close"].round(2)
            st.dataframe(forecast_display)

            st.markdown("### 📈 Price Chart with AI Forecast")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(data["Date"], data["Close"], label="Actual Close", color='blue')
            ax.plot(forecast_df["Date"], forecast_df["Predicted Close"], label="Forecasted Close", linestyle="--", color='red')
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.set_title(f"{ticker} - Actual vs Predicted Close Price")
            ax.legend()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Error during AI prediction: {str(e)}")
