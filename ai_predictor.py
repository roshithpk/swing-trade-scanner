import streamlit as st
import pandas as pd
import yfinance as yf
from xgboost import XGBRegressor
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
import plotly.graph_objects as go

def run():
    st.title("🔮 AI-Based Price Forecast (Next 2 Weeks)")
    stock = st.text_input("Enter Stock Symbol (e.g., INFY)")

    if stock:
        full_ticker = stock.strip().upper() + ".NS"
        data = yf.download(full_ticker, period="6mo", progress=False)
        if data.empty or len(data) < 60:
            st.error("❌ Not enough data to make a prediction.")
            return

        # Technical indicators
        data["EMA20"] = EMAIndicator(data["Close"], window=20).ema_indicator()
        data["RSI"] = RSIIndicator(data["Close"], window=14).rsi()
        data = data.dropna()

        # Prepare features
        X = data[["Close", "EMA20", "RSI"]]
        y = data["Close"].shift(-1)
        X = X[:-1]
        y = y.dropna()

        model = XGBRegressor()
        model.fit(X, y)

        # Predict next 10 days
        preds = []
        last_row = X.iloc[-1]
        for _ in range(10):
            pred = model.predict([last_row])[0]
            preds.append(pred)

            # Update last row for next prediction
            new_close = pred
            new_ema = last_row["EMA20"] * 0.9 + new_close * 0.1
            new_rsi = last_row["RSI"]  # Can be refined
            last_row = pd.Series([new_close, new_ema, new_rsi], index=["Close", "EMA20", "RSI"])

        # Create date index
        future_dates = pd.date_range(start=data.index[-1], periods=11, freq="B")[1:]
        future_df = pd.DataFrame({"Date": future_dates, "Predicted Close": preds})

        # Plot chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index[-30:], y=data["Close"].tail(30), mode='lines', name='Actual'))
        fig.add_trace(go.Scatter(x=future_df["Date"], y=future_df["Predicted Close"], mode='lines+markers', name='Forecast'))

        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(future_df.set_index("Date").style.format({"Predicted Close": "₹{:.2f}"}))

        st.success("📈 Forecast completed! Use this with caution – it's a model, not a guarantee.")
