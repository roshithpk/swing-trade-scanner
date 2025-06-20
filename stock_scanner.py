# stock_scanner.py

import streamlit as st
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from utils import calculate_rsi, get_trend, calculate_entry_exit

# --- LOAD STOCK LIST ---
@st.cache_data
def load_stock_list():
    df = pd.read_csv("data/stocks.csv")  # Make sure this file exists with Ticker, Name, Category
    return df

# --- ANALYZE A SINGLE STOCK FOR SCAN ---
def analyze_stock(ticker):
    try:
        data = yf.download(ticker, period="1mo", interval="1d", progress=False)
        if data.empty or len(data) < 20:
            return None

        close = data["Close"]
        volume = data["Volume"]

        rsi = calculate_rsi(close)
        ema = EMAIndicator(close=close, window=20).ema_indicator()

        current_price = close.iloc[-1]
        avg_volume = volume.mean()
        current_volume = volume.iloc[-1]
        trend = get_trend(close, ema)

        signal = "HOLD"
        reason = []

        if rsi.iloc[-1] < 35:
            signal = "BUY"
            reason.append("RSI indicates oversold")
        elif rsi.iloc[-1] > 70:
            signal = "SELL"
            reason.append("RSI indicates overbought")
        else:
            reason.append("RSI neutral")

        if trend == "UP":
            reason.append("Price above 20 EMA")
        else:
            reason.append("Price below 20 EMA")

        entry, exit_point, growth_pct, est_days = calculate_entry_exit(current_price, trend)

        return {
            "Stock": ticker.replace(".NS", ""),
            "Current Price": f"₹{current_price:.2f}",
            "RSI": f"{rsi.iloc[-1]:.1f}",
            "Trend": trend,
            "Signal": signal,
            "Reason": ", ".join(reason),
            "Entry Point": f"₹{entry:.2f}",
            "Exit Point": f"₹{exit_point:.2f}",
            "Growth %": f"{growth_pct:.1f}%",
            "Est. Days to Target": est_days
        }

    except Exception as e:
        st.warning(f"Failed to analyze {ticker}: {e}")
        return None

# --- MAIN FUNCTION FOR STREAMLIT ---
def run_stock_scanner():
    st.title("📂 Category-based Stock Scanner")

    df = load_stock_list()
    categories = ["All"] + sorted(df["Category"].dropna().unique())
    selected_category = st.selectbox("Select Category", categories)

    if selected_category == "All":
        tickers = df["Ticker"].dropna().tolist()
    else:
        tickers = df[df["Category"] == selected_category]["Ticker"].dropna().tolist()

    if st.button("🔍 Scan Stocks"):
        results = []
        with st.spinner("Scanning..."):
            for ticker in tickers:
                result = analyze_stock(ticker)
                if result:
                    results.append(result)

        if results:
            st.success(f"✅ Found {len(results)} matching stocks")
            st.dataframe(pd.DataFrame(results))
        else:
            st.info("No matching stocks found. Try changing the category.")

