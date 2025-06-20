import streamlit as st
import yfinance as yf
import pandas as pd
from logic.utils import calculate_rsi, determine_trend, suggest_entry_exit

def run_single_stock_analysis():
    st.subheader("🔍 Single Stock Analysis")

    with st.form("analyze_form"):
        stock_symbol = st.text_input("Enter Stock Symbol (e.g., INFY)", value="INFY")
        submit = st.form_submit_button("Analyze")

    if submit and stock_symbol:
        ticker = stock_symbol.strip().upper() + ".NS"
        try:
            df = yf.download(ticker, period="3mo", interval="1d", progress=False)

            if df.empty or len(df) < 30:
                st.error("Not enough data to perform analysis.")
                return

            df.dropna(inplace=True)
            df["RSI"] = calculate_rsi(df["Close"])
            trend_up = determine_trend(df["Close"])
            entry, exit_ = suggest_entry_exit(df["Close"])
            current_price = df["Close"].iloc[-1]
            current_rsi = df["RSI"].iloc[-1]

            # --- Signal Logic ---
            remarks = []
            reasons = []

            if trend_up:
                signal = "BUY"
                reasons.append("Uptrend")
            else:
                signal = "SELL"
                reasons.append("Downtrend")

            if current_rsi < 30:
                signal = "BUY"
                reasons.append("RSI is Oversold (<30)")
            elif current_rsi > 70:
                signal = "SELL"
                reasons.append("RSI is Overbought (>70)")

            if 30 <= current_rsi <= 70 and trend_up:
                signal = "HOLD"
                reasons.append("Neutral RSI but price in uptrend")

            # --- UI Output ---
            st.metric("📌 Current Price", f"₹{current_price:.2f}")
            st.metric("📈 RSI", f"{current_rsi:.2f}")
            st.metric("📉 Trend", "📈 Uptrend" if trend_up else "📉 Downtrend")

            st.success(f"🚦 Signal: {signal}")
            st.write("### 📋 Reasons:")
            for r in reasons:
                st.write(f"- {r}")

            st.write("### 🎯 Suggested Entry & Exit Points:")
            col1, col2 = st.columns(2)
            col1.metric("💰 Entry Price", f"₹{entry}")
            col2.metric("🏁 Exit Price", f"₹{exit_}")

            estimated_days = 5 if trend_up else "N/A"
            expected_growth = ((exit_ - entry) / entry) * 100 if trend_up else 0

            st.write("### ⏳ Estimate:")
            st.markdown(f"- ⏱️ Expected Days to Target: **{estimated_days}**")
            st.markdown(f"- 📈 Expected Gain: **{expected_growth:.2f}%**")

        except Exception as e:
            st.error(f"Failed to analyze stock: {str(e)}")

