import yfinance as yf
import pandas as pd
import streamlit as st
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator

# --- APP SETUP ---
st.set_page_config(page_title="Indian Swing Trade Scanner", layout="wide")
st.title("📈 Indian Swing Trade Scanner (5-10 Days)")

# --- LOAD STOCK LIST FROM CSV ---
@st.cache_data
def load_stocks():
    df = pd.read_csv("stocks.csv")  # Ensure the file is present in the same directory
    return df['Ticker'].dropna().unique().tolist()

stock_list = load_stocks()

stock_df = load_stocks()
stock_list = stock_df["Symbol"].dropna().unique().tolist()

# --- SIDEBAR FILTERS ---
with st.sidebar:
    st.header("🔍 Filters")
    st.markdown("Adjust filters to find high-probability swing setups.")

    min_volume = st.slider("Min Volume (x Avg)", 1.5, 5.0, 2.0, step=0.1)
    rsi_low = st.slider("Min RSI", 30, 50, 40)
    rsi_high = st.slider("Max RSI", 60, 80, 70)
    min_price = st.slider("Min Price (₹)", 10, 1000, 100)
    max_price = st.slider("Max Price (₹)", 200, 10000, 3000)

    breakout_required = st.checkbox("📈 Require 5-Day High Breakout", value=True)
    trend_required = st.checkbox("🟢 Price Above 20 EMA", value=True)

    st.markdown("---")
    manual_ticker = st.text_input("🔎 Check Specific Stock (e.g., INFY)").upper()
    check_button = st.button("Check Stock")

# --- ANALYSIS FUNCTION ---
def analyze_stock(ticker):
    try:
        data = yf.download(ticker, period="1mo", progress=False)
        if data.empty or len(data) < 20:
            return None

        data = data.dropna()
        close_prices = data['Close'].astype(float)
        volumes = data['Volume'].astype(float)

        ema_20 = EMAIndicator(close=close_prices, window=20).ema_indicator()
        rsi = RSIIndicator(close=close_prices, window=14).rsi()

        latest_close = close_prices.iloc[-1]
        latest_volume = volumes.iloc[-1]
        avg_volume = volumes.mean()

        reason = []
        is_buy = True

        if latest_close < min_price or latest_close > max_price:
            reason.append("Price not within desired swing trading range")
            is_buy = False
        if latest_volume < avg_volume * min_volume:
            reason.append("Volume below minimum average threshold")
            is_buy = False
        if rsi.iloc[-1] < rsi_low:
            reason.append("RSI indicates weak momentum")
            is_buy = False
        elif rsi.iloc[-1] > rsi_high:
            reason.append("RSI indicates overbought condition")
            is_buy = False
        if trend_required and latest_close < ema_20.iloc[-1]:
            reason.append("Price is in a downtrend (below 20 EMA)")
            is_buy = False
        if breakout_required and latest_close < close_prices.rolling(5).max().iloc[-1]:
            reason.append("Not near 5-day high breakout level")
            is_buy = False

        remarks = (
            "✅ Good for Swing Trade – meets all key conditions"
            if is_buy else
            "❌ Not ideal – " + ", ".join(reason)
        )

        return {
            "Stock": ticker.replace(".NS", ""),
            "Price (₹)": f"₹{latest_close:.2f}",
            "Volume (x)": f"{latest_volume/avg_volume:.1f}",
            "RSI": f"{rsi.iloc[-1]:.1f}",
            "Trend": "🟢" if latest_close > ema_20.iloc[-1] else "🔴",
            "Remarks": remarks
        }
    except Exception as e:
        st.error(f"Error scanning {ticker}: {str(e)}")
        return None

# --- MANUAL STOCK CHECK ---
if check_button and manual_ticker:
    ticker_symbol = manual_ticker if manual_ticker.endswith(".NS") else manual_ticker + ".NS"
    stock_result = analyze_stock(ticker_symbol)
    if stock_result:
        st.subheader(f"🧾 Result for {manual_ticker}")
        st.dataframe(pd.DataFrame([stock_result]), hide_index=True)
    else:
        st.warning("Unable to fetch or analyze data for the provided ticker.")

# --- RUN BULK SCAN ---
if st.button("🚀 Scan All Stocks"):
    with st.spinner("Scanning NSE stocks..."):
        results = []
        for symbol in stock_list:
            ticker = symbol if symbol.endswith(".NS") else symbol + ".NS"
            result = analyze_stock(ticker)
            if result and "✅" in result["Remarks"]:
                results.append(result)

    if results:
        st.subheader("📊 Swing Trade Opportunities")
        st.dataframe(pd.DataFrame(results), hide_index=True)
    else:
        st.warning("No stocks matched your criteria. Try adjusting the filters.")

# --- FOOTER ---
st.markdown("---")
st.caption("⚡ Built with Streamlit, Yahoo Finance, and love ❤️ | Data: NSE")
