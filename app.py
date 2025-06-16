import yfinance as yf
import pandas as pd
import streamlit as st
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator

# --- APP SETUP ---
st.set_page_config(page_title="Indian Swing Trade Scanner", layout="wide")
st.title("📈 Indian Swing Trade Scanner (5–10 Days)")

# --- LOAD STOCKS CSV ---
@st.cache_data
def load_stocks():
    df = pd.read_csv("stocks.csv")  # Ensure this file exists in same directory
    return df

stock_df = load_stocks()
categories = stock_df["Category"].dropna().unique().tolist()

# --- FILTERS ---
st.sidebar.header("Filters")

min_volume = st.sidebar.slider("Min Volume (x Avg)", 1.0, 5.0, 2.0)
rsi_low = st.sidebar.slider("Min RSI", 10, 50, 30)
rsi_high = st.sidebar.slider("Max RSI", 50, 90, 70)
min_price = st.sidebar.slider("Min Price (₹)", 10, 1000, 100)
max_price = st.sidebar.slider("Max Price (₹)", 200, 10000, 3000)
breakout_required = st.sidebar.checkbox("📈 Require 5-Day High Breakout", value=True)
trend_required = st.sidebar.checkbox("🟢 Price Above 20 EMA", value=True)

# --- SCAN FUNCTION ---
def scan_stock(ticker, apply_filters=True):
    try:
        data = yf.download(ticker, period="1mo", progress=False)

        if data.empty or len(data) < 20:
            return None

        close_prices = pd.Series(data['Close'].astype(float)).squeeze()
        volumes = pd.Series(data['Volume'].astype(float)).squeeze()

        ema_20 = EMAIndicator(close=close_prices, window=20).ema_indicator()
        rsi = RSIIndicator(close=close_prices, window=14).rsi()

        latest_close = close_prices.iloc[-1]
        latest_volume = volumes.iloc[-1]
        avg_volume = volumes.mean()

        volume_ok = (latest_volume > avg_volume * min_volume)
        trend_ok = (latest_close > ema_20.iloc[-1])
        rsi_ok = (rsi_low < rsi.iloc[-1] < rsi_high)
        breakout_ok = (latest_close == close_prices.rolling(5).max().iloc[-1])
        price_ok = (min_price <= latest_close <= max_price)

        if apply_filters:
            if not all([volume_ok, price_ok, rsi_ok]):
                return None
            if breakout_required and not breakout_ok:
                return None
            if trend_required and not trend_ok:
                return None

        # Decide buy reason
        reasons = []
        if breakout_ok:
            reasons.append("📈 Breakout")
        if volume_ok:
            reasons.append("🔥 Volume Spike")
        if trend_ok:
            trend_icon = "🟢"
        else:
            trend_icon = "🔴"

        reason_text = ", ".join(reasons) if reasons else "No buy signal based on current filters"

        return {
            "Stock": ticker.replace(".NS", ""),
            "Price (₹)": f"₹{latest_close:.2f}",
            "Volume (x)": f"{latest_volume / avg_volume:.1f}",
            "RSI": f"{rsi.iloc[-1]:.1f}",
            "Trend": trend_icon,
            "Why Buy?": reason_text
        }

    except Exception as e:
        st.error(f"Error scanning {ticker}: {e}")
        return None

# --- SCAN ALL STOCKS ---
if st.button("🔍 Scan Selected Stocks"):
    selected_category = st.selectbox("Select Stock Category", categories)
    filtered_df = stock_df[stock_df['Category'] == selected_category]
    filtered_tickers = filtered_df['Ticker'].tolist()

    with st.spinner("Scanning stocks..."):
        results = []
        for ticker in filtered_tickers:
            result = scan_stock(ticker, apply_filters=True)
            if result:
                results.append(result)

    if results:
        st.success(f"✅ {len(results)} stocks match the criteria")
        st.dataframe(pd.DataFrame(results), hide_index=True)
    else:
        st.warning("❌ No stocks match the selected filters.")

# --- INDIVIDUAL STOCK CHECK ---
st.markdown("---")
st.subheader("🔎 Analyze a Specific Stock")

user_input = st.text_input("Enter Stock Ticker (e.g., INFY, TCS, RELIANCE)").upper().strip()
if user_input:
    if not user_input.endswith(".NS"):
        user_input += ".NS"

    with st.spinner(f"Analyzing {user_input}..."):
        result = scan_stock(user_input, apply_filters=False)

    if result:
        st.dataframe(pd.DataFrame([result]), hide_index=True)
    else:
        st.error("Could not retrieve or process data for this stock.")

# --- FOOTER ---
st.markdown("---")
st.caption("⚡ Powered by Yahoo Finance + Streamlit | Data: NSE | Built for swing traders")
