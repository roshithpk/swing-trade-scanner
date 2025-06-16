import streamlit as st
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator

# --- PAGE SETUP ---
st.set_page_config(page_title="Indian Swing Trade Scanner", layout="wide")
st.title("📈 Indian Swing Trade Scanner (5-10 Days)")

# --- LOAD STOCK LIST ---
@st.cache_data
def load_stocks():
    df = pd.read_csv("stocks.csv")  # Must have columns: Ticker, Name, Category
    return df, df['Ticker'].dropna().unique().tolist()

stock_df, all_tickers = load_stocks()

# --- SIDEBAR FILTERS ---
st.sidebar.header("📊 Filters")

# Category filter
categories = sorted(stock_df['Category'].dropna().unique())
selected_categories = st.sidebar.multiselect("Select Stock Categories", categories, default=categories)

# Other filters
min_volume = st.sidebar.slider("Min Volume (x Avg)", 1.0, 5.0, 2.0, step=0.1)
rsi_low = st.sidebar.slider("Min RSI", 10, 50, 40)
rsi_high = st.sidebar.slider("Max RSI", 50, 90, 70)
min_price = st.sidebar.slider("Min Price (₹)", 10, 1000, 100)
max_price = st.sidebar.slider("Max Price (₹)", 200, 10000, 3000)
breakout_required = st.sidebar.checkbox("📈 Require 5-Day High Breakout", value=True)
trend_required = st.sidebar.checkbox("🟢 Price Above 20 EMA", value=True)

# --- FILTERED TICKERS ---
filtered_df = stock_df[stock_df['Category'].isin(selected_categories)]
filtered_tickers = filtered_df['Ticker'].tolist()

# --- SCAN FUNCTION ---
def scan_stock(ticker, apply_filters=True):
    try:
        data = yf.download(ticker, period="1mo", progress=False)
        if data.empty or len(data) < 20:
            return None

        data = data.dropna()
        close_prices = pd.Series(data['Close']).astype(float).squeeze()
        volumes = pd.Series(data['Volume']).astype(float).squeeze()

        ema_20 = EMAIndicator(close=close_prices, window=20).ema_indicator()
        rsi = RSIIndicator(close=close_prices, window=14).rsi()

        latest_close = close_prices.iloc[-1]
        latest_volume = volumes.iloc[-1]
        avg_volume = volumes.mean()

        volume_ok = latest_volume > avg_volume * min_volume
        price_ok = min_price <= latest_close <= max_price
        trend_ok = latest_close > ema_20.iloc[-1]
        rsi_ok = rsi_low < rsi.iloc[-1] < rsi_high
        breakout_ok = latest_close == close_prices.rolling(5).max().iloc[-1]

        remarks = []
        if trend_ok: remarks.append("🟢 Above EMA")
        if breakout_ok: remarks.append("📈 Breakout")
        if volume_ok: remarks.append("🔥 Volume Spike")
        if not any([trend_ok, breakout_ok, volume_ok]):
            remarks.append("No buy signal based on current indicators")

        if apply_filters:
            if all([
                volume_ok,
                rsi_ok,
                price_ok,
                (not breakout_required or breakout_ok),
                (not trend_required or trend_ok)
            ]):
                return {
                    "Stock": ticker.replace(".NS", ""),
                    "Price (₹)": f"₹{latest_close:.2f}",
                    "Volume (x)": f"{latest_volume / avg_volume:.1f}",
                    "RSI": f"{rsi.iloc[-1]:.1f}",
                    "Trend": "🟢" if trend_ok else "🔴",
                    "Why Buy?": " + ".join(remarks)
                }
        else:
            return {
                "Stock": ticker.replace(".NS", ""),
                "Price (₹)": f"₹{latest_close:.2f}",
                "Volume (x)": f"{latest_volume / avg_volume:.1f}",
                "RSI": f"{rsi.iloc[-1]:.1f}",
                "Trend": "🟢" if trend_ok else "🔴",
                "Why Buy?": " + ".join(remarks)
            }

    except Exception as e:
        st.error(f"Error scanning {ticker}: {str(e)}")
        return None

# --- SCAN ALL STOCKS ---
if st.button("🔍 Scan Selected Stocks"):
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

# --- ANALYZE A SPECIFIC STOCK ---
st.markdown("---")
st.subheader("🔎 Analyze a Specific Stock (No Filters)")
user_input = st.text_input("Enter stock symbol (e.g., INFY):")

if user_input:
    user_ticker = user_input.strip().upper()
    if not user_ticker.endswith(".NS"):
        user_ticker += ".NS"

    result = scan_stock(user_ticker, apply_filters=False)
    if result:
        st.dataframe(pd.DataFrame([result]), hide_index=True)
    else:
        st.error("❌ Could not retrieve data or stock not suitable for swing trading.")

# --- FOOTER ---
st.markdown("---")
st.caption("⚡ Powered by Yahoo Finance + Streamlit | Data: NSE India")
