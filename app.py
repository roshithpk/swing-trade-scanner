import yfinance as yf
import pandas as pd
import streamlit as st
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator

# --- APP SETUP ---
st.set_page_config(page_title="Indian Swing Trade Scanner", layout="wide")
st.title("📈 Indian Swing Trade Scanner (5-10 Days)")

# --- SIDEBAR FILTERS ---
st.sidebar.header("Filters")

# Volume Spike
min_volume = st.sidebar.slider("Min Volume (x Avg)", 1.5, 5.0, 2.0)

# RSI Range
rsi_low = st.sidebar.slider("Min RSI", 10, 50, 40)
rsi_high = st.sidebar.slider("Max RSI", 60, 90, 70)

# Price Range
min_price = st.sidebar.slider("Min Price (₹)", 10, 1000, 100)
max_price = st.sidebar.slider("Max Price (₹)", 200, 10000, 3000)

# Toggle filters
breakout_required = st.sidebar.checkbox("📈 Require 5-Day High Breakout", value=True)
trend_required = st.sidebar.checkbox("🟢 Price Above 20 EMA", value=True)

# --- INDIAN STOCK TICKERS (NSE) ---
def get_indian_tickers():
    return [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
        'ICICIBANK.NS', 'KOTAKBANK.NS', 'BHARTIARTL.NS', 'LT.NS', 'SBIN.NS',
        'BAJFINANCE.NS', 'ITC.NS', 'ASIANPAINT.NS', 'DMART.NS', 'MARUTI.NS',
        'TITAN.NS', 'SUNPHARMA.NS', 'NESTLEIND.NS', 'ONGC.NS', 'HDFC.NS'
    ]

# --- SCAN FUNCTION ---
def scan_stock(ticker):
    try:
        data = yf.download(ticker, period="1mo", progress=False)
        if data.empty or len(data) < 20:
            return None

        data = data.dropna()
        close_prices = data['Close'].astype(float).squeeze()
        volumes = data['Volume'].astype(float).squeeze()

        ema_20 = EMAIndicator(close=close_prices, window=20).ema_indicator()
        rsi = RSIIndicator(close=close_prices, window=14).rsi()

        latest_close = close_prices.iloc[-1]
        latest_volume = volumes.iloc[-1]
        avg_volume = volumes.mean()

        volume_ok = latest_volume > avg_volume * min_volume
        rsi_ok = rsi_low < rsi.iloc[-1] < rsi_high
        price_ok = min_price <= latest_close <= max_price
        trend_ok = latest_close > ema_20.iloc[-1]
        breakout_ok = latest_close >= close_prices.iloc[-5:].max()

        if not (volume_ok and rsi_ok and price_ok):
            return None
        if trend_required and not trend_ok:
            return None
        if breakout_required and not breakout_ok:
            return None

        return {
            "Stock": ticker.replace(".NS", ""),
            "Price (₹)": f"₹{latest_close:.2f}",
            "Volume (x)": f"{latest_volume / avg_volume:.1f}",
            "RSI": f"{rsi.iloc[-1]:.1f}",
            "Trend": "🟢" if trend_ok else "🔴",
            "Breakout": "✅" if breakout_ok else "❌",
            "Why Buy?": "📈 Breakout + Volume Spike"
        }

    except Exception as e:
        st.warning(f"Error scanning {ticker}: {str(e)}")
        return None

# --- RUN SCAN ---
if st.button("🔍 Scan Indian Stocks"):
    with st.spinner("Scanning NSE stocks..."):
        results = []
        for ticker in get_indian_tickers():
            result = scan_stock(ticker)
            if result:
                results.append(result)

    if results:
        df = pd.DataFrame(results)
        st.success(f"✅ Found {len(df)} matching stocks.")
        st.dataframe(df, use_container_width=True)
        st.download_button("📥 Download Results", df.to_csv(index=False), file_name="scan_results.csv")
    else:
        st.warning("No stocks match your filters. Try adjusting them.")

# --- FOOTER ---
st.markdown("---")
st.caption("⚡ Powered by Yahoo Finance + Streamlit | Data: NSE (India)")
