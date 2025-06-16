import yfinance as yf
import pandas as pd
import streamlit as st
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator

# --- APP SETUP ---
st.set_page_config(page_title="Indian Swing Trade Scanner", layout="wide")
st.title("📈 Indian Swing Trade Scanner (5-10 Days)")

# --- LOAD STOCKS FROM CSV ---
@st.cache_data
def load_stocks():
    df = pd.read_csv("stocks.csv")  # Ensure 'stocks.csv' is in the same folder
    return df['Symbol'].dropna().unique().tolist()

stock_list = load_stocks()

# --- SIDEBAR FILTERS ---
st.sidebar.header("Filters")
min_volume = st.sidebar.slider("Min Volume (x Avg)", 1.5, 5.0, 2.0)
rsi_low = st.sidebar.slider("Min RSI", 30, 50, 40)
rsi_high = st.sidebar.slider("Max RSI", 60, 80, 70)
min_price = st.sidebar.slider("Min Price (₹)", 10, 1000, 100)
max_price = st.sidebar.slider("Max Price (₹)", 200, 10000, 3000)
breakout_required = st.sidebar.checkbox("📈 Require 5-Day High Breakout", value=True)
trend_required = st.sidebar.checkbox("🟢 Price Above 20 EMA", value=True)

# --- ANALYZE SINGLE STOCK ---
st.subheader("🔍 Analyze a Stock Manually")
manual_stock = st.text_input("Enter NSE stock symbol (e.g., INFY, TCS, RELIANCE):")

def analyze_stock(ticker):
    try:
        yf_ticker = ticker if ticker.endswith(".NS") else ticker + ".NS"
        data = yf.download(yf_ticker, period="1mo", progress=False)

        if data.empty or len(data) < 20:
            return None

        data.dropna(inplace=True)
        close_prices = pd.Series(data['Close'].astype(float))
        volumes = pd.Series(data['Volume'].astype(float))
        ema_20 = EMAIndicator(close=close_prices, window=20).ema_indicator()
        rsi = RSIIndicator(close=close_prices, window=14).rsi()
        latest_close = close_prices.iloc[-1]
        avg_volume = volumes.mean()
        latest_volume = volumes.iloc[-1]

        reason = []
        is_buy = True

        # Apply logic
        if latest_close < min_price or latest_close > max_price:
            reason.append("❌ Outside price range")
            is_buy = False
        if latest_volume < avg_volume * min_volume:
            reason.append("❌ Low volume")
            is_buy = False
        if not (rsi_low < rsi.iloc[-1] < rsi_high):
            reason.append("❌ RSI out of range")
            is_buy = False
        if trend_required and latest_close < ema_20.iloc[-1]:
            reason.append("❌ Below EMA20")
            is_buy = False
        if breakout_required and latest_close < close_prices.rolling(5).max().iloc[-1]:
            reason.append("❌ Not at 5-day high")
            is_buy = False
        if is_buy:
            reason = ["✅ Meets all conditions"]

        return {
            "Stock": ticker.upper(),
            "Price (₹)": f"₹{latest_close:.2f}",
            "Volume (x)": f"{latest_volume/avg_volume:.1f}",
            "RSI": f"{rsi.iloc[-1]:.1f}",
            "Trend": "🟢" if latest_close > ema_20.iloc[-1] else "🔴",
            "Why Buy?": " | ".join(reason)
        }
    except Exception as e:
        st.error(f"Error analyzing {ticker}: {str(e)}")
        return None

if manual_stock:
    stock_result = analyze_stock(manual_stock.strip())
    if stock_result:
        st.dataframe(pd.DataFrame([stock_result]), hide_index=True)
    else:
        st.warning("Could not analyze the stock. Check the symbol or data availability.")

# --- SCAN ALL STOCKS ---
st.subheader("📊 Scan All Stocks")

def scan_stock(ticker):
    try:
        data = yf.download(ticker, period="1mo", progress=False)
        if data.empty or len(data) < 20:
            return None
        data.dropna(inplace=True)
        close_prices = pd.Series(data['Close'].astype(float))
        volumes = pd.Series(data['Volume'].astype(float))
        ema_20 = EMAIndicator(close=close_prices, window=20).ema_indicator()
        rsi = RSIIndicator(close=close_prices, window=14).rsi()
        latest_close = close_prices.iloc[-1]
        avg_volume = volumes.mean()
        latest_volume = volumes.iloc[-1]

        # Filters
        if latest_close < min_price or latest_close > max_price:
            return None
        if latest_volume < avg_volume * min_volume:
            return None
        if not (rsi_low < rsi.iloc[-1] < rsi_high):
            return None
        if trend_required and latest_close < ema_20.iloc[-1]:
            return None
        if breakout_required and latest_close < close_prices.rolling(5).max().iloc[-1]:
            return None

        return {
            "Stock": ticker.replace(".NS", ""),
            "Price (₹)": f"₹{latest_close:.2f}",
            "Volume (x)": f"{latest_volume/avg_volume:.1f}",
            "RSI": f"{rsi.iloc[-1]:.1f}",
            "Trend": "🟢" if latest_close > ema_20.iloc[-1] else "🔴",
            "Why Buy?": "📈 Breakout + Volume Spike"
        }
    except Exception as e:
        return None

if st.button("🔍 Scan NSE Stocks"):
    with st.spinner("Scanning..."):
        scan_results = []
        for stock in stock_list:
            result = scan_stock(stock)
            if result:
                scan_results.append(result)
        if scan_results:
            st.success(f"Found {len(scan_results)} matching stocks.")
            st.dataframe(pd.DataFrame(scan_results), hide_index=True)
        else:
            st.warning("No matching stocks. Try changing filters.")

# --- FOOTER ---
st.markdown("---")
st.caption("⚡ Powered by Yahoo Finance + Streamlit | Data: NSE India")
