import yfinance as yf
import pandas as pd
import streamlit as st
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator

# --- APP SETUP ---
st.set_page_config(page_title="Indian Swing Trade Scanner", layout="wide")
st.title("📈 Indian Swing Trade Scanner (5-10 Days)")

# --- LOAD STOCKS FROM LOCAL CSV ---
@st.cache_data
def load_stock_dataframe():
    df = pd.read_csv("stocks.csv")
    return df

stock_df = load_stock_dataframe()

# --- SIDEBAR FILTERS ---
st.sidebar.header("Filters")

# Category filter
categories = ["All"] + sorted(stock_df['Category'].dropna().unique().tolist())
selected_category = st.sidebar.selectbox("📊 Stock Category", categories)

# Filter by category
if selected_category == "All":
    stock_list = stock_df["Ticker"].dropna().unique().tolist()
else:
    stock_list = stock_df[stock_df["Category"] == selected_category]["Ticker"].dropna().unique().tolist()

# RSI and Volume filters
min_volume = st.sidebar.slider("Min Volume (x Avg)", 1.5, 5.0, 2.0)
rsi_low = st.sidebar.slider("Min RSI", 30, 50, 40)
rsi_high = st.sidebar.slider("Max RSI", 60, 80, 70)
min_price = st.sidebar.slider("Min Price (₹)", 10, 1000, 100)
max_price = st.sidebar.slider("Max Price (₹)", 200, 10000, 3000)

# Toggle filters
breakout_required = st.sidebar.checkbox("📈 Require 5-Day High Breakout", value=True)
trend_required = st.sidebar.checkbox("🟢 Price Above 20 EMA", value=True)

# --- ANALYSIS FUNCTION ---
def scan_stock(ticker, apply_filters=True):
    try:
        data = yf.download(ticker, period="1mo", progress=False)
        if data.empty or len(data) < 20:
            return None
        data = data.dropna()
        
        close_prices = pd.Series(data['Close'].astype(float))
        volumes = pd.Series(data['Volume'].astype(float))

        ema_20 = EMAIndicator(close=close_prices, window=20).ema_indicator()
        rsi = RSIIndicator(close=close_prices, window=14).rsi()

        latest_close = close_prices.iloc[-1]
        latest_volume = volumes.iloc[-1]
        avg_volume = volumes.mean()

        volume_ok = latest_volume > avg_volume * min_volume
        trend_ok = latest_close > ema_20.iloc[-1]
        rsi_ok = rsi_low < rsi.iloc[-1] < rsi_high
        breakout_ok = latest_close == close_prices.rolling(5).max().iloc[-1]
        price_ok = min_price <= latest_close <= max_price

        reasons = []
        if trend_ok:
            reasons.append("🟢 Price > EMA20")
        else:
            reasons.append("🔴 Below EMA20")

        if breakout_ok:
            reasons.append("📈 5-Day Breakout")
        if volume_ok:
            reasons.append("🔥 Volume Spike")
        if not rsi_ok:
            reasons.append("❌ RSI out of range")

        if apply_filters:
            if all([volume_ok, trend_ok if trend_required else True, rsi_ok, breakout_ok if breakout_required else True, price_ok]):
                return {
                    "Stock": ticker.replace(".NS", ""),
                    "Price (₹)": f"₹{latest_close:.2f}",
                    "Volume (x)": f"{latest_volume / avg_volume:.1f}",
                    "RSI": f"{rsi.iloc[-1]:.1f}",
                    "Trend": "🟢" if trend_ok else "🔴",
                    "Why Buy?": " + ".join(reasons)
                }
        else:
            return {
                "Stock": ticker.replace(".NS", ""),
                "Price (₹)": f"₹{latest_close:.2f}",
                "Volume (x)": f"{latest_volume / avg_volume:.1f}",
                "RSI": f"{rsi.iloc[-1]:.1f}",
                "Trend": "🟢" if trend_ok else "🔴",
                "Why Buy?": " + ".join(reasons) if trend_ok or breakout_ok or volume_ok else "No strong swing signal"
            }
    except Exception as e:
        st.error(f"Error scanning {ticker}: {str(e)}")
        return None

# --- RUN SCAN ---
if st.button("🔍 Scan Stocks"):
    with st.spinner("Scanning selected stocks..."):
        results = []
        for ticker in stock_list:
            result = scan_stock(ticker, apply_filters=True)
            if result:
                results.append(result)

    if results:
        st.success(f"✅ {len(results)} stocks matched your criteria.")
        st.dataframe(pd.DataFrame(results), hide_index=True, use_container_width=True)
    else:
        st.warning("No stocks match your filters. Try adjusting the sliders.")

# --- SINGLE STOCK LOOKUP ---
st.markdown("---")
st.subheader("🔎 Analyze a Specific Stock")

stock_input = st.text_input("Enter NSE Stock Symbol (e.g., INFY.NS)", value="")

if stock_input:
    result = scan_stock(stock_input.strip().upper(), apply_filters=False)
    if result:
        st.dataframe(pd.DataFrame([result]), hide_index=True, use_container_width=True)
    else:
        st.error("❌ Unable to analyze this stock or no data available.")

# --- FOOTER ---
st.markdown("---")
st.caption("⚡ Powered by Yahoo Finance + Streamlit | Built for Indian Swing Traders")
