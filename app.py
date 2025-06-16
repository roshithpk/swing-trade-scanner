import yfinance as yf
import pandas as pd
import streamlit as st
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator

# --- APP SETUP ---
st.set_page_config(page_title="Swing Trade Scanner", layout="wide")
st.title("📈 Swing Trade Scanner (5-10 Days)")

# --- SIDEBAR FILTERS ---
st.sidebar.header("Filters")
min_volume = st.sidebar.slider("Min Volume (x Avg)", 1.5, 5.0, 2.0)
rsi_low = st.sidebar.slider("Min RSI", 30, 50, 40)
rsi_high = st.sidebar.slider("Max RSI", 60, 80, 70)

# --- FETCH STOCK DATA ---
@st.cache_data
def get_sp500_tickers():
    table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    return table[0]["Symbol"].tolist()

# --- SCAN LOGIC ---
def scan_stock(ticker):
    data = yf.download(ticker, period="1mo", progress=False)
    if len(data) < 20:
        return None
    
    # Calculate indicators
    data["EMA_20"] = EMAIndicator(data["Close"], 20).ema_indicator()
    data["RSI"] = RSIIndicator(data["Close"], 14).rsi()
    avg_volume = data["Volume"].mean()
    latest = data.iloc[-1]
    
    # Check conditions
    volume_ok = latest["Volume"] > avg_volume * min_volume
    trend_ok = latest["Close"] > latest["EMA_20"]
    rsi_ok = rsi_low < latest["RSI"] < rsi_high
    breakout_ok = latest["Close"] == data["Close"].rolling(5).max().iloc[-1]
    
    if volume_ok and trend_ok and rsi_ok and breakout_ok:
        return {
            "Ticker": ticker,
            "Price": f"${latest['Close']:.2f}",
            "Volume (x Avg)": f"{latest['Volume'] / avg_volume:.1f}",
            "RSI": f"{latest['RSI']:.1f}",
            "Why Buy?": "📈 Breakout + Volume Spike"
        }

# --- RUN SCAN ---
if st.button("Scan S&P 500 Stocks"):
    with st.spinner("Scanning..."):
        results = []
        for ticker in get_sp500_tickers()[:50]:  # Scan first 50 for speed
            result = scan_stock(ticker)
            if result:
                results.append(result)
    
    if results:
        st.dataframe(pd.DataFrame(results), hide_index=True)
    else:
        st.warning("No stocks match criteria. Adjust filters.")

# --- FOOTER ---
st.markdown("---")
st.caption("⚡ Powered by Yahoo Finance + Streamlit")
