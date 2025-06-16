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
min_volume = st.sidebar.slider("Min Volume (x Avg)", 1.5, 5.0, 2.0)
rsi_low = st.sidebar.slider("Min RSI", 30, 50, 40)
rsi_high = st.sidebar.slider("Max RSI", 60, 80, 70)

# --- INDIAN STOCK TICKERS (NSE) ---
def get_indian_tickers():
    return [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
        'ICICIBANK.NS', 'KOTAKBANK.NS', 'BHARTIARTL.NS', 'LT.NS', 'SBIN.NS',
        'BAJFINANCE.NS', 'ITC.NS', 'ASIANPAINT.NS', 'DMART.NS', 'MARUTI.NS',
        'TITAN.NS', 'SUNPHARMA.NS', 'NESTLEIND.NS', 'ONGC.NS', 'HDFC.NS'
    ]

# --- FIXED SCAN FUNCTION ---
def scan_stock(ticker):
    try:
        # Get data
        data = yf.download(ticker, period="1mo", progress=False)
        
        # Data validation
        if data.empty or len(data) < 20:
            return None
            
        # Clean data
        data = data.dropna()
        
        # Convert to float and ensure 1D array
        close_prices = data['Close'].astype(float).squeeze()
        volumes = data['Volume'].astype(float).squeeze()
        
        # Calculate indicators
        ema_20 = EMAIndicator(close=close_prices, window=20).ema_indicator()
        rsi = RSIIndicator(close=close_prices, window=14).rsi()
        
        # Get latest values
        latest_close = close_prices.iloc[-1]
        latest_volume = volumes.iloc[-1]
        avg_volume = volumes.mean()
        
        # Check conditions (using .iloc[-1] for single value comparison)
        volume_ok = (latest_volume > avg_volume * min_volume)
        trend_ok = (latest_close > ema_20.iloc[-1])
        rsi_ok = (rsi_low < rsi.iloc[-1] < rsi_high)
        breakout_ok = (latest_close == close_prices.rolling(5).max().iloc[-1])
        
        if all([volume_ok, trend_ok, rsi_ok, breakout_ok]):
            return {
                "Stock": ticker.replace(".NS", ""),
                "Price (₹)": f"₹{latest_close:.2f}",
                "Volume (x)": f"{latest_volume/avg_volume:.1f}",
                "RSI": f"{rsi.iloc[-1]:.1f}",
                "Trend": "🟢",
                "Why Buy?": "📈 Breakout + Volume Spike"
            }
    except Exception as e:
        st.error(f"Error scanning {ticker}: {str(e)}")
        return None

# --- RUN SCAN ---
if st.button("Scan Indian Stocks"):
    with st.spinner("Scanning NSE stocks..."):
        results = []
        for ticker in get_indian_tickers():
            result = scan_stock(ticker)
            if result:
                results.append(result)
    
    if results:
        st.dataframe(pd.DataFrame(results), hide_index=True)
    else:
        st.warning("No stocks match criteria. Try adjusting filters.")

# --- FOOTER ---
st.markdown("---")
st.caption("⚡ Powered by Yahoo Finance + Streamlit | Data: NSE (India)")
