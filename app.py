import yfinance as yf
import pandas as pd
import streamlit as st
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator

# --- APP SETUP ---
st.set_page_config(page_title="Indian Swing Trade Scanner", layout="wide")
st.title("📈 Indian Swing Trade Scanner (5-10 Days)")

# --- LOAD STOCKS CSV ---
@st.cache_data
def load_stocks():
    # Replace the path below with your local path or GitHub raw URL if hosted online
    # Example for GitHub raw: 'https://raw.githubusercontent.com/roshithpk/swing-trade-scanner/main/stocks.csv'
    csv_path = 'stocks.csv'  # local file path
    df = pd.read_csv(csv_path)
    return df

stocks_df = load_stocks()

# --- SIDEBAR FILTERS ---
st.sidebar.header("Filters")

# Category filter dropdown
categories = ['All'] + sorted(stocks_df['Category'].unique().tolist())
selected_category = st.sidebar.selectbox("Select Stock Category", categories)

# Filter DataFrame based on category
if selected_category != 'All':
    filtered_stocks = stocks_df[stocks_df['Category'] == selected_category]
else:
    filtered_stocks = stocks_df.copy()

min_volume = st.sidebar.slider("Min Volume (x Avg)", 1.5, 5.0, 2.0)
rsi_low = st.sidebar.slider("Min RSI", 30, 50, 40)
rsi_high = st.sidebar.slider("Max RSI", 60, 80, 70)

# --- SCAN FUNCTION ---
def scan_stock(ticker):
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

        volume_ok = (latest_volume > avg_volume * min_volume)
        trend_ok = (latest_close > ema_20.iloc[-1])
        rsi_ok = (rsi_low < rsi.iloc[-1] < rsi_high)
        breakout_ok = (latest_close == close_prices.rolling(5).max().iloc[-1])

        if all([volume_ok, trend_ok, rsi_ok, breakout_ok]):
            return {
                "Stock": ticker.replace(".NS", ""),
                "Price (₹)": f"₹{latest_close:.2f}",
                "Volume (x Avg)": f"{latest_volume/avg_volume:.1f}",
                "RSI": f"{rsi.iloc[-1]:.1f}",
                "Trend": "🟢",
                "Why Buy?": "📈 Breakout + Volume Spike"
            }
    except Exception as e:
        st.error(f"Error scanning {ticker}: {str(e)}")
        return None

# --- RUN SCAN ---
if st.button("Scan Indian Stocks"):
    with st.spinner("Scanning selected stocks..."):
        results = []
        for ticker in filtered_stocks['Ticker']:
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
