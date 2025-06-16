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
    df = pd.read_csv('stocks.csv')  # Adjust path if needed
    return df

stocks_df = load_stocks()

# --- SIDEBAR FILTERS ---
st.sidebar.header("Filters")

# Category filter
categories = ['All'] + sorted(stocks_df['Category'].unique())
selected_category = st.sidebar.selectbox("Select Category", categories)

# Price range filter
min_price = st.sidebar.slider("Min Price (₹)", 10, 1000, 100)
max_price = st.sidebar.slider("Max Price (₹)", 200, 10000, 3000)

# Volume and RSI filters
min_volume = st.sidebar.slider("Min Volume (x Avg)", 1.5, 5.0, 2.0)
rsi_low = st.sidebar.slider("Min RSI", 30, 50, 40)
rsi_high = st.sidebar.slider("Max RSI", 60, 80, 70)

# Toggle filters
breakout_required = st.sidebar.checkbox("📈 Require 5-Day High Breakout", value=True)
trend_required = st.sidebar.checkbox("🟢 Price Above 20 EMA", value=True)

# --- FILTER STOCKS BY CATEGORY ---
if selected_category != 'All':
    filtered_stocks = stocks_df[stocks_df['Category'] == selected_category]
else:
    filtered_stocks = stocks_df

tickers_list = filtered_stocks['Ticker'].tolist()

# --- SCAN FUNCTION ---
def scan_stock(ticker):
    try:
        data = yf.download(ticker, period="1mo", progress=False)
        if data.empty or len(data) < 20:
            return None
        
        data = data.dropna()
        
        close_prices = data['Close'].astype(float).squeeze()
        if len(close_prices.shape) > 1:
            close_prices = close_prices.squeeze()
        
       volumes = data['Volume'].astype(float).squeeze()
        if len(volumes.shape) > 1:
            volumes = volumes.squeeze()
        
        ema_20 = EMAIndicator(close=close_prices, window=20).ema_indicator()
        rsi = RSIIndicator(close=close_prices, window=14).rsi()
        
        latest_close = close_prices.iloc[-1]
        latest_volume = volumes.iloc[-1]
        avg_volume = volumes.mean()
        
        # Check Price range filter
        if not (min_price <= latest_close <= max_price):
            return None
        
        volume_ok = (latest_volume > avg_volume * min_volume)
        trend_ok = (latest_close > ema_20.iloc[-1])
        rsi_ok = (rsi_low < rsi.iloc[-1] < rsi_high)
        breakout_ok = (latest_close == close_prices.rolling(5).max().iloc[-1])
        
        # Apply toggles for breakout and trend filters
        conditions = [
            volume_ok,
            rsi_ok
        ]
        if breakout_required:
            conditions.append(breakout_ok)
        if trend_required:
            conditions.append(trend_ok)
        
        if all(conditions):
            return {
                "Stock": ticker.replace(".NS", ""),
                "Price (₹)": f"₹{latest_close:.2f}",
                "Volume (x)": f"{latest_volume / avg_volume:.1f}",
                "RSI": f"{rsi.iloc[-1]:.1f}",
                "Trend": "🟢" if trend_ok else "🔴",
                "Why Buy?": "📈 Breakout + Volume Spike" if breakout_ok else "Volume Spike"
            }
        else:
            # Return basic info with "No buy signal"
            return {
                "Stock": ticker.replace(".NS", ""),
                "Price (₹)": f"₹{latest_close:.2f}",
                "Volume (x)": f"{latest_volume / avg_volume:.1f}",
                "RSI": f"{rsi.iloc[-1]:.1f}",
                "Trend": "🟢" if trend_ok else "🔴",
                "Why Buy?": "No buy signal based on current filters"
            }
    except Exception as e:
        st.error(f"Error scanning {ticker}: {str(e)}")
        return None

# --- USER INPUT FOR SINGLE STOCK ---
st.sidebar.header("Check a Single Stock")
user_stock_input = st.sidebar.text_input("Enter stock ticker (e.g. INFY)", "").strip().upper()

if user_stock_input:
    # Add .NS if missing (assuming NSE stocks)
    if not user_stock_input.endswith(".NS"):
        user_stock_input += ".NS"
    
    st.write(f"### Scan result for {user_stock_input.replace('.NS','')}:")
    result = scan_stock(user_stock_input)
    if result:
        st.write(result)
    else:
        st.warning("No data or no buy signal found for this stock.")

# --- RUN SCAN ON FILTERED LIST ---
if st.button("Scan Indian Stocks"):
    with st.spinner("Scanning selected stocks..."):
        results = []
        for ticker in tickers_list:
            result = scan_stock(ticker)
            if result and "No buy signal" not in result["Why Buy?"]:
                results.append(result)
    
    if results:
        st.dataframe(pd.DataFrame(results), hide_index=True)
    else:
        st.warning("No stocks match criteria. Try adjusting filters or category.")

# --- FOOTER ---
st.markdown("---")
st.caption("⚡ Powered by Yahoo Finance + Streamlit | Data: NSE (India)")
