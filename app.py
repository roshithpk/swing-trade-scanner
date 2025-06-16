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

# --- CATEGORY MAP (MANUAL) ---
CATEGORY_MAP = {
    "RELIANCE.NS": "Nifty 50",
    "TCS.NS": "Nifty 50",
    "TATAMOTORS.NS": "Nifty 50",
    "HINDPETRO.NS": "Midcap 100",
    "LTI.NS": "Next 50",
    # Add the rest here...
}

# --- LOAD STOCKS FROM CSV ---
@st.cache_data
def get_stock_list():
    url = "https://raw.githubusercontent.com/your-username/your-repo/main/stocks.csv"  # replace with your real URL
    df = pd.read_csv(url)
    df["Category"] = df["Ticker"].map(CATEGORY_MAP).fillna("Uncategorized")
    return df, df.set_index("Ticker").to_dict(orient="index")

stock_df, stock_meta = get_stock_list()

# --- CATEGORY FILTER ---
all_categories = sorted(stock_df["Category"].unique())
selected_categories = st.sidebar.multiselect("Select Categories", options=all_categories, default=all_categories)

# Filter stock list
filtered_df = stock_df[stock_df["Category"].isin(selected_categories)]
stock_list = filtered_df["Ticker"].tolist()

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

        volume_ok = latest_volume > avg_volume * min_volume
        trend_ok = latest_close > ema_20.iloc[-1]
        rsi_ok = rsi_low < rsi.iloc[-1] < rsi_high
        breakout_ok = latest_close == close_prices.rolling(5).max().iloc[-1]

        if all([volume_ok, trend_ok, rsi_ok, breakout_ok]):
            meta = stock_meta.get(ticker, {})
            return {
                "Stock": ticker.replace(".NS", ""),
                "Name": meta.get("Name", ""),
                "Price (₹)": f"₹{latest_close:.2f}",
                "Volume (x)": f"{latest_volume/avg_volume:.1f}",
                "RSI": f"{rsi.iloc[-1]:.1f}",
                "Trend": "🟢",
                "Category": meta.get("Category", "—"),
                "Why Buy?": "📈 Breakout + Volume Spike"
            }
    except Exception as e:
        st.error(f"Error scanning {ticker}: {str(e)}")
        return None

# --- SCAN BUTTON ---
if st.button("🔍 Scan Indian Stocks"):
    with st.spinner(f"Scanning {len(stock_list)} selected stocks..."):
        results = []
        for ticker in stock_list:
            result = scan_stock(ticker)
            if result:
                results.append(result)

    if results:
        st.success(f"✅ Found {len(results)} stocks matching criteria")
        st.dataframe(pd.DataFrame(results), hide_index=True)
    else:
        st.warning("⚠️ No stocks match the selected filters.")

# --- FOOTER ---
st.markdown("---")
st.caption("⚡
