import streamlit as st
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
import ai_prediction

# --- CUSTOM SMA RSI FUNCTION ---
def sma_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- DYNAMIC VOLUME + PRICE MOMENTUM SIGNAL ---
def volume_price_momentum_signal(data, volume_spike_threshold=1.2, price_change_threshold=1.0):
    """
    data: DataFrame with 'Close', 'Volume' columns
    Returns dict with volume_ratio, price_change_pct, signal
    """
    close = data['Close']
    volume = data['Volume']

    prev_close = close.shift(1)

    price_change_pct = ((close.iloc[-1] - prev_close.iloc[-1]) / prev_close.iloc[-1]) * 100 if prev_close.iloc[-1] != 0 else 0

    avg_volume = volume[:-1].mean() if len(volume) > 1 else 0
    volume_today = volume.iloc[-1]
    volume_ratio = volume_today / avg_volume if avg_volume > 0 else 0

    if volume_ratio >= volume_spike_threshold and price_change_pct >= price_change_threshold:
        signal = "BUY"
    else:
        signal = "NO BUY"

    return {
        "volume_ratio": volume_ratio,
        "price_change_pct": price_change_pct,
        "signal": signal
    }


if "page" not in st.session_state:
    st.session_state.page = "main"

# --- APP SETUP ---
st.set_page_config(page_title="📊 Indian Swing Trade Scanner", layout="wide")
st.title("📈 Indian Swing Trade Scanner (5-10 Days)")

# --- LOAD STOCK LIST ---
@st.cache_data
def load_stocks():
    df = pd.read_csv("stocks.csv")  # Ensure this CSV has Ticker, Name, Category
    return df

stock_df = load_stocks()

# --- SIDEBAR FILTERS ---
st.sidebar.header("🔧 Filters")

min_volume = st.sidebar.slider("Min Volume (x Avg)", 1.0, 5.0, 2.0)
rsi_low = st.sidebar.slider("Min RSI", 10, 50, 40)
rsi_high = st.sidebar.slider("Max RSI", 50, 90, 70)
min_price = st.sidebar.slider("Min Price (₹)", 10, 1000, 100)
max_price = st.sidebar.slider("Max Price (₹)", 1000, 10000, 3000)
breakout_required = st.sidebar.checkbox("📈 Require 5-Day High Breakout", value=True)
trend_required = st.sidebar.checkbox("🟢 Price Above 20 EMA", value=True)

# --- MAIN FILTER FOR CATEGORY ---
st.subheader("📂 Select Stock Category to Scan")
categories = ["All"] + sorted(stock_df["Category"].dropna().unique())
selected_category = st.selectbox("Category", categories, index=0)

if selected_category == "All":
    filtered_df = stock_df
else:
    filtered_df = stock_df[stock_df["Category"] == selected_category]

filtered_tickers = filtered_df["Ticker"].dropna().unique().tolist()

# --- SCAN FUNCTION ---
def scan_stock(ticker):
    try:
        data = yf.download(ticker, period="1mo", progress=False)
        if data.empty or len(data) < 20:
            return None

        data = data.dropna()
        close_prices = pd.Series(data["Close"].values.flatten(), dtype=float)
        volumes = pd.Series(data["Volume"].values.flatten(), dtype=float)

        if close_prices.empty or volumes.empty:
            return None

        ema_20 = EMAIndicator(close=close_prices, window=20).ema_indicator()
        rsi = sma_rsi(close_prices, window=14)

        latest_close = close_prices.iloc[-1]
        latest_volume = volumes.iloc[-1]
        avg_volume = volumes.mean()
        latest_rsi = rsi.iloc[-1]

        volume_ok = latest_volume > avg_volume * min_volume
        trend_ok = latest_close > ema_20.iloc[-1] if trend_required else True
        rsi_ok = rsi_low < latest_rsi < rsi_high
        breakout_ok = bool(abs(latest_close - close_prices.rolling(5).max().iloc[-1]) < 0.0001) if breakout_required else True


        price_ok = min_price <= latest_close <= max_price

        momentum = volume_price_momentum_signal(data)

        if all([volume_ok, trend_ok, rsi_ok, breakout_ok, price_ok]):
            return {
                "Stock": ticker.replace(".NS", ""),
                "Price (₹)": f"₹{latest_close:.2f}",
                "Volume (x)": f"{latest_volume/avg_volume:.1f}",
                "RSI": f"{latest_rsi:.1f}",
                "Trend": "🟢" if latest_close > ema_20.iloc[-1] else "🔴",
                "Why Buy?": "📈 Breakout + Volume Spike",
                "Momentum Signal": momentum['signal'],
                "Volume Ratio": f"{momentum['volume_ratio']:.2f}",
                "Price Change %": f"{momentum['price_change_pct']:.2f}%"
            }
        else:
            # Show momentum buys even if other filters fail
            if momentum['signal'] == "BUY":
                return {
                    "Stock": ticker.replace(".NS", ""),
                    "Price (₹)": f"₹{latest_close:.2f}",
                    "Volume (x)": f"{latest_volume/avg_volume:.1f}",
                    "RSI": f"{latest_rsi:.1f}",
                    "Trend": "🟢" if latest_close > ema_20.iloc[-1] else "🔴",
                    "Why Buy?": "⚡ Momentum BUY signal (volume spike + price gain)",
                    "Momentum Signal": momentum['signal'],
                    "Volume Ratio": f"{momentum['volume_ratio']:.2f}",
                    "Price Change %": f"{momentum['price_change_pct']:.2f}%"
                }

    except Exception as e:
        st.error(f"Error scanning {ticker}: {str(e)}")
    return None

# --- SCAN SELECTED STOCKS ---
if st.button("🔍 Scan Selected Stocks"):
    with st.spinner("Scanning selected stocks..."):
        results = []
        for ticker in filtered_tickers:
            result = scan_stock(ticker)
            if result:
                results.append(result)
    if results:
        st.success(f"✅ Found {len(results)} potential swing trades.")
        df_results = pd.DataFrame(results)
        st.dataframe(df_results, hide_index=True)
    else:
        st.warning("⚠️ No stocks matched the criteria. Adjust your filters and try again.")

# --- ANALYZE SPECIFIC STOCK ---
st.markdown("---")
st.subheader("🔎 Analyze a Specific Stock")
with st.form("analyze_stock_form"):
    user_stock = st.text_input("Enter NSE Stock Symbol (e.g., INFY)")
    analyze_button = st.form_submit_button("🔍 Analyze")

if analyze_button and user_stock:
    full_ticker = user_stock.upper().strip() + ".NS"
    try:
        data = yf.download(full_ticker, period="1mo", progress=False)
        if not data.empty and len(data) > 14:
            data = data.dropna()
            close_prices = pd.Series(data["Close"].values.flatten(), index=data.index)
            volumes = pd.Series(data["Volume"].values.flatten(), index=data.index)

            ema_20 = EMAIndicator(close=close_prices, window=20).ema_indicator()
            rsi = sma_rsi(close_prices, window=14)

            latest_close = close_prices.iloc[-1]
            latest_volume = volumes.iloc[-1]
            avg_volume = volumes.mean()
            latest_rsi = rsi.iloc[-1]
            trend = "🟢" if latest_close > ema_20.iloc[-1] else "🔴"

            remarks = []
            if latest_close != close_prices.rolling(5).max().iloc[-1]:
                remarks.append("Not at 5-day breakout")
            if latest_volume < avg_volume * min_volume:
                remarks.append("Low volume")
            if not (rsi_low < latest_rsi < rsi_high):
                remarks.append("RSI not in range")
            if latest_close < min_price or latest_close > max_price:
                remarks.append("Price not in range")

            st.markdown("#### 🔬 Result:")
            result = {
                "Stock": user_stock.upper(),
                "Price (₹)": f"₹{latest_close:.2f}",
                "Volume (x)": f"{latest_volume / avg_volume:.1f}",
                "RSI": f"{latest_rsi:.1f}",
                "Trend": trend,
                "Remarks?": "✅ Good for Swing Trade" if not remarks else "❌ " + ", ".join(remarks)
            }
            st.dataframe(pd.DataFrame([result]), hide_index=True)
        else:
            st.error("❌ Not enough data for analysis.")
    except Exception as e:
        st.error(f"Error fetching data for {user_stock.upper()}: {str(e)}")

# --- AI BUTTON ---
# Call AI section
st.markdown("---")
ai_prediction.run_ai_prediction()

# --- FOOTER ---
st.markdown("---")
st.caption("⚡ Developed by Roshith •  Please feed your comments to roshith77@gmail.com")
