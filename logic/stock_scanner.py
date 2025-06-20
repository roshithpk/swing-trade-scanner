import streamlit as st
import pandas as pd
import yfinance as yf
from logic.utils import calculate_rsi, determine_trend, suggest_entry_exit

@st.cache_data
def load_stock_list():
    try:
        return pd.read_csv("data/stocks.csv")
    except Exception as e:
        st.error(f"Error loading stock list: {e}")
        return pd.DataFrame(columns=["Ticker", "Name", "Category"])

def run_category_stock_scanner():
    st.subheader("📂 Scan Stocks by Category")

    # Load stock list
    stock_df = load_stock_list()
    if stock_df.empty:
        return

    categories = ["All"] + sorted(stock_df["Category"].dropna().unique())
    selected_category = st.selectbox("Select Category", categories)

    # Filtered list
    if selected_category == "All":
        filtered_df = stock_df
    else:
        filtered_df = stock_df[stock_df["Category"] == selected_category]

    tickers = filtered_df["Ticker"].dropna().unique().tolist()

    # Filters
    with st.expander("⚙️ Filter Settings", expanded=True):
        min_rsi = st.slider("Min RSI", 10, 50, 30)
        max_rsi = st.slider("Max RSI", 50, 90, 70)
        require_uptrend = st.checkbox("Require Uptrend (Price > EMA)", value=True)

    if st.button("🔍 Start Scanning"):
        results = []

        with st.spinner("Scanning stocks..."):
            for ticker in tickers:
                try:
                    full_ticker = ticker.strip().upper() + ".NS"
                    data = yf.download(full_ticker, period="3mo", interval="1d", progress=False)
                    if data.empty or len(data) < 30:
                        continue

                    data.dropna(inplace=True)
                    data["RSI"] = calculate_rsi(data["Close"])
                    trend = determine_trend(data["Close"])
                    latest_close = data["Close"].iloc[-1]
                    rsi_value = data["RSI"].iloc[-1]
                    entry, exit_ = suggest_entry_exit(data["Close"])
                    signal = None
                    remarks = []

                    # Signal logic
                    if rsi_value < 30 and trend:
                        signal = "BUY"
                        remarks.append("RSI < 30 & Uptrend")
                    elif rsi_value > 70:
                        signal = "SELL"
                        remarks.append("RSI > 70")
                    elif trend:
                        signal = "HOLD"
                        remarks.append("Trending but neutral RSI")
                    else:
                        signal = "AVOID"
                        remarks.append("Not in uptrend")

                    # Apply filters
                    if not (min_rsi <= rsi_value <= max_rsi):
                        continue
                    if require_uptrend and not trend:
                        continue

                    results.append({
                        "Stock": ticker,
                        "Price (₹)": f"{latest_close:.2f}",
                        "RSI": f"{rsi_value:.1f}",
                        "Trend": "📈" if trend else "📉",
                        "Signal": signal,
                        "Entry": f"{entry}",
                        "Exit": f"{exit_}",
                        "Remarks": ", ".join(remarks)
                    })
                except Exception as e:
                    st.warning(f"⚠️ {ticker}: {e}")
                    continue

        if results:
            st.success(f"✅ Found {len(results)} matching stocks.")
            st.dataframe(pd.DataFrame(results), use_container_width=True)
        else:
            st.warning("⚠️ No stocks matched the filters.")

