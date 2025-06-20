
# category_scanner.py

import streamlit as st
import pandas as pd
from stock_analyzer import analyze_stock

# --- Load Stock List ---
@st.cache_data
def load_stock_list():
    df = pd.read_csv("data/stocks.csv")  # Ensure columns: Ticker, Name, Category
    return df.dropna(subset=["Ticker", "Category"])

# --- Scanner UI ---
def run_category_scanner():
    st.title("📂 Category-based Stock Scanner")

    stock_df = load_stock_list()
    categories = ["All"] + sorted(stock_df["Category"].unique())
    selected_category = st.selectbox("Select Category", categories, index=0)

    if selected_category == "All":
        selected_stocks = stock_df
    else:
        selected_stocks = stock_df[stock_df["Category"] == selected_category]

    if st.button("🔍 Scan Category"):
        st.info(f"Scanning {len(selected_stocks)} stocks from category: {selected_category}...")
        results = []

        progress_bar = st.progress(0)
        for idx, row in enumerate(selected_stocks.itertuples(), start=1):
            ticker = row.Ticker
            result = analyze_stock(ticker, return_dict=True)  # expects dict output
            if result:
                results.append(result)
            progress_bar.progress(idx / len(selected_stocks))

        progress_bar.empty()

        if results:
            st.success(f"✅ Found {len(results)} valid stocks.")
            result_df = pd.DataFrame(results)
            st.dataframe(result_df, use_container_width=True, hide_index=True)

            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Results", data=csv, file_name="scan_results.csv", mime="text/csv")
        else:
            st.warning("⚠️ No matching stocks found. Try changing category or updating data.")

if __name__ == "__main__":
    run_category_scanner()
