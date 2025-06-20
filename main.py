import streamlit as st
from logic import stock_analyzer, stock_scanner

st.set_page_config(page_title="📊 Stock Analyzer & Scanner", layout="wide")

st.title("📈 Stock Analyzer & Category Scanner")

# Tabs for the two features
tabs = st.tabs(["🔍 Analyze a Stock", "📂 Category Stock Scanner"])

# --- Tab 1: Analyze a Single Stock ---
with tabs[0]:
    stock_analyzer.run_single_stock_analysis()

# --- Tab 2: Scan by Category ---
with tabs[1]:
    stock_scanner.run_category_stock_scanner()

