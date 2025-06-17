# AI_Prediction.py

import streamlit as st

def run():
    st.set_page_config(page_title="🤖 AI Prediction", layout="wide")
    st.title("🤖 AI-Based Stock Price Prediction")

    st.markdown("### 📊 Enter stock name to begin prediction")
    stock = st.text_input("Enter NSE Stock Symbol (e.g., INFY)")

    if stock:
        st.info(f"AI-based prediction placeholder for **{stock.upper()}**")
        # Add ML prediction logic later
    else:
        st.warning("Please enter a stock symbol above.")

    # Optional: Back Button
    if st.button("🔙 Back to Scanner"):
        st.session_state.page = "main"
        st.experimental_rerun()
