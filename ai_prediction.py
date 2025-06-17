import streamlit as st
import pandas as pd
import yfinance as yf
from prophet import Prophet
import plotly.graph_objects as go

def run_ai_prediction():
    st.header("🧠 AI-Based Stock Price Forecast")
    st.caption("This model uses historical daily data and Facebook Prophet to forecast the next 14 days.")

    with st.expander("🔮 AI Prediction Section", expanded=True):
        stock_input = st.text_input("Enter NSE Stock Symbol (e.g., INFY)", key="ai_input")
        if st.button("Predict", key="ai_button"):
            if stock_input:
                ticker = stock_input.strip().upper() + ".NS"
                try:
                    df = yf.download(ticker, period="6mo", interval="1d", progress=False)
                    if df.empty:
                        st.error("⚠️ No data found.")
                        return

                    df.reset_index(inplace=True)
                    df = df[["Date", "Open", "High", "Low", "Close"]].dropna()
                    df["Close"] = pd.to_numeric(df["Close"], errors='coerce')
                    df = df.dropna(subset=["Close"])
                    df["Date"] = pd.to_datetime(df["Date"])

                    # Prepare data for Prophet
                    prophet_df = df[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
                    prophet_df["y"] = prophet_df["y"].astype(float)

                    model = Prophet()
                    model.fit(prophet_df)

                    future = model.make_future_dataframe(periods=14)
                    forecast = model.predict(future)

                    # Merge actuals with forecast
                    forecast_df = forecast[["ds", "yhat"]].rename(columns={"ds": "Date", "yhat": "Predicted"})
                    merged = pd.merge(df, forecast_df, on="Date", how="outer")

                    # Plot candlestick with predicted line
                    fig = go.Figure()

                    # Actual candlestick
                    fig.add_trace(go.Candlestick(
                        x=df["Date"],
                        open=df["Open"],
                        high=df["High"],
                        low=df["Low"],
                        close=df["Close"],
                        name="Actual",
                        increasing_line_color='green',
                        decreasing_line_color='red'
                    ))

                    # Prediction line
                    pred_range = merged[merged["Predicted"].notnull() & merged["Date"] > df["Date"].max()]
                    fig.add_trace(go.Scatter(
                        x=pred_range["Date"],
                        y=pred_range["Predicted"],
                        mode='lines+markers',
                        name="Predicted",
                        line=dict(color='blue', dash="dot")
                    ))

                    fig.update_layout(
                        title=f"📈 {stock_input.upper()} - AI Forecast (Next 14 Days)",
                        xaxis_title="Date",
                        yaxis_title="Price (₹)",
                        xaxis_rangeslider_visible=False,
                        template="plotly_dark",
                        height=600
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Show table of predicted prices
                    pred_table = pred_range[["Date", "Predicted"]].copy()
                    pred_table["Predicted"] = pred_table["Predicted"].apply(lambda x: f"₹{x:.2f}")
                    st.markdown("### 📅 Forecast Table")
                    st.dataframe(pred_table, hide_index=True)

                except Exception as e:
                    st.error(f"Error during AI prediction: {str(e)}")
            else:
                st.warning("⚠️ Please enter a valid stock symbol.")
