import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator

# --- LSTM HELPER FUNCTION ---
def prepare_lstm_data(series, n_steps=30):
    X, y = [], []
    for i in range(n_steps, len(series)):
        X.append(series[i - n_steps:i])
        y.append(series[i])
    return np.array(X), np.array(y)

# --- MAIN FUNCTION ---
def run_ai_prediction():
    st.markdown("---")
    st.header("🤖 AI Based Price Forecast")
    st.caption("Uses LSTM model with technical indicators to forecast closing price")

    with st.expander("🔮 AI Prediction Panel", expanded=False):
        user_stock = st.text_input("Enter NSE Stock Symbol for Prediction (e.g., INFY)")
        pred_days = st.slider("Prediction Horizon (days)", min_value=5, max_value=15, value=10)

        if st.button("🚀 Run AI Forecast") and user_stock:
            ticker = user_stock.upper().strip() + ".NS"

            try:
                st.write("📥 Fetching stock data...")
                df = yf.download(ticker, period="1y", progress=False)
                st.write("✅ Raw data fetched")
                st.dataframe(df.tail())

                if df.empty or len(df) < 90:
                    st.error("❌ Not enough data for forecasting.")
                    return

                df = df.dropna()
                st.write("🧪 Adding Technical Indicators...")
                df['RSI'] = RSIIndicator(close=df['Close'].squeeze(), window=14).rsi().fillna(method='bfill')
                df['EMA_20'] = EMAIndicator(close=df['Close'].squeeze(), window=20).ema_indicator().fillna(method='bfill')
                st.write("🧪 Adding and checking...")
                df = df.dropna()

                st.write(f"✅ Data after indicators: {df.shape}")
                st.dataframe(df.tail())

                # Normalize close price
                st.write("📊 Scaling close prices...")
                scaler = MinMaxScaler()
                scaled_close = scaler.fit_transform(df[['Close']])

                # LSTM training data
                st.write("🔄 Preparing LSTM sequences...")
                X, y = prepare_lstm_data(scaled_close)
                X = X.reshape((X.shape[0], X.shape[1], 1))

                # Build LSTM model
                st.write("⚙️ Training LSTM model...")
                model = Sequential()
                model.add(LSTM(50, activation='relu', input_shape=(X.shape[1], 1)))
                model.add(Dense(1))
                model.compile(optimizer='adam', loss='mse')
                model.fit(X, y, epochs=20, verbose=0)
                st.write("✅ Model training complete!")

                # Predict next days
                st.write("🔮 Predicting future prices...")
                last_seq = scaled_close[-30:].reshape((1, 30, 1))
                future_preds = []

                for _ in range(pred_days):
                    next_pred = model.predict(last_seq)[0, 0]
                    future_preds.append(next_pred)
                    last_seq = np.append(last_seq[:, 1:, :], [[[next_pred]]], axis=1)

                future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()
                future_dates = pd.bdate_range(start=df.index[-1] + timedelta(days=1), periods=pred_days)
                forecast_df = pd.DataFrame({"Date": future_dates, "Predicted Close": future_preds})

                st.success("✅ Forecast generated successfully!")
                st.dataframe(forecast_df, use_container_width=True, hide_index=True)

                # Create candlestick chart with forecast
                st.write("📊 Candlestick Chart with Forecast")
                
                # Create figure
                fig = go.Figure()
                
                # Add candlestick
                fig.add_trace(go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name='Price History'
                ))
                
                # Add EMA
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['EMA_20'],
                    line=dict(color='orange', width=1),
                    name='20 EMA'
                ))
                
                # Add forecast
                fig.add_trace(go.Scatter(
                    x=forecast_df['Date'],
                    y=forecast_df['Predicted Close'],
                    line=dict(color='green', width=2, dash='dot'),
                    name='AI Forecast',
                    mode='lines+markers'
                ))
                
                # Update layout
                fig.update_layout(
                    title=f"{user_stock.upper()} Candlestick Chart with {pred_days}-Day AI Forecast",
                    xaxis_title='Date',
                    yaxis_title='Price',
                    xaxis_rangeslider_visible=False,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"❌ Error during AI prediction: {str(e)}")

# Run the app
if __name__ == "__main__":
    run_ai_prediction()
