import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
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
                st.write(f"🔍 Scaled close shape: {scaled_close.shape}")

                # LSTM training data
                st.write("🔄 Preparing LSTM sequences...")
                X, y = prepare_lstm_data(scaled_close)
                st.write(f"📊 X shape: {X.shape}")
                st.write(f"📉 y shape: {y.shape}")

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

                st.write(f"📈 Raw predictions (scaled): {future_preds[:5]}...")

                future_preds = np.array(future_preds).reshape(-1, 1)
                st.write(f"📐 Shape before inverse_transform: {future_preds.shape}")

                future_preds = scaler.inverse_transform(future_preds).flatten()
                st.write(f"✅ Final predicted prices: {future_preds[:5]}...")

                future_dates = pd.bdate_range(start=df.index[-1] + timedelta(days=1), periods=pred_days)
                forecast_df = pd.DataFrame({"Date": future_dates, "Predicted Close": future_preds})

                st.success("✅ Forecast generated successfully!")
                st.dataframe(forecast_df, use_container_width=True, hide_index=True)

                # Plot forecast vs history
                fig, ax = plt.subplots(figsize=(10, 4))
                df['Close'].plot(ax=ax, label='Historical Close', color='blue')
                forecast_df.set_index('Date')['Predicted Close'].plot(ax=ax, label='Forecast', color='orange')
                ax.set_title(f"LSTM Forecast for {user_stock.upper()} for next {pred_days} days")
                ax.legend()
                st.pyplot(fig)

            except Exception as e:
                st.error(f"❌ Error during AI prediction: {str(e)}")
