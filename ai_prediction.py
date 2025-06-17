import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands
from ta.volume import VolumeWeightedAveragePrice

# --- LSTM HELPER FUNCTION ---
def prepare_lstm_data(data, n_steps=30):
    X, y = [], []
    for i in range(n_steps, len(data)):
        X.append(data[i - n_steps:i])
        y.append(data[i, 0])  # Predicting close price only
    return np.array(X), np.array(y)

# --- TECHNICAL INDICATORS (Optional — still used for metrics display) ---
def add_technical_indicators(df):
    try:
        if len(df) < 30:
            st.error(f"Need at least 30 data points, only have {len(df)}")
            return None

        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']

        df['RSI'] = RSIIndicator(close=close, window=14).rsi()
        df['EMA_20'] = EMAIndicator(close=close, window=20).ema_indicator()
        df['MACD'] = MACD(close=close).macd()
        df['MACD_Signal'] = MACD(close=close).macd_signal()
        df['BB_Upper'] = BollingerBands(close=close, window=20).bollinger_hband()
        df['BB_Lower'] = BollingerBands(close=close, window=20).bollinger_lband()

        df.ffill(inplace=True)
        df.bfill(inplace=True)
        return df

    except Exception as e:
        st.error(f"Technical indicators failed: {str(e)}")
        return None

# --- SIMPLIFIED SIGNAL BASED ONLY ON AI PREDICTION ---
def generate_signals(df, forecast):
    try:
        last_row = df.iloc[-1]
        current_close = float(last_row['Close'])
        pred_close = float(forecast['Predicted Close'].iloc[0])

        if pred_close > current_close:
            final_signal = "BUY"
            reasons = [f"Predicted price ₹{pred_close:.2f} is higher than current price ₹{current_close:.2f}"]
        else:
            final_signal = "HOLD"
            reasons = [f"Predicted price ₹{pred_close:.2f} is not higher than current price ₹{current_close:.2f}"]

        return final_signal, reasons

    except Exception as e:
        st.error(f"Signal generation failed: {str(e)}")
        return "ERROR", ["Could not generate signals"]

# --- MAIN APP ---
def run_ai_prediction():
    st.title("📈 AI Stock Prediction Dashboard")

    with st.expander("⚙️ Settings", expanded=True):
        col1, col2 = st.columns(2)
        user_stock = col1.text_input("Stock Symbol (e.g., INFY)", value="INFY")
        pred_days = col2.slider("Forecast Days", 5, 15, 7)

    if st.button("🚀 Generate Forecast"):
        ticker = f"{user_stock.upper().strip()}.NS"

        with st.spinner("Processing..."):
            try:
                df = yf.download(ticker, period="6mo", interval="1d", progress=False)
                if df.empty:
                    st.error("No data found for this stock")
                    return

                df = add_technical_indicators(df)
                if df is None:
                    return

                features = ['Close', 'RSI', 'EMA_20', 'MACD', 'BB_Upper', 'BB_Lower']
                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(df[features])

                X, y = prepare_lstm_data(scaled_data)
                X = X.reshape((X.shape[0], X.shape[1], len(features)))

                model = Sequential([
                    LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
                    Dropout(0.2),
                    LSTM(64),
                    Dropout(0.2),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mse')
                model.fit(X, y, epochs=50, batch_size=32, verbose=0)

                last_seq = scaled_data[-30:]
                future_preds = []
                for _ in range(pred_days):
                    next_pred = model.predict(last_seq.reshape(1, 30, len(features)), verbose=0)[0, 0]
                    future_preds.append(next_pred)
                    new_row = np.zeros(len(features))
                    new_row[0] = next_pred
                    last_seq = np.vstack([last_seq[1:], new_row])

                dummy = np.zeros((len(future_preds), len(features)))
                dummy[:, 0] = future_preds
                future_preds = scaler.inverse_transform(dummy)[:, 0]

                future_dates = pd.bdate_range(start=df.index[-1] + timedelta(days=1), periods=pred_days)
                forecast_df = pd.DataFrame({
                    "Date": future_dates,
                    "Predicted Close": future_preds
                })

                signal, reasons = generate_signals(df, forecast_df)

                st.success("🎯 Forecast Complete!")

                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name="Price"
                ))
                fig.add_trace(go.Scatter(
                    x=forecast_df['Date'],
                    y=forecast_df['Predicted Close'],
                    line=dict(color='green', dash='dot'),
                    name="Forecast"
                ))
                fig.update_layout(title=f"{user_stock} Price & Forecast", xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

                if signal == "BUY":
                    st.success(f"✅ SIGNAL: {signal}")
                elif signal == "HOLD":
                    st.warning(f"🔄 SIGNAL: {signal}")
                else:
                    st.error(f"❌ SIGNAL: {signal}")

                st.subheader("Reason:")
                for reason in reasons:
                    st.write(f"- {reason}")

                # Key Metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Current Price", f"₹{df['Close'].iloc[-1]:.2f}")
                col2.metric("Predicted Price", f"₹{float(forecast_df['Predicted Close'].iloc[0]):.2f}",
                            f"{((float(forecast_df['Predicted Close'].iloc[0]) / df['Close'].iloc[-1]) - 1) * 100:.2f}%")
                col3.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}")

                st.subheader("Forecast Details:")
                st.dataframe(forecast_df.set_index('Date'))

            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    run_ai_prediction()
