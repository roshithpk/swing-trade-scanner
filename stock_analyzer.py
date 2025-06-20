import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


def sma_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def prepare_lstm_data(data, n_steps=30):
    X, y = [], []
    for i in range(n_steps, len(data)):
        X.append(data[i - n_steps:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def analyze_stock(ticker: str, settings: dict):
    try:
        df = yf.download(ticker, period="6mo", interval="1d", progress=False)
        if df.empty:
            return {"error": "No data found"}

        df.dropna(inplace=True)

        close = df['Close']
        volume = df['Volume']
        df['RSI'] = sma_rsi(close, window=14)
        df['EMA_20'] = EMAIndicator(close=close, window=20).ema_indicator()

        current_price = close.iloc[-1]
        avg_volume = volume.mean()
        latest_volume = volume.iloc[-1]
        latest_rsi = df['RSI'].iloc[-1]
        ema_20 = df['EMA_20'].iloc[-1]
        trend = "🟢 Uptrend" if current_price > ema_20 else "🔴 Downtrend"

        remarks = []
        if latest_volume < avg_volume * settings['min_volume_x_avg']:
            remarks.append("Low Volume")
        if not (settings['rsi_low'] < latest_rsi < settings['rsi_high']):
            remarks.append("RSI out of range")
        if not (settings['min_price'] <= current_price <= settings['max_price']):
            remarks.append("Price out of range")

        # Forecast
        features = ['Close', 'RSI', 'EMA_20']
        df.dropna(inplace=True)
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[features])
        X, y = prepare_lstm_data(scaled_data)
        X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))

        model = build_lstm_model((X.shape[1], X.shape[2]))
        model.fit(X, y, epochs=50, batch_size=32, verbose=0)

        last_seq = scaled_data[-30:]
        next_pred_scaled = model.predict(last_seq.reshape(1, 30, len(features)), verbose=0)[0][0]
        dummy_row = np.zeros(len(features))
        dummy_row[0] = next_pred_scaled
        inverse = scaler.inverse_transform([dummy_row])
        predicted_price = inverse[0][0]

        signal = "BUY" if predicted_price > current_price * 1.02 else "SELL" if predicted_price < current_price * 0.98 else "HOLD"
        growth_pct = ((predicted_price / current_price) - 1) * 100

        return {
            "Stock": ticker.replace(".NS", ""),
            "Current Price": f"₹{current_price:.2f}",
            "Predicted Price": f"₹{predicted_price:.2f}",
            "Expected Growth %": f"{growth_pct:.2f}%",
            "Trend": trend,
            "RSI": f"{latest_rsi:.1f}",
            "Volume (x Avg)": f"{latest_volume / avg_volume:.1f}",
            "Signal": signal,
            "Remarks": remarks if remarks else ["Good for swing"],
            "Suggested Entry": f"₹{current_price:.2f}" if signal == "BUY" else "-",
            "Suggested Exit": f"₹{predicted_price:.2f}" if signal == "BUY" else "-",
            "Days to Target (approx)": settings.get('forecast_days', 7)
        }

    except Exception as e:
        return {"error": str(e)}
