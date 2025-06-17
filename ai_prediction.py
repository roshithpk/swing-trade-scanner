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
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.volatility import BollingerBands
from ta.volume import VolumeWeightedAveragePrice

# --- LSTM HELPER FUNCTION ---
def prepare_lstm_data(data, n_steps=30):
    X, y = [], []
    for i in range(n_steps, len(data)):
        X.append(data[i - n_steps:i])
        y.append(data[i, 0])  # Predicting close price only
    return np.array(X), np.array(y)

# --- TECHNICAL INDICATORS ---
def add_technical_indicators(df):
    # Convert all inputs to 1D numpy arrays
    def to_1d(series):
        arr = series.values if isinstance(series, (pd.Series, pd.DataFrame)) else series
        return arr.ravel()
    
    close = to_1d(df['Close'])
    high = to_1d(df['High'])
    low = to_1d(df['Low'])
    volume = to_1d(df['Volume'])

    try:
        # Momentum Indicators
        df['RSI'] = RSIIndicator(close=close, window=14).rsi()
        stoch = StochasticOscillator(high=high, low=low, close=close, window=14)
        df['Stoch_%K'] = stoch.stoch()
        df['Stoch_%D'] = stoch.stoch_signal()
        
        # Trend Indicators
        df['EMA_20'] = EMAIndicator(close=close, window=20).ema_indicator()
        df['EMA_50'] = EMAIndicator(close=close, window=50).ema_indicator()
        macd = MACD(close=close)
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['ADX'] = ADXIndicator(high=high, low=low, close=close, window=14).adx()
        
        # Volatility Indicators
        bb = BollingerBands(close=close, window=20, window_dev=2)
        df['BB_Upper'] = bb.bollinger_hband()
        df['BB_Lower'] = bb.bollinger_lband()
        
        # Volume Indicators
        df['VWAP'] = VolumeWeightedAveragePrice(
            high=high, low=low, close=close, volume=volume, window=14
        ).volume_weighted_average_price()
        
        return df.dropna()
    except Exception as e:
        st.error(f"Technical indicator error: {str(e)}")
        return df

# --- TRADING SIGNALS ---
def generate_signals(df, forecast):
    try:
        signals = []
        reasons = []
        last_row = df.iloc[-1].copy()
        pred = forecast.iloc[0]['Predicted Close']
        
        # Price vs Prediction
        price_diff = (pred - last_row['Close']) / last_row['Close']
        if price_diff > 0.02:
            signals.append("BUY")
            reasons.append(f"Predicted +{price_diff:.2%} above current")
        elif price_diff < -0.02:
            signals.append("SELL") 
            reasons.append(f"Predicted {abs(price_diff):.2%} below current")
        else:
            signals.append("HOLD")
            reasons.append("Prediction within 2% range")
        
        # RSI
        if last_row['RSI'] < 30:
            signals.append("BUY")
            reasons.append(f"RSI {last_row['RSI']:.1f} (oversold)")
        elif last_row['RSI'] > 70:
            signals.append("SELL")
            reasons.append(f"RSI {last_row['RSI']:.1f} (overbought)")
        
        # MACD
        if last_row['MACD'] > last_row['MACD_Signal']:
            signals.append("BUY")
            reasons.append("MACD above signal line")
        else:
            signals.append("SELL")
            reasons.append("MACD below signal line")
        
        # Bollinger Bands
        if last_row['Close'] < last_row['BB_Lower']:
            signals.append("BUY")
            reasons.append("Price below lower band")
        elif last_row['Close'] > last_row['BB_Upper']:
            signals.append("SELL")
            reasons.append("Price above upper band")
        
        # Determine final signal
        buy_count = signals.count("BUY")
        sell_count = signals.count("SELL")
        final_signal = "BUY" if buy_count > sell_count else "SELL" if sell_count > buy_count else "HOLD"
        
        return final_signal, list(set(reasons))  # Remove duplicates
    
    except Exception as e:
        st.error(f"Signal generation error: {str(e)}")
        return "ERROR", [str(e)]

# --- MAIN APP ---
def run_ai_prediction():
    st.title("📈 AI Stock Prediction Dashboard")
    
    with st.expander("⚙️ Settings", expanded=True):
        col1, col2 = st.columns(2)
        user_stock = col1.text_input("Stock Symbol (e.g., INFY)", value="INFY")
        pred_days = col2.slider("Forecast Days", 5, 15, 7)
    
    if st.button("🚀 Generate Forecast"):
        ticker = f"{user_stock.upper().strip()}.NS"
        
        with st.spinner("Fetching data..."):
            try:
                # Data Collection
                df = yf.download(ticker, period="6mo", interval="1d", progress=False)
                if df.empty:
                    st.error("No data found for this stock")
                    return
                
                # Technical Indicators
                df = add_technical_indicators(df)
                if df.empty:
                    st.error("Technical indicators failed")
                    return
                
                # Model Training
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
                
                # Prediction
                last_seq = scaled_data[-30:]
                future_preds = []
                for _ in range(pred_days):
                    next_pred = model.predict(last_seq.reshape(1, 30, len(features)))[0,0]
                    future_preds.append(next_pred)
                    last_seq = np.append(last_seq[1:], [[next_pred] + [0]*(len(features)-1)], axis=0)
                
                # Inverse transform
                dummy = np.zeros((len(future_preds), len(features)))
                dummy[:,0] = future_preds
                future_preds = scaler.inverse_transform(dummy)[:,0]
                
                future_dates = pd.bdate_range(start=df.index[-1] + timedelta(days=1), periods=pred_days)
                forecast_df = pd.DataFrame({
                    "Date": future_dates,
                    "Predicted Close": future_preds
                })
                
                # Generate Signals
                signal, reasons = generate_signals(df, forecast_df)
                
                # Display Results
                st.success("Forecast Complete!")
                
                # Candlestick Chart
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
                fig.update_layout(
                    title=f"{user_stock} Price & Forecast",
                    xaxis_rangeslider_visible=False
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Signal Card
                if signal == "BUY":
                    st.success(f"✅ SIGNAL: {signal}")
                elif signal == "SELL":
                    st.error(f"❌ SIGNAL: {signal}")
                else:
                    st.warning(f"🔄 SIGNAL: {signal}")
                
                st.subheader("Reasons:")
                for reason in reasons:
                    st.write(f"- {reason}")
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Current Price", f"₹{df['Close'].iloc[-1]:.2f}")
                col2.metric("Predicted Price", f"₹{forecast_df['Predicted Close'].iloc[0]:.2f}",
                          f"{((forecast_df['Predicted Close'].iloc[0]/df['Close'].iloc[-1])-1)*100:.2f}%")
                col3.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}")
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    run_ai_prediction()
