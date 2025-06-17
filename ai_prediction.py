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
    try:
        # Check if we have enough data (minimum 30 days)
        if len(df) < 30:
            st.error(f"Need at least 30 data points, only have {len(df)}")
            return None
        
        # Ensure we have required price columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            return None
        
        # Convert to pandas Series with proper index
        close = pd.Series(df['Close'].values.flatten(), index=df.index)
        high = pd.Series(df['High'].values.flatten(), index=df.index)
        low = pd.Series(df['Low'].values.flatten(), index=df.index)
        volume = pd.Series(df['Volume'].values.flatten(), index=df.index)

        # Calculate each indicator with individual error handling
        indicators = {
            'RSI': lambda: RSIIndicator(close=close, window=14).rsi(),
            'Stoch_%K': lambda: StochasticOscillator(high=high, low=low, close=close, window=14).stoch(),
            'Stoch_%D': lambda: StochasticOscillator(high=high, low=low, close=close, window=14).stoch_signal(),
            'EMA_20': lambda: EMAIndicator(close=close, window=20).ema_indicator(),
            'EMA_50': lambda: EMAIndicator(close=close, window=50).ema_indicator(),
            'MACD': lambda: MACD(close=close).macd(),
            'MACD_Signal': lambda: MACD(close=close).macd_signal(),
            'MACD_Hist': lambda: MACD(close=close).macd_diff(),
            'BB_Upper': lambda: BollingerBands(close=close, window=20, window_dev=2).bollinger_hband(),
            'BB_Middle': lambda: BollingerBands(close=close, window=20, window_dev=2).bollinger_mavg(),
            'BB_Lower': lambda: BollingerBands(close=close, window=20, window_dev=2).bollinger_lband(),
            'VWAP': lambda: VolumeWeightedAveragePrice(high=high, low=low, close=close, volume=volume, window=14).volume_weighted_average_price()
        }

        for name, calc in indicators.items():
            try:
                df[name] = calc()
            except Exception as e:
                st.warning(f"Could not calculate {name}: {str(e)}")
                df[name] = np.nan

        # Fill any remaining NA values
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        
        # Verify we have the essential indicators
        required_indicators = ['RSI', 'EMA_20', 'MACD', 'BB_Upper', 'BB_Lower']
        if not all(ind in df.columns for ind in required_indicators):
            st.error("Failed to calculate essential indicators")
            return None
            
        return df
    
    except Exception as e:
        st.error(f"Technical indicators failed: {str(e)}")
        return None

# --- TRADING SIGNALS ---
def generate_signals(df, forecast):
    try:
        signals = []
        reasons = []
        last_row = df.iloc[-1]
        pred = float(forecast['Predicted Close'].iloc[0])
        
        # 1. Price vs Prediction
        price_diff = (pred - last_row['Close']) / last_row['Close']
        if price_diff > 0.02:
            signals.append("BUY")
            reasons.append(f"Predicted price {price_diff:.2%} higher")
        elif price_diff < -0.02:
            signals.append("SELL") 
            reasons.append(f"Predicted price {abs(price_diff):.2%} lower")
        else:
            signals.append("HOLD")
            reasons.append("Prediction within 2% range")
        
        # 2. RSI
        if 'RSI' in df.columns:
            if last_row['RSI'] < 30:
                signals.append("BUY")
                reasons.append(f"RSI {last_row['RSI']:.1f} (oversold)")
            elif last_row['RSI'] > 70:
                signals.append("SELL")
                reasons.append(f"RSI {last_row['RSI']:.1f} (overbought)")
        
        # 3. MACD
        if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
            if last_row['MACD'] > last_row['MACD_Signal']:
                signals.append("BUY")
                reasons.append("MACD above signal line")
            else:
                signals.append("SELL")
                reasons.append("MACD below signal line")
        
        # 4. Bollinger Bands
        if 'BB_Lower' in df.columns and 'BB_Upper' in df.columns:
            if last_row['Close'] < last_row['BB_Lower']:
                signals.append("BUY")
                reasons.append("Price below lower band")
            elif last_row['Close'] > last_row['BB_Upper']:
                signals.append("SELL")
                reasons.append("Price above upper band")
        
        # Determine final signal
        buy_count = signals.count("BUY")
        sell_count = signals.count("SELL")
        
        if buy_count > sell_count:
            final_signal = "BUY"
        elif sell_count > buy_count:
            final_signal = "SELL"
        else:
            final_signal = "HOLD"
        
        return final_signal, list(set(reasons))
    
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
                # 1. Data Collection
                df = yf.download(ticker, period="6mo", interval="1d", progress=False)
                if df.empty:
                    st.error("No data found for this stock")
                    return
                
                # 2. Add Technical Indicators
                df = add_technical_indicators(df)
                if df is None:
                    st.error("Failed to calculate technical indicators")
                    return
                
                # 3. Prepare LSTM Data
                features = ['Close', 'RSI', 'EMA_20', 'MACD', 'BB_Upper', 'BB_Lower']
                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(df[features])
                
                X, y = prepare_lstm_data(scaled_data)
                X = X.reshape((X.shape[0], X.shape[1], len(features)))
                
                # 4. Build and Train Model
                model = Sequential([
                    LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
                    Dropout(0.2),
                    LSTM(64),
                    Dropout(0.2),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mse')
                model.fit(X, y, epochs=50, batch_size=32, verbose=0)
                
                # 5. Generate Predictions
                last_seq = scaled_data[-30:]
                future_preds = []
                for _ in range(pred_days):
                    next_pred = model.predict(last_seq.reshape(1, 30, len(features)), verbose=0)[0,0]
                    future_preds.append(next_pred)
                    new_row = np.zeros(len(features))
                    new_row[0] = next_pred
                    last_seq = np.vstack([last_seq[1:], new_row])
                
                # 6. Inverse Transform Predictions
                dummy = np.zeros((len(future_preds), len(features)))
                dummy[:,0] = future_preds
                future_preds = scaler.inverse_transform(dummy)[:,0]
                
                # 7. Create Forecast DataFrame
                future_dates = pd.bdate_range(start=df.index[-1] + timedelta(days=1), periods=pred_days)
                forecast_df = pd.DataFrame({
                    "Date": future_dates,
                    "Predicted Close": future_preds
                })
                
                # 8. Generate Signals
                signal, reasons = generate_signals(df, forecast_df)
                
                # 9. Display Results
                st.success("🎯 Forecast Complete!")
                
                # Price Chart
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
                
                # Trading Signal
                if signal == "BUY":
                    st.success(f"✅ SIGNAL: {signal}")
                elif signal == "SELL":
                    st.error(f"❌ SIGNAL: {signal}")
                else:
                    st.warning(f"🔄 SIGNAL: {signal}")
                
                st.subheader("Reasons:")
                for reason in reasons:
                    st.write(f"- {reason}")
                
                # Key Metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Current Price", f"₹{df['Close'].iloc[-1]:.2f}")
                col2.metric("Predicted Price", f"₹{float(forecast_df['Predicted Close'].iloc[0]):.2f}", 
                           f"{((float(forecast_df['Predicted Close'].iloc[0])/df['Close'].iloc[-1])-1)*100:.2f}%")
                col3.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}")
                
                # Forecast Table
                st.subheader("Forecast Details:")
                st.dataframe(forecast_df.set_index('Date'))
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    run_ai_prediction()
