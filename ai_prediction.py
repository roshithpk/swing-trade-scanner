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
        # Make sure we have enough data
        if len(df) < 20:  # Minimum window size for indicators
            st.warning("Not enough data points for technical indicators")
            return df
        
        # Convert to pandas Series if they aren't already
        close = pd.Series(df['Close'])
        high = pd.Series(df['High'])
        low = pd.Series(df['Low'])
        volume = pd.Series(df['Volume'])

        # Momentum Indicators
        df['RSI'] = RSIIndicator(close=close, window=14).rsi()
        
        # Stochastic Oscillator
        stoch = StochasticOscillator(
            high=high,
            low=low,
            close=close,
            window=14,
            smooth_window=3
        )
        df['Stoch_%K'] = stoch.stoch()
        df['Stoch_%D'] = stoch.stoch_signal()
        
        # Trend Indicators
        df['EMA_20'] = EMAIndicator(close=close, window=20).ema_indicator()
        df['EMA_50'] = EMAIndicator(close=close, window=50).ema_indicator()
        
        # MACD
        macd = MACD(close=close)
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist'] = macd.macd_diff()
        
        # ADX
        df['ADX'] = ADXIndicator(
            high=high,
            low=low,
            close=close,
            window=14
        ).adx()
        
        # Bollinger Bands
        bb = BollingerBands(
            close=close,
            window=20,
            window_dev=2
        )
        df['BB_Upper'] = bb.bollinger_hband()
        df['BB_Middle'] = bb.bollinger_mavg()
        df['BB_Lower'] = bb.bollinger_lband()
        
        # Volume Indicators
        df['VWAP'] = VolumeWeightedAveragePrice(
            high=high,
            low=low,
            close=close,
            volume=volume,
            window=14
        ).volume_weighted_average_price()
        
        # Fill any NA values that might have been created
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        
        return df
    
    except Exception as e:
        st.error(f"Error calculating technical indicators: {str(e)}")
        # Return the original DF if indicators fail
        return df

# --- TRADING SIGNALS ---
def generate_signals(df, forecast):
    try:
        signals = []
        reasons = []
        last_row = df.iloc[-1]
        pred = forecast.iloc[0]['Predicted Close']
        
        # 1. Price vs Prediction Signal
        price_diff = (pred - last_row['Close']) / last_row['Close']
        if price_diff > 0.02:
            signals.append("BUY")
            reasons.append(f"Predicted price {price_diff:.2%} higher than current")
        elif price_diff < -0.02:
            signals.append("SELL")
            reasons.append(f"Predicted price {abs(price_diff):.2%} lower than current")
        else:
            signals.append("HOLD")
            reasons.append("Predicted price within 2% of current")
        
        # 2. RSI Signal
        if 'RSI' in df.columns:
            if last_row['RSI'] < 30:
                signals.append("BUY")
                reasons.append(f"RSI {last_row['RSI']:.1f} (oversold)")
            elif last_row['RSI'] > 70:
                signals.append("SELL")
                reasons.append(f"RSI {last_row['RSI']:.1f} (overbought)")
        
        # 3. MACD Signal
        if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
            if last_row['MACD'] > last_row['MACD_Signal']:
                signals.append("BUY")
                reasons.append("MACD above signal line (bullish)")
            else:
                signals.append("SELL")
                reasons.append("MACD below signal line (bearish)")
        
        # 4. Bollinger Bands Signal
        if 'BB_Lower' in df.columns and 'BB_Upper' in df.columns:
            if last_row['Close'] < last_row['BB_Lower']:
                signals.append("BUY")
                reasons.append("Price below lower Bollinger Band")
            elif last_row['Close'] > last_row['BB_Upper']:
                signals.append("SELL")
                reasons.append("Price above upper Bollinger Band")
        
        # Determine final signal
        buy_count = signals.count("BUY")
        sell_count = signals.count("SELL")
        
        if buy_count > sell_count:
            final_signal = "BUY"
        elif sell_count > buy_count:
            final_signal = "SELL"
        else:
            final_signal = "HOLD"
        
        return final_signal, list(set(reasons))  # Remove duplicate reasons
    
    except Exception as e:
        st.error(f"Error generating signals: {str(e)}")
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
        
        with st.spinner("Fetching data..."):
            try:
                # 1. Data Collection
                df = yf.download(ticker, period="6mo", interval="1d", progress=False)
                if df.empty:
                    st.error("No data found for this stock")
                    return
                
                st.write(f"📊 Retrieved {len(df)} days of data")
                
                # 2. Add Technical Indicators
                df = add_technical_indicators(df)
                if df.empty:
                    st.error("Failed to calculate technical indicators")
                    return
                
                # 3. Verify we have all required columns
                required_indicators = ['RSI', 'EMA_20', 'MACD', 'BB_Upper', 'BB_Lower']
                missing = [ind for ind in required_indicators if ind not in df.columns]
                if missing:
                    st.error(f"Missing indicators: {', '.join(missing)}")
                    return
                
                # 4. Prepare data for LSTM
                features = ['Close', 'RSI', 'EMA_20', 'MACD', 'BB_Upper', 'BB_Lower']
                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(df[features])
                
                X, y = prepare_lstm_data(scaled_data)
                X = X.reshape((X.shape[0], X.shape[1], len(features)))
                
                # 5. Build and train LSTM model
                model = Sequential([
                    LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
                    Dropout(0.2),
                    LSTM(64),
                    Dropout(0.2),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mse')
                model.fit(X, y, epochs=50, batch_size=32, verbose=0)
                
                # 6. Generate predictions
                last_seq = scaled_data[-30:]
                future_preds = []
                for _ in range(pred_days):
                    next_pred = model.predict(last_seq.reshape(1, 30, len(features)), verbose=0)[0,0]
                    future_preds.append(next_pred)
                    # Maintain sequence shape with new prediction
                    new_row = np.zeros(len(features))
                    new_row[0] = next_pred  # Only update Close price prediction
                    last_seq = np.vstack([last_seq[1:], new_row])
                
                # 7. Inverse transform predictions
                dummy = np.zeros((len(future_preds), len(features)))
                dummy[:,0] = future_preds
                future_preds = scaler.inverse_transform(dummy)[:,0]
                
                # 8. Create forecast DataFrame
                future_dates = pd.bdate_range(start=df.index[-1] + timedelta(days=1), periods=pred_days)
                forecast_df = pd.DataFrame({
                    "Date": future_dates,
                    "Predicted Close": future_preds
                })
                
                # 9. Generate trading signals
                signal, reasons = generate_signals(df, forecast_df)
                
                # 10. Display results
                st.success("🎯 Forecast Complete!")
                
                # Price chart with forecast
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
                    title=f"{user_stock} Price & {pred_days}-Day Forecast",
                    xaxis_rangeslider_visible=False
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Trading signal
                if signal == "BUY":
                    st.success(f"✅ SIGNAL: {signal}")
                elif signal == "SELL":
                    st.error(f"❌ SIGNAL: {signal}")
                else:
                    st.warning(f"🔄 SIGNAL: {signal}")
                
                st.subheader("Reasons:")
                for reason in reasons:
                    st.write(f"- {reason}")
                
                # Key metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Current Price", f"₹{df['Close'].iloc[-1]:.2f}")
                col2.metric("Predicted Price", f"₹{forecast_df['Predicted Close'].iloc[0]:.2f}", 
                           f"{((forecast_df['Predicted Close'].iloc[0]/df['Close'].iloc[-1]-1)*100:.2f}%")
                col3.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}")
                
                # Show forecast table
                st.subheader("Forecast Details:")
                st.dataframe(forecast_df.set_index('Date'))
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    run_ai_prediction()
