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
    # Momentum Indicators
    df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
    stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
    df['Stoch_%K'] = stoch.stoch()
    df['Stoch_%D'] = stoch.stoch_signal()
    
    # Trend Indicators
    df['EMA_20'] = EMAIndicator(close=df['Close'], window=20).ema_indicator()
    df['EMA_50'] = EMAIndicator(close=df['Close'], window=50).ema_indicator()
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['ADX'] = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14).adx()
    
    # Volatility Indicators
    bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Lower'] = bb.bollinger_lband()
    
    # Volume Indicators
    df['VWAP'] = VolumeWeightedAveragePrice(high=df['High'], low=df['Low'], 
                                          close=df['Close'], volume=df['Volume'], window=14).volume_weighted_average_price()
    
    return df.dropna()

# --- TRADING SIGNAL GENERATION ---
def generate_signals(df, forecast):
    signals = []
    reasons = []
    
    last_row = df.iloc[-1]
    pred = forecast.iloc[0]['Predicted Close']
    
    # Signal 1: Price vs Prediction
    if pred > last_row['Close'] * 1.02:  # 2% above current
        signals.append("BUY")
        reasons.append("Predicted price is significantly higher than current")
    elif pred < last_row['Close'] * 0.98:  # 2% below current
        signals.append("SELL")
        reasons.append("Predicted price is significantly lower than current")
    else:
        signals.append("HOLD")
        reasons.append("Predicted price is close to current")
    
    # Signal 2: RSI
    if last_row['RSI'] < 30:
        signals.append("BUY")
        reasons.append("RSI indicates oversold condition")
    elif last_row['RSI'] > 70:
        signals.append("SELL")
        reasons.append("RSI indicates overbought condition")
    
    # Signal 3: MACD
    if last_row['MACD'] > last_row['MACD_Signal']:
        signals.append("BUY")
        reasons.append("MACD crossover bullish signal")
    elif last_row['MACD'] < last_row['MACD_Signal']:
        signals.append("SELL")
        reasons.append("MACD crossover bearish signal")
    
    # Signal 4: Bollinger Bands
    if last_row['Close'] < last_row['BB_Lower']:
        signals.append("BUY")
        reasons.append("Price below lower Bollinger Band (potential rebound)")
    elif last_row['Close'] > last_row['BB_Upper']:
        signals.append("SELL")
        reasons.append("Price above upper Bollinger Band (potential pullback)")
    
    # Count signals
    buy_count = signals.count("BUY")
    sell_count = signals.count("SELL")
    
    if buy_count > sell_count:
        final_signal = "BUY"
    elif sell_count > buy_count:
        final_signal = "SELL"
    else:
        final_signal = "HOLD"
    
    return final_signal, reasons

# --- MAIN FUNCTION ---
def run_ai_prediction():
    st.markdown("---")
    st.header("🤖 Advanced AI Stock Prediction")
    st.caption("Enhanced LSTM model with trading signals and explanations")

    with st.expander("🔮 AI Prediction Panel", expanded=True):
        user_stock = st.text_input("Enter NSE Stock Symbol (e.g., INFY)", value="INFY")
        pred_days = st.slider("Prediction Horizon (days)", 5, 15, 7)
        
        if st.button("🚀 Generate Advanced Forecast"):
            ticker = user_stock.upper().strip() + ".NS"
            
            try:
                # Data Collection
                with st.spinner("📥 Fetching enhanced stock data..."):
                    df = yf.download(ticker, period="6mo", interval="1d", progress=False)
                    if df.empty:
                        st.error("❌ No data found for this stock")
                        return
                    
                    df = add_technical_indicators(df)
                
                # Data Preparation
                with st.spinner("🧠 Preparing AI model..."):
                    features = ['Close', 'RSI', 'EMA_20', 'EMA_50', 'MACD', 'MACD_Signal', 'ADX', 'BB_Upper', 'BB_Lower']
                    scaler = MinMaxScaler()
                    scaled_data = scaler.fit_transform(df[features])
                    
                    # Create sequences
                    X, y = prepare_lstm_data(scaled_data)
                    
                    # Enhanced LSTM Model
                    model = Sequential([
                        LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
                        Dropout(0.2),
                        LSTM(64, return_sequences=False),
                        Dropout(0.2),
                        Dense(25),
                        Dense(1)
                    ])
                    
                    model.compile(optimizer='adam', loss='mse')
                    model.fit(X, y, epochs=50, batch_size=32, verbose=0)
                
                # Prediction
                with st.spinner("🔮 Generating forecast..."):
                    last_seq = scaled_data[-30:]
                    future_preds = []
                    
                    for _ in range(pred_days):
                        next_pred = model.predict(last_seq.reshape(1, 30, len(features)))[0, 0]
                        future_preds.append(next_pred)
                        last_seq = np.append(last_seq[1:], [[next_pred] + [0]*(len(features)-1)], axis=0)
                    
                    # Inverse transform just the close price predictions
                    dummy = np.zeros((len(future_preds), len(features)))
                    dummy[:, 0] = future_preds
                    future_preds = scaler.inverse_transform(dummy)[:, 0]
                    
                    future_dates = pd.bdate_range(start=df.index[-1] + timedelta(days=1), periods=pred_days)
                    forecast_df = pd.DataFrame({"Date": future_dates, "Predicted Close": future_preds})
                
                # Generate Signals
                final_signal, reasons = generate_signals(df, forecast_df)
                
                # Display Results
                st.success("✅ Advanced Forecast Complete!")
                
                # Create tabs
                tab1, tab2, tab3 = st.tabs(["📈 Forecast", "📊 Technicals", "📢 Trading Signal"])
                
                with tab1:
                    # Candlestick Chart with Forecast
                    fig = go.Figure()
                    
                    # Candlestick
                    fig.add_trace(go.Candlestick(
                        x=df.index,
                        open=df['Open'],
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'],
                        name='Price'
                    ))
                    
                    # Forecast
                    fig.add_trace(go.Scatter(
                        x=forecast_df['Date'],
                        y=forecast_df['Predicted Close'],
                        line=dict(color='green', width=2, dash='dot'),
                        name='AI Forecast',
                        mode='lines+markers'
                    ))
                    
                    fig.update_layout(
                        title=f"{user_stock} Price & {pred_days}-Day Forecast",
                        xaxis_rangeslider_visible=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(forecast_df, use_container_width=True)
                
                with tab2:
                    # Technical Indicators Visualization
                    fig1 = go.Figure()
                    fig1.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'))
                    fig1.add_hline(y=30, line_dash="dot", line_color="green")
                    fig1.add_hline(y=70, line_dash="dot", line_color="red")
                    fig1.update_layout(title="RSI Indicator")
                    
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD'))
                    fig2.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal'))
                    fig2.update_layout(title="MACD Indicator")
                    
                    st.plotly_chart(fig1, use_container_width=True)
                    st.plotly_chart(fig2, use_container_width=True)
                
                with tab3:
                    # Trading Signal Card
                    if final_signal == "BUY":
                        st.success(f"🚀 SIGNAL: {final_signal}")
                    elif final_signal == "SELL":
                        st.error(f"⚠️ SIGNAL: {final_signal}")
                    else:
                        st.warning(f"🔄 SIGNAL: {final_signal}")
                    
                    st.subheader("Reasons:")
                    for reason in set(reasons):  # Remove duplicates
                        st.write(f"- {reason}")
                    
                    st.subheader("Key Metrics:")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Current Price", f"₹{df['Close'].iloc[-1]:.2f}")
                    col2.metric("Predicted Price", f"₹{forecast_df['Predicted Close'].iloc[0]:.2f}", 
                               f"{(forecast_df['Predicted Close'].iloc[0]/df['Close'].iloc[-1]-1)*100:.2f}%")
                    col3.metric("RSI", f"{df['RSI'].iloc[-1]:.2f}")
                    
                    st.metric("MACD Signal", 
                            "Bullish" if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1] else "Bearish")
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    run_ai_prediction()
