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
        # Check if we have enough data (minimum 30 days for most indicators)
        if len(df) < 30:
            st.error(f"Need at least 30 data points, only have {len(df)}")
            return None
        
        # Ensure we have all required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            return None
        
        # Convert to pandas Series (some TA libraries need this)
        close = pd.Series(df['Close'], index=df.index)
        high = pd.Series(df['High'], index=df.index)
        low = pd.Series(df['Low'], index=df.index)
        volume = pd.Series(df['Volume'], index=df.index)

        # Calculate indicators one by one with error handling
        try:
            df['RSI'] = RSIIndicator(close=close, window=14).rsi()
        except Exception as e:
            st.error(f"RSI calculation failed: {str(e)}")
            df['RSI'] = np.nan

        try:
            stoch = StochasticOscillator(high=high, low=low, close=close, window=14)
            df['Stoch_%K'] = stoch.stoch()
            df['Stoch_%D'] = stoch.stoch_signal()
        except Exception as e:
            st.error(f"Stochastic calculation failed: {str(e)}")
            df['Stoch_%K'] = np.nan
            df['Stoch_%D'] = np.nan

        try:
            df['EMA_20'] = EMAIndicator(close=close, window=20).ema_indicator()
            df['EMA_50'] = EMAIndicator(close=close, window=50).ema_indicator()
        except Exception as e:
            st.error(f"EMA calculation failed: {str(e)}")
            df['EMA_20'] = np.nan
            df['EMA_50'] = np.nan

        try:
            macd = MACD(close=close)
            df['MACD'] = macd.macd()
            df['MACD_Signal'] = macd.macd_signal()
            df['MACD_Hist'] = macd.macd_diff()
        except Exception as e:
            st.error(f"MACD calculation failed: {str(e)}")
            df['MACD'] = np.nan
            df['MACD_Signal'] = np.nan
            df['MACD_Hist'] = np.nan

        try:
            bb = BollingerBands(close=close, window=20, window_dev=2)
            df['BB_Upper'] = bb.bollinger_hband()
            df['BB_Middle'] = bb.bollinger_mavg()
            df['BB_Lower'] = bb.bollinger_lband()
        except Exception as e:
            st.error(f"Bollinger Bands calculation failed: {str(e)}")
            df['BB_Upper'] = np.nan
            df['BB_Middle'] = np.nan
            df['BB_Lower'] = np.nan

        try:
            df['VWAP'] = VolumeWeightedAveragePrice(
                high=high, low=low, close=close, volume=volume, window=14
            ).volume_weighted_average_price()
        except Exception as e:
            st.error(f"VWAP calculation failed: {str(e)}")
            df['VWAP'] = np.nan

        # Forward fill then backfill any remaining NAs
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        
        # Verify we have at least the basic indicators
        required_indicators = ['RSI', 'EMA_20', 'MACD', 'BB_Upper', 'BB_Lower']
        if not all(ind in df.columns for ind in required_indicators):
            st.error("Failed to calculate some essential indicators")
            return None
            
        return df
    
    except Exception as e:
        st.error(f"Technical indicators failed completely: {str(e)}")
        return None

# --- [Rest of your code remains the same] ---
