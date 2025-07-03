import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from st_aggrid import AgGrid, GridOptionsBuilder
import torch
import torch.nn as nn
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.volatility import AverageTrueRange
import math

# --- Helper function to skip weekends ---
def next_business_day(date):
    """Get the next business day (skips weekends)"""
    next_day = date + timedelta(days=1)
    while next_day.weekday() >= 5:  # 5=Saturday, 6=Sunday
        next_day += timedelta(days=1)
    return next_day

# --- Positional Encoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

# --- Transformer Model ---
class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, dropout=0.1, max_len=500):
        super().__init__()
        self.input_linear = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_linear = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()  # Constrain outputs to 0-1 range
        )

    def forward(self, src):
        src = self.input_linear(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.output_linear(output[:, -1, :])
        return output

# --- Feature Engineering ---
def add_indicators(df):
    df = df.copy()
    if len(df) >= 14:  # Minimum required for RSI
        df['RSI'] = RSIIndicator(close=df['Close'].squeeze(), window=14).rsi()
    if len(df) >= 20:  # Minimum required for EMA20
        df['EMA20'] = EMAIndicator(close=df['Close'].squeeze(), window=20).ema_indicator()
    df['MACD'] = MACD(close=df['Close'].squeeze()).macd()
    df['ADX'] = ADXIndicator(
        high=df['High'].squeeze(),
        low=df['Low'].squeeze(),
        close=df['Close'].squeeze()
    ).adx()
    df['ATR'] = AverageTrueRange(
        high=df['High'].squeeze(),
        low=df['Low'].squeeze(),
        close=df['Close'].squeeze()
    ).average_true_range()
    df.dropna(inplace=True)
    return df

# --- Sequence Creation ---
def create_sequences(data, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        x = data[i:i + seq_len]
        y = data[i + seq_len, 0]  # Predict Close only
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys).flatten()

# --- Main Function ---
def run_ai_prediction():
    st.title("📈 Transformer-based Stock Forecast with Technical Indicators")

    with st.expander("⚙️ Settings", expanded=True):
        col1, col2 = st.columns(2)
        user_stock = col1.text_input("Stock Symbol (e.g., INFY)", value="INFY")
        pred_days = col2.slider("Forecast Days", 5, 15, 7)

    if st.button("🚀 Predict with Transformer"):
        ticker = f"{user_stock.upper().strip()}.NS"
        try:
            df = yf.download(ticker, period="2y", interval="1d", progress=False)
            if df.empty:
                st.error("No data found for this stock")
                return

            # Ensure we only have business days in historical data
            df = df[df.index.dayofweek < 5]  # 0-4 = Monday-Friday

            df = add_indicators(df)
            features = ['Close', 'RSI', 'EMA20', 'MACD', 'ADX', 'ATR']
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(df[features])

            seq_len = 30
            X, y = create_sequences(scaled, seq_len)
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32)

            model = TransformerModel(
                input_size=len(features),
                d_model=64,
                nhead=4,
                num_layers=2,
                dropout=0.2
            )
            
            loss_fn = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            # --- Training with Progress Bar ---
            model.train()
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for epoch in range(50):
                optimizer.zero_grad()
                output = model(X_tensor)
                loss = loss_fn(output.view(-1), y_tensor)
                loss.backward()
                optimizer.step()
            
                percent_complete = int(((epoch + 1) / 50) * 100)
                progress_bar.progress(percent_complete)
                status_text.text(f"Training progress: {percent_complete}% (Epoch {epoch+1}/50)")
            
            status_text.text("✅ Training completed!")

            # --- Prediction with Proper Sequence Updating ---
          
# --- Enhanced Prediction Loop ---
            # --- In your prediction loop ---
            model.eval()
            preds = []
            pred_dates = []
            current_sequence = X_tensor[-1:].clone()  # Start with last known sequence
            last_known = df.copy()
            current_date = last_known.index[-1]
            
            # Get business days to predict
            business_days_to_predict = []
            temp_date = current_date
            for _ in range(pred_days):
                temp_date = next_business_day(temp_date)
                business_days_to_predict.append(temp_date)
            
            for i, target_date in enumerate(business_days_to_predict):
                with torch.no_grad():
                    # Get scaled prediction (0-1 range)
                    pred_scaled = model(current_sequence).item()
                
                # 1. Generate realistic OHLC values for the prediction
                last_close = last_known['Close'].iloc[-1]
                pred_close = scaler.inverse_transform([[pred_scaled] + [0]*(len(features)-1)])[0][0]
                
                # Calculate realistic High/Low based on predicted Close
                pred_high = pred_close * 1.005  # Typically 0.5% higher than close
                pred_low = pred_close * 0.995   # Typically 0.5% lower than close
                pred_open = (last_close + pred_close) / 2  # Open between last close and current pred
                
                # 2. Create complete OHLC row
                new_row = {
                    'Open': pred_open,
                    'High': pred_high,
                    'Low': pred_low,
                    'Close': pred_close,
                    'Volume': last_known['Volume'].iloc[-1] * 0.95  # Slight volume decrease
                }
                
                # 3. Add to dataframe (preserve all columns)
                new_df = pd.DataFrame([new_row], index=[target_date])
                for col in last_known.columns:
                    if col not in new_row:
                        new_df[col] = last_known[col].iloc[-1]  # Carry forward other values
                
                last_known = pd.concat([last_known, new_df])
                
                # 4. Now recompute indicators with proper OHLC data
                last_known = add_indicators(last_known)
                
                # 5. Get the ACTUAL new scaled values with proper indicators
                new_scaled = scaler.transform(last_known[features].iloc[-1:])[0]
                
                # 6. Update sequence properly
                current_sequence_np = current_sequence.numpy()[0]
                new_sequence_np = np.vstack([current_sequence_np[1:], new_scaled])
                current_sequence = torch.tensor(new_sequence_np[np.newaxis], dtype=torch.float32)
                
                # Store prediction
                preds.append(pred_close)
                pred_dates.append(target_date)
                
                # Debug output
                st.write(f"📅 {target_date.date()}: "
                         f"O:{pred_open:.2f} H:{pred_high:.2f} L:{pred_low:.2f} C:{pred_close:.2f}")
                st.write(f"📈 New RSI: {last_known['RSI'].iloc[-1]:.2f}, "
                         f"EMA20: {last_known['EMA20'].iloc[-1]:.2f}")
            # --- Continue with existing code ---
            forecast_df = pd.DataFrame({
                "Date": pred_dates,
                "Predicted Close": preds
            })

            # --- Model Signal ---
            current_price = float(df['Close'].iloc[-1])
            predicted_price = float(forecast_df['Predicted Close'].iloc[0])
            pct_diff = ((predicted_price - current_price) / current_price) * 100

            if pct_diff >= 2:
                signal = "BUY"
                reason = f"📈 Forecasted to rise by {pct_diff:.2f}%"
            elif pct_diff <= -2:
                signal = "SELL"
                reason = f"📉 Forecasted to fall by {pct_diff:.2f}%"
            else:
                signal = "HOLD"
                reason = f"🔄 Minimal change expected ({pct_diff:.2f}%)"

            st.markdown("### 🧠 Model Signal")
            if signal == "BUY":
                st.success(f"✅ SIGNAL: **{signal}**  \n**Reason:** {reason}")
            elif signal == "SELL":
                st.error(f"❌ SIGNAL: **{signal}**  \n**Reason:** {reason}")
            else:
                st.warning(f"🔄 SIGNAL: **{signal}**  \n**Reason:** {reason}")

            # --- Metrics ---
            col1, col2 = st.columns(2)
            col1.metric("Current Price", f"₹{current_price:.2f}")
            col2.metric("Predicted Price", f"₹{predicted_price:.2f}", f"{pct_diff:.2f}%")

            # --- Forecast Table ---
            gb = GridOptionsBuilder.from_dataframe(forecast_df)
            gb.configure_default_column(resizable=True, wrapText=True)
            grid_options = gb.build()
            AgGrid(forecast_df, gridOptions=grid_options, theme="balham", height=350)

            # --- Plot Chart ---
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df.index[-60:], 
                y=df['Close'].iloc[-60:], 
                mode='lines', 
                name='Historical'
            ))
            fig.add_trace(go.Scatter(
                x=forecast_df['Date'], 
                y=forecast_df['Predicted Close'], 
                mode='lines+markers', 
                name='Predicted'
            ))
            fig.update_layout(
                title=f"{user_stock.upper()} Forecast (Business Days Only)",
                xaxis_title="Date",
                yaxis_title="Close Price"
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")

if __name__ == "__main__":
    run_ai_prediction()
