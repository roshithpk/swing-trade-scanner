import pandas as pd

# --- RSI (Relative Strength Index) ---
def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- Trend Determination ---
def determine_trend(close_prices, ema_window=20):
    ema = close_prices.rolling(window=ema_window).mean()
    return close_prices.iloc[-1] > ema.iloc[-1]  # True = Uptrend

# --- Entry/Exit Points ---
def suggest_entry_exit(close_prices, buffer_pct=0.02):
    latest_price = close_prices.iloc[-1]
    entry = latest_price * (1 - buffer_pct)
    exit_ = latest_price * (1 + buffer_pct)
    return round(entry, 2), round(exit_, 2)

