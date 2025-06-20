# stock_analyzer.py

import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands


# --- Main analysis function ---
def analyze_stock(ticker: str, rsi_bounds=(40, 70)):
    try:
        df = yf.download(ticker, period="6mo", interval="1d", progress=False)
        if df.empty or len(df) < 30:
            return {"error": "Not enough data"}

        df.dropna(inplace=True)

        # Calculate technical indicators
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']

        rsi = RSIIndicator(close=close).rsi()
        ema20 = EMAIndicator(close=close, window=20).ema_indicator()
        macd = MACD(close=close).macd()
        macd_signal = MACD(close=close).macd_signal()
        bb = BollingerBands(close=close)
        atr = AverageTrueRange(high=high, low=low, close=close).average_true_range()

        # Get latest values
        latest = {
            "price": close.iloc[-1],
            "rsi": rsi.iloc[-1],
            "ema20": ema20.iloc[-1],
            "macd": macd.iloc[-1],
            "macd_signal": macd_signal.iloc[-1],
            "bb_upper": bb.bollinger_hband().iloc[-1],
            "bb_lower": bb.bollinger_lband().iloc[-1],
            "atr": atr.iloc[-1],
        }

        # Determine trend
        trend = "🟢 Uptrend" if latest['price'] > latest['ema20'] else "🔴 Downtrend"

        # Generate signal logic
        signal = "HOLD"
        reasons = []

        if latest['rsi'] < 30:
            reasons.append("RSI below 30 (Oversold)")
        elif latest['rsi'] > 70:
            reasons.append("RSI above 70 (Overbought)")

        if latest['macd'] > latest['macd_signal']:
            reasons.append("MACD crossover bullish")
        else:
            reasons.append("MACD crossover bearish")

        if latest['price'] < latest['bb_lower']:
            reasons.append("Price below Bollinger Band (Support zone)")
        elif latest['price'] > latest['bb_upper']:
            reasons.append("Price above Bollinger Band (Resistance zone)")

        # Suggest signal
        if latest['rsi'] < 40 and latest['macd'] > latest['macd_signal']:
            signal = "BUY"
        elif latest['rsi'] > 70 and latest['macd'] < latest['macd_signal']:
            signal = "SELL"

        # Confidence score
        confidence = 0
        if signal == "BUY":
            if latest['price'] < latest['bb_lower']:
                confidence += 25
            if latest['macd'] > latest['macd_signal']:
                confidence += 25
            if latest['rsi'] < 40:
                confidence += 25
            if latest['price'] > latest['ema20']:
                confidence += 25
        elif signal == "SELL":
            if latest['price'] > latest['bb_upper']:
                confidence += 25
            if latest['macd'] < latest['macd_signal']:
                confidence += 25
            if latest['rsi'] > 70:
                confidence += 25
            if latest['price'] < latest['ema20']:
                confidence += 25

        # Risk analysis
        risk_pct = (latest['atr'] / latest['price']) * 100
        if risk_pct < 2:
            risk_level = "🟢 Low"
        elif risk_pct < 4:
            risk_level = "🟡 Medium"
        else:
            risk_level = "🔴 High"

        stop_loss = round(latest['price'] - latest['atr'], 2)
        entry = round(latest['price'], 2)
        exit_price = round(latest['price'] * 1.04, 2) if signal == "BUY" else round(latest['price'] * 0.96, 2)

        return {
            "ticker": ticker.replace(".NS", ""),
            "current_price": f"₹{entry}",
            "signal": signal,
            "confidence": f"{confidence}%",
            "trend": trend,
            "rsi": f"{latest['rsi']:.1f}",
            "stop_loss": f"₹{stop_loss}",
            "entry": f"₹{entry}",
            "exit": f"₹{exit_price}",
            "risk": risk_level,
            "reasons": reasons
        }

    except Exception as e:
        return {"error": str(e)}

