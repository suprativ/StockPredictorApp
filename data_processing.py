import pandas as pd

# Moving Average
def moving_average(data, window):
    return data['Close'].rolling(window=window).mean()

# Relative Strength Index (RSI)
def rsi(data, window=14):
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Moving Average Convergence Divergence (MACD)
def macd(data, fast=12, slow=26, signal=9):
    fast_ema = data['Close'].ewm(span=fast, adjust=False).mean()
    slow_ema = data['Close'].ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line
