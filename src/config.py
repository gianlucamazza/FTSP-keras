"""
config.py

Configuration file containing constants and parameters for the LSTM and ARIMA model training and prediction.

Sections:
1. Column Definitions
2. Column Sets
3. Model Parameters
4. Hyperparameters
5. ARIMA Parameters
"""

# Column Definitions
# -------------------
# Basic columns in the dataset
OPEN = 'Open'
HIGH = 'High'
LOW = 'Low'
CLOSE = 'Close'
VOLUME = 'Volume'
TREND = 'Trend'

# Technical indicator columns
MA50 = 'MA50'
MA200 = 'MA200'
RSI = 'RSI'
MACD = 'MACD'
RETURNS = 'Returns'
VOLATILITY = 'Volatility'
MA20 = 'MA20'
UPPER = 'Upper'
LOWER = 'Lower'
RANGE = 'Range'
FIB_23_6 = 'Fibonacci_23.6%'
FIB_38_2 = 'Fibonacci_38.2%'
FIB_50 = 'Fibonacci_50%'
FIB_61_8 = 'Fibonacci_61.8%'
ATR = 'ATR'
CCI = 'CCI'
OBV = 'OBV'
EMA = 'EMA'

# Column Sets
# -----------
# Sets of columns for different purposes
COLUMN_SETS = {
    'to_scale': [
        OPEN, HIGH, LOW, CLOSE, VOLUME,
        MA50, MA200, RETURNS, VOLATILITY, MA20,
        UPPER, LOWER, RSI, MACD, RANGE,
        FIB_23_6, FIB_38_2, FIB_50, FIB_61_8, ATR, CCI, OBV, EMA, TREND
    ],
    'basic': [
        OPEN, HIGH, LOW, CLOSE, VOLUME
    ],
    'required': [
        CLOSE
    ]
}

# Model Parameters
# ----------------
TRAIN_VALIDATION_SPLIT = 0.8

# ARIMA Parameters
# ----------------
# Parameters for ARIMA model configuration
ARIMA_ORDER = (2, 1, 3)  # p, d, q parameters
ARIMA_SEASONAL_ORDER = (1, 0, 1, 12)  # P, D, Q, S parameters
