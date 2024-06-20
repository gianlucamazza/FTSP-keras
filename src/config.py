"""
config.py

Configuration file containing constants and parameters for the LSTM model training and prediction.

Sections:
1. Column Definitions
2. Column Sets
3. Model Parameters
"""

# Column Definitions
# -------------------
# Basic columns in the dataset
OPEN = 'Open'
HIGH = 'High'
LOW = 'Low'
CLOSE = 'Close'
VOLUME = 'Volume'

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

# Column Sets
# -----------
# Sets of columns for different purposes
COLUMN_SETS = {
    'to_scale': [
        OPEN, HIGH, LOW, CLOSE, VOLUME,
        MA50, MA200, RETURNS, VOLATILITY, MA20,
        UPPER, LOWER, RSI, MACD, RANGE,
        FIB_23_6, FIB_38_2, FIB_50, FIB_61_8
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
# Parameters for the LSTM model
PARAMETERS = {
    'neurons': 100,
    'dropout': 0.3,
    'additional_layers': 2,
    'bidirectional': True,
    'epochs': 50,
    'batch_size': 32,
    'train_steps': 60,
    'test_steps': 30
}
