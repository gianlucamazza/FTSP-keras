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
    'neurons': 150,            # Number of neurons in each LSTM layer
    'dropout': 0.3,            # Dropout rate for regularization
    'additional_layers': 1,    # Number of additional dense layers after LSTM layers
    'bidirectional': True,     # Use bidirectional LSTM if True
    'epochs': 100,             # Number of training epochs
    'batch_size': 16,          # Batch size for training
    'train_steps': 90,         # Number of steps (time frames) for each training sample
    'test_steps': 30,          # Number of steps (time frames) for each test sample
    'n_folds': 5,              # Number of folds for cross-validation
    'early_stopping_patience': 10 # Patience for early stopping during training
}
