# config.py

# base columns
OPEN = 'Open'
HIGH = 'High'
LOW = 'Low'
CLOSE = 'Close'
ADJ_CLOSE = 'Adj Close'
VOLUME = 'Volume'

# indicator columns
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


COLUMN_SETS = {
    'to_scale': [
        OPEN, HIGH, LOW, ADJ_CLOSE, VOLUME,
        MA50, MA200, RETURNS, VOLATILITY, MA20,
        UPPER, LOWER, RSI, MACD, RANGE,
        FIB_23_6, FIB_38_2, FIB_50, FIB_61_8
    ],

    'basic': [
        OPEN, HIGH, LOW, CLOSE, ADJ_CLOSE, VOLUME
    ],

    'required': [
        CLOSE
    ]
}
