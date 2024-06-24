import pandas as pd

# Constants for technical indicators
RSI_WINDOW = 14
MACD_FAST_WINDOW = 12
MACD_SLOW_WINDOW = 26
BOLLINGER_WINDOW = 20
FIBONACCI_WINDOW = 50
MA50_WINDOW = 50
MA200_WINDOW = 200

def calculate_rsi(prices, n=RSI_WINDOW):
    deltas = prices.diff()
    gain = deltas.where(deltas > 0, 0)
    loss = -deltas.where(deltas < 0, 0)
    avg_gain = gain.rolling(window=n, min_periods=1).mean()
    avg_loss = loss.rolling(window=n, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, n_fast=MACD_FAST_WINDOW, n_slow=MACD_SLOW_WINDOW):
    ema_fast = prices.ewm(span=n_fast, min_periods=n_slow).mean()
    ema_slow = prices.ewm(span=n_slow, min_periods=n_slow).mean()
    macd = ema_fast - ema_slow
    return macd

def calculate_bollinger_bands(df, n=BOLLINGER_WINDOW):
    df['MA20'] = df['Close'].rolling(n).mean()
    df['Volatility'] = df['Close'].rolling(n).std()
    df['Upper'] = df['MA20'] + 2 * df['Volatility']
    df['Lower'] = df['MA20'] - 2 * df['Volatility']
    return df


def calculate_fibonacci_levels(df, window=FIBONACCI_WINDOW):
    df['High'] = df['Close'].rolling(window=window).max()
    df['Low'] = df['Close'].rolling(window=window).min()
    df['Range'] = df['High'] - df['Low']

    df['Fibonacci_23.6%'] = df['High'] - 0.236 * df['Range']
    df['Fibonacci_38.2%'] = df['High'] - 0.382 * df['Range']
    df['Fibonacci_50%'] = df['High'] - 0.5 * df['Range']
    df['Fibonacci_61.8%'] = df['High'] - 0.618 * df['Range']

    return df


def calculate_technical_indicators(df):
    required_columns = ['Close']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain the following columns: {required_columns}")

    df['MA50'] = df['Close'].rolling(MA50_WINDOW).mean()
    df['MA200'] = df['Close'].rolling(MA200_WINDOW).mean()
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(BOLLINGER_WINDOW).std()

    df = calculate_bollinger_bands(df)
    df['RSI'] = calculate_rsi(df['Close'])
    df['MACD'] = calculate_macd(df['Close'])
    df = calculate_fibonacci_levels(df)

    df['Trend'] = (df['MA50'] < df['Close']).astype(int)

    df.ffill(inplace=True)
    df.bfill(inplace=True)

    return df
