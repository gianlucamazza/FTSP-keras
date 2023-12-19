# technical_indicators.py

RSI_WINDOW = 14
MACD_FAST_WINDOW = 12
MACD_SLOW_WINDOW = 26


def calculate_rsi(prices, n=RSI_WINDOW):
    deltas = prices.diff()
    loss = deltas.where(deltas < 0, 0)
    gain = deltas.where(deltas > 0, 0)
    avg_gain = gain.rolling(window=n, min_periods=1).mean()
    avg_loss = -loss.rolling(window=n, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def calculate_macd(prices, n_fast=MACD_FAST_WINDOW, n_slow=MACD_SLOW_WINDOW):
    ema_fast = prices.ewm(span=n_fast, min_periods=n_slow).mean()
    ema_slow = prices.ewm(span=n_slow, min_periods=n_slow).mean()
    return ema_fast - ema_slow


def calculate_bollinger_bands(df):
    df['MA20'] = df['Close'].rolling(20).mean()
    df['Volatility'] = df['Returns'].rolling(20).std()
    df['Upper'] = df['MA20'] + 2 * df['Volatility']
    df['Lower'] = df['MA20'] - 2 * df['Volatility']
    return df


def calculate_fibonacci_levels(df, window):
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

    df['MA50'] = df['Close'].rolling(50).mean()
    df['MA200'] = df['Close'].rolling(200).mean()
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(50).std()
    df = calculate_bollinger_bands(df)
    df['RSI'] = calculate_rsi(df['Close'])
    df['MACD'] = calculate_macd(df['Close'])
    df = calculate_fibonacci_levels(df, window=50)

    return df
