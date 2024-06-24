import numpy as np

# Constants for technical indicators
RSI_WINDOW = 14
MACD_FAST_WINDOW = 12
MACD_SLOW_WINDOW = 26
BOLLINGER_WINDOW = 20
FIBONACCI_WINDOW = 50
MA50_WINDOW = 50
MA200_WINDOW = 200
ATR_WINDOW = 14
CCI_WINDOW = 20
EMA_WINDOW = 12


def calculate_rsi(prices, window=RSI_WINDOW):
    """
    Calculate the Relative Strength Index (RSI) for a given series of prices.

    Parameters:
    prices (pd.Series): Series of prices.
    window (int): Lookback period for RSI calculation.

    Returns:
    pd.Series: RSI values.
    """
    deltas = prices.diff()
    gain = deltas.where(deltas > 0, 0)
    loss = -deltas.where(deltas < 0, 0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(prices, fast_window=MACD_FAST_WINDOW, slow_window=MACD_SLOW_WINDOW):
    """
    Calculate the Moving Average Convergence Divergence (MACD) for a given series of prices.

    Parameters:
    prices (pd.Series): Series of prices.
    fast_window (int): Fast EMA period.
    slow_window (int): Slow EMA period.

    Returns:
    pd.Series: MACD values.
    """
    ema_fast = prices.ewm(span=fast_window, min_periods=slow_window).mean()
    ema_slow = prices.ewm(span=slow_window, min_periods=slow_window).mean()
    macd = ema_fast - ema_slow
    return macd


def calculate_bollinger_bands(df, window=BOLLINGER_WINDOW):
    """
    Calculate the Bollinger Bands for a given DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing the 'Close' column.
    window (int): Lookback period for Bollinger Bands calculation.

    Returns:
    pd.DataFrame: DataFrame with Bollinger Bands columns added.
    """
    df['MA20'] = df['Close'].rolling(window).mean()
    df['Volatility'] = df['Close'].rolling(window).std()
    df['Upper'] = df['MA20'] + 2 * df['Volatility']
    df['Lower'] = df['MA20'] - 2 * df['Volatility']
    return df


def calculate_fibonacci_levels(df, window=FIBONACCI_WINDOW):
    """
    Calculate the Fibonacci levels for a given DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing the 'Close' column.
    window (int): Lookback period for Fibonacci levels calculation.

    Returns:
    pd.DataFrame: DataFrame with Fibonacci levels columns added.
    """
    df['High'] = df['Close'].rolling(window=window).max()
    df['Low'] = df['Close'].rolling(window=window).min()
    df['Range'] = df['High'] - df['Low']

    df['Fibonacci_23.6%'] = df['High'] - 0.236 * df['Range']
    df['Fibonacci_38.2%'] = df['High'] - 0.382 * df['Range']
    df['Fibonacci_50%'] = df['High'] - 0.5 * df['Range']
    df['Fibonacci_61.8%'] = df['High'] - 0.618 * df['Range']

    return df


def calculate_atr(df, window=ATR_WINDOW):
    """
    Calculate the Average True Range (ATR) for a given DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'High', 'Low', and 'Close' columns.
    window (int): Lookback period for ATR calculation.

    Returns:
    pd.Series: ATR values.
    """
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    true_range = high_low.combine(high_close, max).combine(low_close, max)
    atr = true_range.rolling(window=window, min_periods=1).mean()
    return atr


def calculate_cci(df, window=CCI_WINDOW):
    """
    Calculate the Commodity Channel Index (CCI) for a given DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'High', 'Low', and 'Close' columns.
    window (int): Lookback period for CCI calculation.

    Returns:
    pd.Series: CCI values.
    """
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    ma = tp.rolling(window=window).mean()
    md = tp.rolling(window=window).apply(lambda x: (np.fabs(x - x.mean())).mean())
    cci = (tp - ma) / (0.015 * md)
    return cci


def calculate_obv(df):
    """
    Calculate the On-Balance Volume (OBV) for a given DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'Close' and 'Volume' columns.

    Returns:
    pd.Series: OBV values.
    """
    direction = np.sign(df['Close'].diff()).fillna(0)
    obv = (df['Volume'] * direction).cumsum()
    return obv


def calculate_ema(prices, window=EMA_WINDOW):
    """
    Calculate the Exponential Moving Average (EMA) for a given series of prices.

    Parameters:
    prices (pd.Series): Series of prices.
    window (int): EMA period.

    Returns:
    pd.Series: EMA values.
    """
    ema = prices.ewm(span=window, adjust=False).mean()
    return ema


def calculate_technical_indicators(df):
    """
    Calculate various technical indicators and add them to the DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing the necessary columns.

    Returns:
    pd.DataFrame: DataFrame with technical indicators added.
    """
    required_columns = ['Close', 'High', 'Low', 'Volume']
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
    df['ATR'] = calculate_atr(df)
    df['CCI'] = calculate_cci(df)
    df['OBV'] = calculate_obv(df)
    df['EMA'] = calculate_ema(df['Close'])

    df['Trend'] = (df['MA50'] < df['Close']).astype(int)

    df.ffill(inplace=True)
    df.bfill(inplace=True)

    return df
