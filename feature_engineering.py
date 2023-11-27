import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def add_additional_features(df):
    """
    Adds additional technical indicators (RSI and MACD) to the DataFrame.

    Parameters:
    df (pandas.DataFrame): Dataframe containing Bitcoin price data.

    Returns:
    pandas.DataFrame: Dataframe with additional columns for technical indicators.
    """
    df['RSI'] = calculate_rsi(df['Close'])
    df['MACD'] = calculate_macd(df['Close'])
    return df


def calculate_rsi(prices, n=14):
    """
    Calculates the Relative Strength Index (RSI) for a given price series.

    Parameters:
    prices (pandas.Series): Series of prices.
    n (int): Lookback period.

    Returns:
    pandas.Series: Series of RSI values.
    """
    deltas = prices.diff().fillna(0)
    gain = deltas.clip(lower=0)
    loss = -deltas.clip(upper=0)
    avg_gain = gain.rolling(window=n).mean()
    avg_loss = loss.rolling(window=n).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_macd(prices, n_fast=12, n_slow=26):
    """
    Calculates the Moving Average Convergence/Divergence (MACD) for a given price series.

    Parameters:
    prices (pandas.Series): Series of prices.
    n_fast (int): Lookback period for fast EMA.
    n_slow (int): Lookback period for slow EMA.

    Returns:
    pandas.Series: Series of MACD values.
    """
    ema_fast = prices.ewm(span=n_fast, min_periods=n_slow).mean()
    ema_slow = prices.ewm(span=n_slow, min_periods=n_slow).mean()
    return ema_fast - ema_slow


def normalize_features(df):
    """
    Normalizes the numerical features in the DataFrame using MinMaxScaler.

    Parameters:
    df (pandas.DataFrame): Dataframe containing Bitcoin price data.

    Returns:
    pandas.DataFrame: Dataframe with normalized numerical features.
    """
    scaler = MinMaxScaler()
    numeric_columns = df.select_dtypes(include=['number']).columns
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    return df


def main():
    df = pd.read_csv('data/processed_data.csv')

    # Handle 'Date' column
    dates = pd.to_datetime(df.pop('Date')) if 'Date' in df.columns else None

    df = add_additional_features(df)
    df_scaled = normalize_features(df)

    # Reattach 'Date' column and set as index
    if dates is not None:
        df_scaled = df_scaled.set_index(dates)

    df_scaled.to_csv('data/scaled_data.csv', index_label='Date')


if __name__ == '__main__':
    main()
