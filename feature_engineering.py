# feature_engineering.py
import argparse

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os


def add_additional_features(df):
    """
    Adds additional features to the given dataframe.

    Parameters:
    df (pandas.DataFrame): Dataframe containing Bitcoin price data.

    Returns:
    pandas.DataFrame: Dataframe with additional features.
    """
    df['RSI'] = calculate_rsi(df['Close'])
    df['MACD'] = calculate_macd(df['Close'])
    return df.ffill()


def calculate_rsi(prices, n=14):
    """
    Calculates the Relative Strength Index (RSI) for the given prices.

    Parameters:
    prices (pandas.Series): Series containing the prices.
    n (int): Number of time steps to look back.

    Returns:
    pandas.Series: Series containing the RSI values.
    """
    deltas = prices.diff()
    loss = deltas.where(deltas < 0, 0)
    gain = deltas.where(deltas > 0, 0)
    avg_gain = gain.rolling(window=n, min_periods=1).mean()
    avg_loss = -loss.rolling(window=n, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def calculate_macd(prices, n_fast=12, n_slow=26):
    """
    Calculates the Moving Average Convergence Divergence (MACD) for the given prices.

    Parameters:
    prices (pandas.Series): Series containing the prices.
    n_fast (int): Number of time steps to use for the fast EMA.
    n_slow (int): Number of time steps to use for the slow EMA.

    Returns:
    pandas.Series: Series containing the MACD values.
    """
    ema_fast = prices.ewm(span=n_fast, min_periods=n_slow).mean()
    ema_slow = prices.ewm(span=n_slow, min_periods=n_slow).mean()
    return ema_fast - ema_slow


def normalize_features(df):
    """
    Normalizes the values of the given columns using the MinMaxScaler.

    Parameters:
    df (pandas.DataFrame): Dataframe containing Bitcoin price data.

    Returns:
    pandas.DataFrame: Dataframe with normalized values.
    MinMaxScaler: The fitted scaler instance.
    """
    scaler = MinMaxScaler()
    numeric_columns = df.select_dtypes(include=['number']).columns
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    return df, scaler


def save_scaler(scaler, path='models/scaler.pkl'):
    """
    Saves the MinMaxScaler to a file.

    Parameters:
    scaler (MinMaxScaler): The fitted scaler instance.
    path (str): File path to save the scaler.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)


def main(data, output):
    df = pd.read_csv(data, index_col='Date')
    df = add_additional_features(df)
    df_scaled, scaler = normalize_features(df)
    save_scaler(scaler)  # Save the scaler after normalization
    df_scaled.to_csv(output, index=True)
    print("Data scaled and saved successfully.")


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/processed_data.csv',
                        help='Path to the CSV file containing the data.')
    parser.add_argument('--output', type=str, default='data/scaled_data.csv',
                        help='Path to save the scaled data.')
    args = parser.parse_args()

    main(args.data, args.output)
