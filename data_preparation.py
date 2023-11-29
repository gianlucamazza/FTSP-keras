# data_preparation.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib
import os


def load_data(file_path):
    """
    Loads Bitcoin price data from a CSV file.

    Parameters:
    file_path (str): Path to the CSV file containing the data.

    Returns:
    pandas.DataFrame: Dataframe containing the data.

    Raises:
    FileNotFoundError: If the file does not exist.
    pd.errors.EmptyDataError: If the file is empty.
    """
    try:
        df = pd.read_csv(file_path)

        if 'Date' not in df.columns:
            raise ValueError("DataFrame must contain a 'Date' column.")
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        if df.isnull().any().any():
            df.interpolate(method='time', inplace=True)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{file_path}' not found.")
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError(f"File '{file_path}' is empty.")


def normalize_features(df, columns_to_scale):
    """
    Normalizes the values of the given columns using the MinMaxScaler.

    Parameters:
    df (pandas.DataFrame): Dataframe containing Bitcoin price data.
    columns_to_scale (list): List of column names to scale.

    Returns:
    pandas.DataFrame: Dataframe with normalized values.
    """
    missing_columns = [col for col in columns_to_scale if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Columns not found in DataFrame: {missing_columns}")

    scaler = MinMaxScaler()
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
    return df, scaler


def save_scaler(scaler, path='models/scaler.pkl'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)


def calculate_technical_indicators(df):
    """
    Calculates technical indicators for the given price data.

    Parameters:
    df (pandas.DataFrame): Dataframe containing Bitcoin price data.

    Returns:
    pandas.DataFrame: Dataframe with technical indicators.

    Raises:
    ValueError: If the DataFrame does not contain a 'Date' column.
    ValueError: If the DataFrame does not contain the required columns.
    """
    required_columns = ['Close']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain the following columns: {required_columns}")

    df['MA50'] = df['Close'].rolling(50).mean()
    df['MA200'] = df['Close'].rolling(200).mean()
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(50).std()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['Upper'] = df['MA20'] + 2 * df['Volatility']
    df['Lower'] = df['MA20'] - 2 * df['Volatility']
    return df


def visualize_data(df):
    """
    Visualizes the given data.

    Parameters:
    df (pandas.DataFrame): Dataframe containing Bitcoin price data.
    """
    plt.figure(figsize=(15, 10))
    plt.plot(df['Close'], label='Close')
    plt.plot(df['MA50'], label='MA50')
    plt.plot(df['MA200'], label='MA200')
    plt.plot(df['Upper'], label='Upper')
    plt.plot(df['Lower'], label='Lower')
    plt.title('Bitcoin price history')
    plt.ylabel('Price (USD)')
    plt.xlabel('Date')
    plt.legend(loc='upper left')
    plt.show()


def main():
    file_path = 'data/BTC-USD.csv'
    df = load_data(file_path)
    df = calculate_technical_indicators(df)
    columns_to_scale = ['Close', 'MA50', 'MA200', 'Returns', 'Volatility', 'MA20', 'Upper', 'Lower']
    df, scaler = normalize_features(df, columns_to_scale)
    save_scaler(scaler)  # Save the scaler separately
    df.to_csv('data/processed_data.csv', index=True)
    visualize_data(df)

if __name__ == '__main__':
    main()
