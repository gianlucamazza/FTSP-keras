# data_preparation.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib
import os


def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        # Ensure data consistency by checking for missing values
        if df.isnull().any().any():
            raise ValueError("Data contains missing values.")
        return df
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"File '{file_path}' is empty.")
        return None


def normalize_features(df, columns_to_scale):
    """
    Normalizes specified columns in the DataFrame.

    Parameters:
    df (pandas.DataFrame): Dataframe containing data to normalize.
    columns_to_scale (list): List of column names to normalize.

    Returns:
    pandas.DataFrame: Dataframe with normalized columns.
    """
    scaler = MinMaxScaler()
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    return df


def calculate_technical_indicators(df):
    """
    Calculates various technical indicators for Bitcoin price analysis.

    Parameters:
    df (pandas.DataFrame): Dataframe containing Bitcoin price data.

    Returns:
    pandas.DataFrame: Dataframe with additional columns for technical indicators.
    """
    if df.isna().any().any():
        df.ffill(inplace=True)

    if 'Date' not in df.columns:
        raise ValueError("DataFrame must contain a 'Date' column.")

    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    required_columns = ['Close']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain the following columns: {required_columns}")

    # Moving averages
    df['MA50'] = df['Close'].rolling(50).mean()
    df['MA200'] = df['Close'].rolling(200).mean()

    # Daily returns
    df['Returns'] = df['Close'].pct_change()

    # Volatility (standard deviation of returns)
    df['Volatility'] = df['Returns'].rolling(50).std()

    # Bollinger Bands
    df['MA20'] = df['Close'].rolling(20).mean()
    df['Upper'] = df['MA20'] + 2 * df['Volatility']
    df['Lower'] = df['MA20'] - 2 * df['Volatility']

    if df.isna().any().any():
        df.ffill(inplace=True)

    return df


def plot_technical_indicators(df):
    """
    Plots various technical indicators for Bitcoin price data.

    Parameters:
    df (pandas.DataFrame): Dataframe containing Bitcoin price data and technical indicators.
    """
    plot_indicator(df, ['Close', 'MA50', 'MA200'], 'Bitcoin Price and Moving Averages')
    plot_indicator(df, ['Returns'], 'Bitcoin Daily Returns')
    plot_indicator(df, ['Volatility'], 'Bitcoin Price Volatility')
    plot_indicator(df, ['Close', 'MA20', 'Upper', 'Lower'], 'Bitcoin Price and Bollinger Bands')


def plot_indicator(df, columns, title):
    """
    Plots a technical indicator for Bitcoin price data.

    Parameters:
    df (pandas.DataFrame): Dataframe containing Bitcoin price data and technical indicators.
    columns (list): List of column names to plot.
    title (str): Title of the plot.
    """
    df[columns].plot(figsize=(12, 6))
    plt.title(title)
    plt.show()


def main():
    file_path = 'data/BTC-USD.csv'
    df = load_data(file_path)
    df = calculate_technical_indicators(df)
    columns_to_scale = ['Close', 'MA50', 'MA200', 'Returns', 'Volatility', 'MA20', 'Upper', 'Lower']
    df = normalize_features(df, columns_to_scale)
    plot_technical_indicators(df)
    df.to_csv('data/processed_data.csv', index_label='Date')


if __name__ == '__main__':
    main()
