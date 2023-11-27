import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib


def load_data(file_path):
    """
    Loads Bitcoin price data from a CSV file.

    Parameters:
    file_path (str): Path to the CSV file containing the data.

    Returns:
    pandas.DataFrame: Dataframe containing the loaded data.
    """
    return pd.read_csv(file_path)


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

    # Salva lo scaler per un uso futuro
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
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

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

    return df


def plot_technical_indicators(df):
    """
    Plots various technical indicators for Bitcoin price data.

    Parameters:
    df (pandas.DataFrame): Dataframe containing Bitcoin price data and technical indicators.
    """
    # Moving Averages Plot
    df[['Close', 'MA50', 'MA200']].plot(figsize=(12, 6))
    plt.title('Bitcoin Price and Moving Averages')
    plt.show()

    # Returns Plot
    df['Returns'].plot(figsize=(12, 6))
    plt.title('Bitcoin Daily Returns')
    plt.show()

    # Volatility Plot
    df['Volatility'].plot(figsize=(12, 6))
    plt.title('Bitcoin Price Volatility')
    plt.show()

    # Bollinger Bands Plot
    df[['Close', 'MA20', 'Upper', 'Lower']].plot(figsize=(12, 6))
    plt.title('Bitcoin Price and Bollinger Bands')
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
