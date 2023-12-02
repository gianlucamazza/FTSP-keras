# data_preparation.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import yfinance as yf


def load_data(file_path):
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


def merge_data(df1, df2):
    df = pd.concat([df1, df2], axis=0)
    df = df[~df.index.duplicated(keep='last')]
    return df


def get_new_financial_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    df.reset_index(inplace=True)
    df.rename(columns={'Adj Close': 'Adj_Close'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df[['Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']]
    return df


def normalize_features(df, columns_to_scale):
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
    plt.figure(figsize=(15, 10))
    plt.plot(df['Close'], label='Close')
    plt.title('Bitcoin price history')
    plt.ylabel('Price (USD)')
    plt.xlabel('Date')
    plt.legend(loc='upper left')
    plt.show()


def main():
    file_path = 'data/BTC-USD.csv'
    df = load_data(file_path)
    last_date = df.index[-1]

    if last_date.date() < pd.Timestamp.today().date():
        start_date = last_date + pd.Timedelta(days=1)
        end_date = pd.Timestamp.today()
        new_data = get_new_financial_data('BTC-USD', start_date, end_date)
        df = merge_data(df, new_data)
        df.to_csv(file_path, index=True)

    df = calculate_technical_indicators(df)
    columns_to_scale = ['MA50', 'MA200', 'Returns', 'Volatility', 'MA20', 'Upper', 'Lower']
    df, scaler = normalize_features(df, columns_to_scale)
    save_scaler(scaler)
    df.to_csv('data/processed_data.csv', index=True)
    visualize_data(df)


if __name__ == '__main__':
    main()
