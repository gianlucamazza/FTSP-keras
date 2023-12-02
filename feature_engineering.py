# feature_engineering.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os


def add_additional_features(df):
    df['RSI'] = calculate_rsi(df['Close'])
    df['MACD'] = calculate_macd(df['Close'])
    return df.ffill()


def calculate_rsi(prices, n=14):
    deltas = prices.diff()
    loss = deltas.where(deltas < 0, 0)
    gain = deltas.where(deltas > 0, 0)
    avg_gain = gain.rolling(window=n, min_periods=1).mean()
    avg_loss = -loss.rolling(window=n, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def calculate_macd(prices, n_fast=12, n_slow=26):
    ema_fast = prices.ewm(span=n_fast, min_periods=n_slow).mean()
    ema_slow = prices.ewm(span=n_slow, min_periods=n_slow).mean()
    return ema_fast - ema_slow


def normalize_features(df):
    scaler = MinMaxScaler()
    numeric_columns = df.select_dtypes(include=['number']).columns
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    return df, scaler


def save_scaler(scaler, path='scalers/scaler.pkl'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)


def normalize_close_feature(df):
    close_scaler = MinMaxScaler()
    df['Close'] = close_scaler.fit_transform(df[['Close']])
    return df, close_scaler


def save_close_scaler(scaler, path='scalers/close_scaler.pkl'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)


def main():
    df = pd.read_csv('data/processed_data.csv', index_col='Date')
    df = add_additional_features(df)
    df, close_scaler = normalize_close_feature(df)
    save_close_scaler(close_scaler, path='scalers/close_scaler.pkl')
    df, scaler = normalize_features(df)
    save_scaler(scaler)
    df.to_csv('data/scaled_data.csv', index=True)


if __name__ == '__main__':
    main()
