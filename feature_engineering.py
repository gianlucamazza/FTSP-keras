# feature_engineering.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from data_preparation import COLUMNS_TO_SCALE
import joblib
import os


def validate_input_data(df, required_columns):
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in input data: {missing_columns}")


def normalize_features(df, columns_to_normalize):
    scaler = MinMaxScaler()
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    return df, scaler


def normalize_close(df):
    scaler = MinMaxScaler()
    df['Close'] = scaler.fit_transform(df[['Close']])
    return scaler


def save_scaler(scaler, ticker, scaler_type='feature'):
    path = f'scalers/{scaler_type}_scaler_{ticker}.pkl'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)


def process_and_save_features(df, ticker):
    required_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'] + COLUMNS_TO_SCALE
    validate_input_data(df, required_columns)

    # Normalize all features except Close
    columns_to_normalize = [col for col in df.columns if col != 'Close']
    df, feature_scaler = normalize_features(df, columns_to_normalize)

    # Normalize Close
    close_scaler = normalize_close(df)
    save_scaler(close_scaler, ticker, scaler_type='close')

    # Save scaler for all features
    save_scaler(feature_scaler, ticker, scaler_type='feature')

    # Save scaled data
    output_file_path = f'data/scaled_data_{ticker}.csv'
    df.to_csv(output_file_path, index=True)
    return df


def main(ticker='BTC-USD'):
    file_path = f'data/processed_data_{ticker}.csv'
    df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    process_and_save_features(df, ticker)


if __name__ == '__main__':
    main(ticker='BTC-USD')
