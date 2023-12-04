# feature_engineering.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os


def normalize_features(df, columns_to_normalize):
    scaler = MinMaxScaler()
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    return df, scaler


def save_scaler(scaler, ticker, scaler_type='feature'):
    path = f'scalers/{scaler_type}_scaler_{ticker}.pkl'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)


def process_and_save_features(df, ticker):
    # Normalize all features except Close
    columns_to_normalize = df.select_dtypes(include=['number']).columns.tolist()
    df, feature_scaler = normalize_features(df, columns_to_normalize)

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
