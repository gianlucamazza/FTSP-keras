import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from technical_indicators import calculate_technical_indicators
from config import COLUMN_SETS
from keras.models import load_model
from logger import setup_logger
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
logger = setup_logger('predict_logger', BASE_DIR / 'logs', 'predict.log')


def arrange_and_fill(df):
    df.sort_index(inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df


class DataPreparator:
    def __init__(self, ticker):
        self.ticker = ticker
        self.data_path = BASE_DIR / f'data/raw_data_{self.ticker}.csv'
        self.processed_data_path = BASE_DIR / f'data/processed_data_{self.ticker}.csv'
        self.scaled_data_path = BASE_DIR / f'data/scaled_data_{self.ticker}.csv'
        self.feature_scaler_path = BASE_DIR / f'scalers/feature_scaler_{self.ticker}.pkl'
        self.close_scaler_path = BASE_DIR / f'scalers/close_scaler_{self.ticker}.pkl'

    def process_and_save_features(self, df):
        missing_columns = set(COLUMN_SETS['to_scale']) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing columns in DataFrame: {missing_columns}")

        feature_scaler = MinMaxScaler()
        close_scaler = MinMaxScaler()

        df[COLUMN_SETS['to_scale']] = feature_scaler.fit_transform(df[COLUMN_SETS['to_scale']])
        df[['Close']] = close_scaler.fit_transform(df[['Close']])

        joblib.dump(feature_scaler, self.feature_scaler_path)
        joblib.dump(close_scaler, self.close_scaler_path)

        return df, feature_scaler, close_scaler

    def prepare_data(self):
        df = pd.read_csv(self.data_path, index_col='Date', parse_dates=True)
        df = arrange_and_fill(df)
        df = calculate_technical_indicators(df)
        df.to_csv(self.processed_data_path, index=True)
        df_scaled, feature_scaler, close_scaler = self.process_and_save_features(df)
        df_scaled.to_csv(self.scaled_data_path, index=True)
        return df_scaled, feature_scaler, close_scaler


class ModelPredictor:
    def __init__(self, ticker):
        self.model_path = BASE_DIR / f'models/model_{ticker}.keras'
        self.model = load_model(self.model_path)
        self.feature_scaler = joblib.load(BASE_DIR / f'scalers/feature_scaler_{ticker}.pkl')
        self.close_scaler = joblib.load(BASE_DIR / f'scalers/close_scaler_{ticker}.pkl')

    def predict(self, df_scaled):
        predictions = self.model.predict(df_scaled)
        return predictions

    def inverse_scale_predictions(self, predictions):
        return self.close_scaler.inverse_transform(predictions)


def plot_predictions(predictions, actual_values, ticker, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(predictions, label='Predicted')
    plt.plot(actual_values, label='Actual')
    plt.title(f'{ticker} Price Predictions')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig(save_path)
    plt.show()


def main(ticker='BTC-USD', start_date='2022-01-01', end_date='2023-01-01'):
    data_preparator = DataPreparator(ticker)
    df_scaled, feature_scaler, close_scaler = data_preparator.prepare_data()

    model_predictor = ModelPredictor(ticker)
    predictions_scaled = model_predictor.predict(df_scaled)
    predictions = model_predictor.inverse_scale_predictions(predictions_scaled)

    df = pd.read_csv(BASE_DIR / f'data/raw_data_{ticker}.csv', index_col='Date', parse_dates=True)
    plot_predictions(predictions, df['Close'].values, ticker, BASE_DIR / f'predictions/predictions_{ticker}.png')


if __name__ == '__main__':
    main()
