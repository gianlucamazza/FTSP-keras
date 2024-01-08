import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
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
        self.feature_scaler_path = BASE_DIR / f'scalers/feature_scaler_{self.ticker}.pkl'
        self.close_scaler_path = BASE_DIR / f'scalers/close_scaler_{self.ticker}.pkl'

    def prepare_data_for_prediction(self):
        df = pd.read_csv(self.data_path, index_col='Date', parse_dates=True)
        df = arrange_and_fill(df)
        df = calculate_technical_indicators(df)

        N = 60  # La lunghezza della sequenza aspettata dal modello
        last_n_rows = df.tail(N).copy()

        # Carica gli scaler esistenti
        feature_scaler = joblib.load(self.feature_scaler_path)
        close_scaler = joblib.load(self.close_scaler_path)

        # Applica la trasformazione
        last_n_rows[COLUMN_SETS['to_scale']] = feature_scaler.transform(last_n_rows[COLUMN_SETS['to_scale']])
        last_n_rows[['Close']] = close_scaler.transform(last_n_rows[['Close']])

        # Trasforma i dati in sequenze
        sequences = []
        for i in range(len(last_n_rows) - N + 1):
            sequence = last_n_rows.iloc[i:i + N]
            sequences.append(sequence.values)

        return np.array(sequences)  # Ristruttura in un array numpy


class ModelPredictor:
    def __init__(self, ticker):
        self.model_path = BASE_DIR / f'models/model_{ticker}.keras'
        self.model = load_model(self.model_path)
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
    df_scaled = data_preparator.prepare_data_for_prediction()

    model_predictor = ModelPredictor(ticker)
    predictions_scaled = model_predictor.predict(df_scaled)
    predictions = model_predictor.inverse_scale_predictions(predictions_scaled)

    df = pd.read_csv(BASE_DIR / f'data/raw_data_{ticker}.csv', index_col='Date', parse_dates=True)
    actual_values = df['Close'].tail(len(predictions)).values
    plot_predictions(predictions, actual_values, ticker, BASE_DIR / f'predictions/predictions_{ticker}.png')


if __name__ == '__main__':
    main()
