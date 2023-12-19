import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from data_preparation import calculate_technical_indicators
from data_preparation import COLUMNS_TO_SCALE as FEATURES
from keras.models import load_model
from logger import setup_logger

logger = setup_logger('predict_logger', 'logs', 'predict.log')


def arrange_and_fill(df):
    df.sort_index(inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df


class DataPreparator:
    def __init__(self, ticker, data_path, processed_data_path, scaled_data_path, feature_scaler_path, close_scaler_path):
        self.ticker = ticker
        self.data_path = data_path
        self.processed_data_path = processed_data_path
        self.scaled_data_path = scaled_data_path
        self.feature_scaler_path = feature_scaler_path
        self.close_scaler_path = close_scaler_path

    def process_and_save_features(self, df):
        missing_columns = set(FEATURES) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing columns in DataFrame: {missing_columns}")

        feature_scaler = MinMaxScaler()
        close_scaler = MinMaxScaler()

        df[FEATURES] = feature_scaler.fit_transform(df[FEATURES])
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
    def __init__(self, model_path, feature_scaler, close_scaler):
        self.model = load_model(model_path)
        self.feature_scaler = feature_scaler
        self.close_scaler = close_scaler

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
    data_path = f'data/raw_data_{ticker}.csv'
    processed_data_path = f'data/processed_data_{ticker}.csv'
    scaled_data_path = f'data/scaled_data_{ticker}.csv'
    model_path = f'models/model_{ticker}.keras'
    feature_scaler_path = f'scalers/feature_scaler_{ticker}.pkl'
    close_scaler_path = f'scalers/close_scaler_{ticker}.pkl'

    data_preparator = DataPreparator(ticker, data_path, processed_data_path, scaled_data_path, feature_scaler_path,
                                     close_scaler_path)
    df_scaled, feature_scaler, close_scaler = data_preparator.prepare_data()

    model_predictor = ModelPredictor(model_path, feature_scaler, close_scaler)
    predictions_scaled = model_predictor.predict(df_scaled)
    predictions = model_predictor.inverse_scale_predictions(predictions_scaled)

    df = pd.read_csv(data_path, index_col='Date', parse_dates=True)
    plot_predictions(predictions, df['Close'].values, ticker, f'predictions/predictions_{ticker}.png')


if __name__ == '__main__':
    main()
