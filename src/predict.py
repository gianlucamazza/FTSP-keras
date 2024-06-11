import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from technical_indicators import calculate_technical_indicators
from config import COLUMN_SETS, CLOSE
from keras.models import load_model
from logger import setup_logger
from pathlib import Path
from datetime import timedelta

BASE_DIR = Path(__file__).parent.parent
logger = setup_logger('predict_logger', BASE_DIR / 'logs', 'predict.log')


def arrange_and_fill(df):
    """Sorts the DataFrame by index and fills missing values."""
    df.sort_index(inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df


def create_windowed_data(df, steps):
    """Creates windowed data for time-series prediction."""
    x = []
    for i in range(steps, len(df)):
        x.append(df[i - steps:i])
    return np.array(x)


def scale_close_column(df, df_scaled, close_scaler_path):
    """Scales the 'Close' column of the DataFrame using the saved scaler."""
    if 'Close' in COLUMN_SETS['to_scale'] and 'Close' in df_scaled.columns:
        close_scaler = joblib.load(close_scaler_path)
        close_index = df_scaled.columns.tolist().index('Close')
        df_scaled.iloc[:, close_index] = close_scaler.transform(df[['Close']])
    else:
        close_scaler = None
    return close_scaler


def scale_features(df):
    """Scales the features of the DataFrame."""
    feature_scaler = MinMaxScaler()
    scaler_columns = [col for col in COLUMN_SETS['to_scale'] if col in df.columns]
    df_scaled = df[scaler_columns]
    df_scaled = feature_scaler.fit_transform(df_scaled)
    return feature_scaler, pd.DataFrame(df_scaled, columns=scaler_columns, index=df.index)


class DataPreparator:
    def __init__(self, ticker):
        self.ticker = ticker
        self.data_path = BASE_DIR / f'data/raw_data_{self.ticker}.csv'
        self.processed_data_path = BASE_DIR / f'data/processed_data_{self.ticker}.csv'
        self.scaled_data_path = BASE_DIR / f'data/scaled_data_{self.ticker}.csv'
        self.feature_scaler_path = BASE_DIR / f'scalers/feature_scaler_{self.ticker}.pkl'
        self.close_scaler_path = BASE_DIR / f'scalers/close_scaler_{self.ticker}.pkl'

    def process_and_save_features(self, df):
        """Scales features and saves the scaled data."""
        missing_columns = set(COLUMN_SETS['to_scale']) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing columns in DataFrame: {missing_columns}")

        feature_scaler, df_scaled = scale_features(df)
        joblib.dump(feature_scaler, self.feature_scaler_path)

        close_scaler = scale_close_column(df, df_scaled, self.close_scaler_path)
        if close_scaler:
            joblib.dump(close_scaler, self.close_scaler_path)

        return df_scaled, feature_scaler, close_scaler

    def prepare_data(self):
        """Prepares data for model training and prediction."""
        df = pd.read_csv(self.data_path, index_col='Date', parse_dates=True)
        df = arrange_and_fill(df)
        df = calculate_technical_indicators(df)
        df.to_csv(self.processed_data_path, index=True)

        df_support = df[[CLOSE]].copy()
        df_support.index.name = 'Date'

        df_scaled, feature_scaler, close_scaler = self.process_and_save_features(df)
        df_scaled.to_csv(self.scaled_data_path, index=True)
        return df_scaled, feature_scaler, close_scaler, df_support


def prepare_data_for_plotting(predictions, actual_values, historical_dates, future_dates):
    """Prepares data for plotting by creating a combined DataFrame."""
    all_dates = historical_dates.tolist() + future_dates.tolist()
    plot_df = pd.DataFrame(index=all_dates)

    plot_df['Actual'] = np.concatenate((actual_values, [np.nan] * len(future_dates)))
    plot_df['Predicted'] = np.concatenate(([np.nan] * len(historical_dates), predictions.flatten()))

    return plot_df


class ModelPredictor:
    def __init__(self, ticker):
        self.model_path = BASE_DIR / f'models/model_{ticker}.keras'
        try:
            self.model = load_model(self.model_path)
            self.feature_scaler = joblib.load(BASE_DIR / f'scalers/feature_scaler_{ticker}.pkl')
            self.close_scaler = joblib.load(BASE_DIR / f'scalers/close_scaler_{ticker}.pkl')
        except Exception as e:
            logger.error(f"Failed to initialize ModelPredictor: {e}")
            raise
        self.train_steps = 60

    def predict(self, df_scaled):
        """Predicts future values based on the scaled data."""
        df_windowed = create_windowed_data(df_scaled, self.train_steps)
        try:
            predictions = self.model.predict(df_windowed)
            assert len(predictions) == len(
                df_windowed), "The number of predictions does not match the length of the dataframe"
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
        return predictions

    def inverse_scale_predictions(self, predictions):
        """Inverse scales the predictions to the original scale."""
        predictions = predictions.reshape(-1, 1)
        return self.close_scaler.inverse_transform(predictions)


def main(ticker='BTC-USD'):
    try:
        data_preparator = DataPreparator(ticker)
        df_scaled, feature_scaler, close_scaler, df_support = data_preparator.prepare_data()

        model_predictor = ModelPredictor(ticker)
        predictions_scaled = model_predictor.predict(df_scaled)
        predictions = model_predictor.inverse_scale_predictions(predictions_scaled)
        logger.info(f"Number of predictions made: {len(predictions)}")

        df = pd.read_csv(BASE_DIR / f'data/raw_data_{ticker}.csv', index_col='Date', parse_dates=True)
        historical_dates = df.index
        last_historical_date = historical_dates[-1]

        # Log the start and end of the historical dates
        logger.info(f"Historical data starts on: {historical_dates[0]}")
        logger.info(f"Historical data ends on: {last_historical_date}")

        future_dates = pd.date_range(start=last_historical_date + timedelta(days=1), periods=len(predictions_scaled))

        plot_df = prepare_data_for_plotting(
            predictions,
            df['Close'].values[-len(historical_dates):],
            historical_dates,
            future_dates
        )

        # Log the start and end of the future dates
        logger.info(f"Predictions start on: {future_dates[0]}")
        logger.info(f"Predictions end on: {future_dates[-1]}")

        plt.figure(figsize=(15, 7))
        plt.plot(plot_df.index, plot_df['Actual'], label='Actual', color='blue')
        plt.plot(plot_df.index, plot_df['Predicted'], label='Predicted', linestyle='--', color='orange')

        plt.legend()
        plt.show()

    except Exception as e:
        logger.error(f"Error in main: {e}")


if __name__ == '__main__':
    main()
