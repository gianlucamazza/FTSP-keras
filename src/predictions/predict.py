import sys
import argparse
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from keras._tf_keras.keras.models import load_model
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

project_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_dir))

from src.logging.logger import setup_logger
from src.config import COLUMN_SETS, CLOSE
from src.indicators.technical_indicators import calculate_technical_indicators
from src.data.feature_engineering import process_and_save_features
from utils import load_from_json

ROOT_DIR = project_dir
logger = setup_logger('predict_logger', 'logs', 'predict.log')


def load_scaler(path):
    """Load a scaler from disk."""
    if not path.exists():
        logger.error(f"Scaler file not found: {path}")
        raise FileNotFoundError(f"Scaler file not found: {path}")
    try:
        return joblib.load(path)
    except Exception as e:
        logger.error(f"Error loading scaler from {path}: {e}")
        raise


class ModelPredictor:
    COLUMN_TO_PREDICT = CLOSE
    DATA_FOLDER = 'data'
    SCALERS_FOLDER = 'scalers'
    MODELS_FOLDER = 'models'
    PREDICTIONS_FOLDER = 'predictions'

    def __init__(self, ticker='BTC'):
        self.ticker = ticker
        self.data_path = ROOT_DIR / f'{self.DATA_FOLDER}/processed_data_{self.ticker}.csv'
        self.scaler_path = ROOT_DIR / f'{self.SCALERS_FOLDER}/feature_scaler_{self.ticker}.pkl'
        self.arima_path = ROOT_DIR / f'{self.ticker}_arima_params.json'
        self.model_path = ROOT_DIR / f'{self.MODELS_FOLDER}/{self.ticker}_best_model.keras'
        self.params_path = ROOT_DIR / f'{self.ticker}_best_params.json'

        logger.info(f"Initializing ModelPredictor for ticker {ticker}")
        logger.info(f"Data path: {self.data_path}")
        logger.info(f"Scaler path: {self.scaler_path}")
        logger.info(f"Model path: {self.model_path}")
        logger.info(f"Params path: {self.params_path}")

        self.feature_scaler = load_scaler(self.scaler_path)
        self.model = load_model(self.model_path)
        self.df = self.load_dataset()

        self.best_params = load_from_json(self.params_path)
        self.prediction_steps = self.best_params.get('train_steps', 30)
        self.arima_params = load_from_json(self.arima_path)
        logger.info(f"Using train_steps: {self.prediction_steps}")

    def load_dataset(self):
        """Load the dataset from disk."""
        if not self.data_path.exists():
            logger.error(f"Dataset file not found: {self.data_path}")
            raise FileNotFoundError(f"Dataset file not found: {self.data_path}")
        try:
            df = pd.read_csv(self.data_path, index_col='Date', parse_dates=True)
            df.ffill(inplace=True)
            return df
        except Exception as e:
            logger.error(f"Error loading dataset: {e}", exc_info=True)
            raise

    def prepare_data(self):
        """Prepare the data by calculating indicators and scaling."""
        logger.info("Calculating technical indicators.")
        self.df = calculate_technical_indicators(self.df)

        logger.info("Scaling data.")
        scaler_columns = COLUMN_SETS['to_scale']

        # Check for missing columns
        missing_columns = [col for col in scaler_columns if col not in self.df.columns]
        if missing_columns:
            logger.error(f"Missing columns for scaling: {missing_columns}")
            raise KeyError(f"Missing columns for scaling: {missing_columns}")

        self.df[scaler_columns] = self.feature_scaler.transform(self.df[scaler_columns])

    def create_input_sequences(self):
        """Create input sequences for prediction."""
        self.prepare_data()
        # Use only the last 'prediction_steps' days
        df_last_30 = self.df.tail(self.prediction_steps)
        if len(df_last_30) < self.prediction_steps:
            logger.error(f"Not enough data to create input sequences. "
                         f"Required: {self.prediction_steps}, available: {len(df_last_30)}")
            raise ValueError(f"Not enough data to create input sequences. "
                             f"Required: {self.prediction_steps}, available: {len(df_last_30)}")
        x = []
        data = df_last_30.values
        x.append(data[:self.prediction_steps])
        return np.array(x)

    def predict(self, num_predictions=10):
        """Make predictions using the trained model."""
        x = self.create_input_sequences()
        predictions = []

        # Start with the last sequence
        seq = x[-1]

        for _ in range(num_predictions):
            x_input = np.expand_dims(seq, axis=0)
            pred = self.model.predict(x_input)
            predictions.append(pred.flatten()[0])

            # Update the sequence with the new prediction
            # Ensure new_input has the same number of features as seq
            new_row = np.append(seq[-1, 1:], pred.flatten()[0]).reshape(1, -1)
            seq = np.vstack([seq[1:], new_row])

        return np.array(predictions)

    def predict_arima(self, num_predictions=10):
        """Make predictions using the ARIMA model."""
        arima_order = tuple(self.arima_params['arima_order'])
        seasonal_order = tuple(self.arima_params['arima_seasonal_order'])
        model = ARIMA(self.df[self.COLUMN_TO_PREDICT], order=arima_order, seasonal_order=seasonal_order,
                      freq=self.arima_params['frequency'])
        model_fit = model.fit()
        predictions = model_fit.forecast(steps=num_predictions)
        return predictions

    def inverse_transform_predictions(self, predictions):
        """Inverse transform the predictions."""
        predictions = np.array(predictions).reshape(-1, 1)
        # Inverse transform using only the 'Close' column scaler
        scaler_columns = self.df.columns.tolist()
        close_index = scaler_columns.index(self.COLUMN_TO_PREDICT)
        full_inverse_scaled = self.feature_scaler.inverse_transform(
            np.hstack([np.zeros((predictions.shape[0], len(scaler_columns) - 1)), predictions])
        )
        return full_inverse_scaled[:, close_index]

    def inverse_transform_historical(self):
        """Inverse transform the historical data."""
        # Inverse transform using the feature scaler
        scaler_columns = self.df.columns.tolist()
        transformed_data = self.feature_scaler.inverse_transform(self.df[scaler_columns])
        return pd.DataFrame(transformed_data, columns=scaler_columns, index=self.df.index)

    def save_predictions(self, predictions, file_path):
        """Save the historical data and predictions to disk."""
        # Derive frequency from historical data
        last_date = self.df.index[-1]
        inferred_freq = pd.infer_freq(self.df.index)
        if not inferred_freq:
            logger.error("Unable to infer frequency from the historical data index.")
            raise ValueError("Unable to infer frequency from the historical data index.")

        future_dates = pd.date_range(start=last_date, periods=len(predictions) + 1, freq=inferred_freq)[1:]
        predicted_data = pd.Series(predictions, index=future_dates)

        historical_data = self.inverse_transform_historical()
        combined_df = pd.concat([historical_data[self.COLUMN_TO_PREDICT], predicted_data], axis=1)
        combined_df.columns = ['Historical', 'Predicted']

        combined_df.to_csv(file_path, index=True)
        logger.info(f"Predictions saved at {file_path}")

    def plot_predictions(self, predictions):
        """Plot the predictions along with the actual data."""
        plt.style.use('ggplot')
        plt.figure(figsize=(15, 7))

        # Combine historical data with predictions
        actual_data = self.inverse_transform_historical()[self.COLUMN_TO_PREDICT]
        future_dates = pd.date_range(start=actual_data.index[-1], periods=len(predictions) + 1, freq='B')[1:]
        predicted_data = pd.Series(predictions, index=future_dates)

        plt.plot(actual_data.index, actual_data, label='Actual', color='blue', linewidth=2)
        plt.plot(predicted_data.index, predicted_data, label='Predicted', linestyle='--', color='orange', linewidth=2)

        plt.fill_between(predicted_data.index, actual_data.iloc[-1], predicted_data, color='gray', alpha=0.2)

        # Annotate max, min, and some significant points
        plt.scatter(actual_data.idxmax(), actual_data.max(), color='red', marker='o', label='Max Actual')
        plt.scatter(actual_data.idxmin(), actual_data.min(), color='green', marker='o', label='Min Actual')
        plt.scatter(actual_data.index[-1], actual_data.iloc[-1], color='purple', marker='o', label='Last Actual')
        plt.scatter(predicted_data.index[0], predicted_data.iloc[0], color='yellow', marker='o',
                    label='First Predicted')
        plt.scatter(predicted_data.index[-1], predicted_data.iloc[-1], color='yellow', marker='o',
                    label='Last Predicted')

        plt.legend()
        plt.title(f'{self.ticker} Close Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.xticks(rotation=45)

        # Add a shaded area for prediction period
        plt.axvspan(actual_data.index[-1], predicted_data.index[-1], color='lightblue', alpha=0.3,
                    label='Prediction Period')

        plt.tight_layout()

        # Save plot
        plot_path = ROOT_DIR / f'{self.PREDICTIONS_FOLDER}/{self.ticker}_prediction_plot.png'
        plt.savefig(plot_path, dpi=300)
        logger.info(f"Prediction plot saved at {plot_path}")

        plt.show()

    def run(self):
        """Run the prediction process."""
        try:
            predictions_scaled = self.predict()
            predictions = self.inverse_transform_predictions(predictions_scaled)
            self.plot_predictions(predictions)
            predictions_path = ROOT_DIR / f'{self.PREDICTIONS_FOLDER}/{self.ticker}_predictions.csv'
            self.save_predictions(predictions, predictions_path)
        except Exception as e:
            logger.error(f"Error in prediction process: {e}", exc_info=True)


def main(ticker: str):
    """Main function to run the prediction script."""
    predictor = ModelPredictor(ticker)
    predictor.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data Prediction')
    parser.add_argument('--ticker', type=str, required=True, help='Ticker label')

    main(ticker=parser.parse_args().ticker)
