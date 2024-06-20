import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from keras.models import load_model
import matplotlib.pyplot as plt
import logger as logger_module
from config import COLUMN_SETS, CLOSE, PARAMETERS
from technical_indicators import calculate_technical_indicators

# Add the project directory to the sys.path
project_dir = Path(__file__).resolve().parent
sys.path.append(str(project_dir))

BASE_DIR = Path(__file__).parent.parent
logger = logger_module.setup_logger('predict_logger', BASE_DIR / 'logs', 'predict.log')


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

    def __init__(self, ticker='BTC-USD'):
        self.ticker = ticker
        self.data_path = BASE_DIR / f'{self.DATA_FOLDER}/raw_data_{self.ticker}.csv'
        self.scaler_path = BASE_DIR / f'{self.SCALERS_FOLDER}/feature_scaler_{self.ticker}.pkl'
        self.model_path = BASE_DIR / f'{self.MODELS_FOLDER}/model_{self.ticker}_best.keras'

        logger.info(f"Initializing ModelPredictor for ticker {ticker}")
        logger.info(f"Data path: {self.data_path}")
        logger.info(f"Scaler path: {self.scaler_path}")
        logger.info(f"Model path: {self.model_path}")

        self.feature_scaler = load_scaler(self.scaler_path)
        self.model = load_model(self.model_path)
        self.df = self.load_dataset()

    def load_dataset(self):
        """Load the dataset from disk."""
        if not self.data_path.exists():
            logger.error(f"Dataset file not found: {self.data_path}")
            raise FileNotFoundError(f"Dataset file not found: {self.data_path}")
        try:
            df = pd.read_csv(self.data_path, index_col='Date')
            df.ffill(inplace=True)
            return df
        except Exception as e:
            logger.error(f"Error loading dataset: {e}", exc_info=True)
            raise

    def prepare_data(self):
        """Prepare the data by adding technical indicators and scaling."""
        logger.info("Calculating technical indicators.")
        self.df = calculate_technical_indicators(self.df)

        # Ensure all required columns are present
        missing_columns = set(COLUMN_SETS['to_scale']) - set(self.df.columns)
        if missing_columns:
            logger.error(f"Missing required columns in the DataFrame: {missing_columns}")
            raise ValueError(f"Missing required columns in the DataFrame: {missing_columns}")

        logger.info("Scaling data.")
        scaler_columns = COLUMN_SETS['to_scale']
        self.df[scaler_columns] = self.feature_scaler.transform(self.df[scaler_columns])

    def predict(self):
        """Make predictions using the trained model."""
        self.prepare_data()
        x = self.df.values[-PARAMETERS['train_steps']:]
        x = np.expand_dims(x, axis=0)
        predictions = self.model.predict(x)
        return predictions.flatten()

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

    def save_predictions(self, predictions, file_path):
        """Save the predictions to disk."""
        predictions_df = pd.DataFrame(predictions, columns=[self.COLUMN_TO_PREDICT])
        predictions_df.to_csv(file_path, index=False)
        logger.info(f"Predictions saved at {file_path}")

    def plot_predictions(self, predictions):
        """Plot the predictions along with the actual data."""
        plt.style.use('ggplot')
        plt.figure(figsize=(15, 7))
        actual_data = self.df[self.COLUMN_TO_PREDICT][-len(predictions):]

        plt.plot(actual_data.index, actual_data, label='Actual', color='blue', linewidth=2)
        plt.plot(actual_data.index, predictions, label='Predicted', linestyle='--', color='orange', linewidth=2)

        plt.fill_between(actual_data.index, actual_data, predictions, color='gray', alpha=0.2)

        # Annotate max, min, and some significant points
        plt.scatter(actual_data.idxmax(), actual_data.max(), color='red', marker='o', label='Max Actual')
        plt.scatter(actual_data.idxmin(), actual_data.min(), color='green', marker='o', label='Min Actual')
        plt.scatter(actual_data.index[-1], actual_data.iloc[-1], color='purple', marker='o', label='Last Actual')
        plt.scatter(actual_data.index[-1], predictions[-1], color='yellow', marker='o', label='Last Predicted')

        plt.legend()
        plt.title(f'{self.ticker} Close Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.xticks(rotation=45)

        # Add a shaded area for prediction period
        plt.axvspan(actual_data.index[-1], actual_data.index[-1], color='lightblue', alpha=0.3, label='Prediction Period')

        plt.tight_layout()

        # Save plot
        plot_path = BASE_DIR / f'{self.PREDICTIONS_FOLDER}/{self.ticker}_prediction_plot.png'
        plt.savefig(plot_path, dpi=300)
        logger.info(f"Prediction plot saved at {plot_path}")

        plt.show()

    def run(self):
        """Run the prediction process."""
        try:
            predictions_scaled = self.predict()
            predictions = self.inverse_transform_predictions(predictions_scaled)
            self.plot_predictions(predictions)
            predictions_path = BASE_DIR / f'{self.PREDICTIONS_FOLDER}/{self.ticker}_predictions.csv'
            self.save_predictions(predictions, predictions_path)
        except Exception as e:
            logger.error(f"Error in prediction process: {e}", exc_info=True)


def main(ticker='BTC-USD'):
    """Main function to run the prediction script."""
    predictor = ModelPredictor(ticker)
    predictor.run()


if __name__ == '__main__':
    main()