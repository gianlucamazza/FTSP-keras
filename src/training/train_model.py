import sys
import time
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from typing import Tuple, Optional, Dict

from src.training.objective import optimize_hyperparameters

# Ensure the project directory is in the sys.path
project_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_dir))

from src.config import COLUMN_SETS
from src.logging.logger import setup_logger
from src.data.data_utils import prepare_data
from src.models.model_builder import build_model
from src.models.callbacks import prepare_callbacks
from src.training.train_utils import load_best_params

# Setup logger
logger = setup_logger('train_model', 'logs', 'train_model.log')


def train_model(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, model_dir: str,
                ticker: str, fold_index: int, params: Dict, trial_id: int) -> Tuple[Model, dict]:
    """Train the LSTM model."""
    logger.info(f"Starting training for fold {fold_index} in trial {trial_id}...")
    start_time = time.time()

    model = build_model(
        input_shape=(params['train_steps'], len(COLUMN_SETS['to_scale'])),
        neurons=params['neurons'],
        dropout=params['dropout'],
        additional_layers=params['additional_layers'],
        bidirectional=params['bidirectional'],
        l1_reg=params.get('l1_reg', 1e-5),
        l2_reg=params.get('l2_reg', 1e-5),
        optimizer='adam'
    )

    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mean_squared_error']
    )

    model_dir_path = Path(model_dir)
    model_dir_path.mkdir(parents=True, exist_ok=True)

    callbacks = prepare_callbacks(model_dir=model_dir_path, ticker=ticker, monitor='val_loss', epoch=0)

    try:
        history = model.fit(
            x_train, y_train,
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            validation_data=(x_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        raise

    logger.info(f"Training completed for fold {fold_index} in trial {trial_id}.")
    model_path = model_dir_path / f"model_{ticker}_trial_{trial_id}_fold_{fold_index}.h5"  # Use .h5 extension
    model.save(model_path)
    logger.info(f"Model saved at {model_path}")

    end_time = time.time()
    logger.info(f"Training for fold {fold_index} in trial {trial_id} completed in {end_time - start_time:.2f} seconds.")
    return model, history


class ModelTrainer:
    COLUMN_TO_PREDICT = 'Close'
    DATA_FOLDER = 'data'
    SCALERS_FOLDER = 'scalers'
    MODELS_FOLDER = 'models'

    def __init__(self, ticker: str = 'BTC', params: Optional[Dict] = None):
        self.ticker = ticker
        self.parameters = params
        self.data_path = f'{self.DATA_FOLDER}/scaled_data_{self.ticker}.csv'
        self.scaler_path = f'{self.SCALERS_FOLDER}/feature_scaler_{self.ticker}.pkl'

        logger.info(f"Initializing ModelTrainer for ticker {ticker}")
        logger.info(f"Data path: {self.data_path}")
        logger.info(f"Scaler path: {self.scaler_path}")

        self.feature_scaler = self.load_scaler()
        self.df = self.load_dataset()
        self.df = prepare_data(self.df, self.feature_scaler)
        self.x, self.y = self.create_windowed_data(self.df, self.parameters['train_steps'], self.COLUMN_TO_PREDICT)

    def load_scaler(self) -> object:
        """Load the feature scaler from disk."""
        logger.info(f"Loading scaler from {self.scaler_path}")
        try:
            return joblib.load(self.scaler_path)
        except FileNotFoundError as e:
            logger.error(f"Scaler file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading scaler: {e}", exc_info=True)
            raise

    def load_dataset(self) -> pd.DataFrame:
        """Load the dataset from disk."""
        logger.info(f"Loading dataset from {self.data_path}")
        try:
            df = pd.read_csv(self.data_path, index_col='Date')
            df.ffill(inplace=True)
            if self.COLUMN_TO_PREDICT not in df.columns:
                raise ValueError(f"Column {self.COLUMN_TO_PREDICT} not found in dataset.")
            return df
        except FileNotFoundError as e:
            logger.error(f"Dataset file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading dataset: {e}", exc_info=True)
            raise

    @staticmethod
    def create_windowed_data(df: pd.DataFrame, steps: int, target_column: str) -> Tuple[np.ndarray, np.ndarray]:
        """Create windowed data for LSTM."""
        x, y = [], []
        target_index = df.columns.get_loc(target_column)
        data = df.values
        for i in range(steps, len(data)):
            x.append(data[i - steps:i])
            y.append(data[i, target_index])
        return np.array(x), np.array(y)


def main(ticker: str, params: Dict) -> None:
    """Main function to start the model training."""
    logger.info(f"Starting model training for ticker {ticker}")
    trainer = ModelTrainer(ticker=ticker, params=params)

    # Splitting data into training and validation sets
    split_index = int(len(trainer.x) * 0.8)
    x_train, y_train = trainer.x[:split_index], trainer.y[:split_index]
    x_val, y_val = trainer.x[split_index:], trainer.y[split_index:]

    logger.debug(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    logger.debug(f"x_val shape: {x_val.shape}, y_val shape: {y_val.shape}")

    model_dir = ModelTrainer.MODELS_FOLDER
    model, history = train_model(x_train, y_train, x_val, y_val, str(model_dir), ticker, 0, params, 0)

    # Utilize model and history if needed
    logger.info(f"Model training for ticker {ticker} completed.")
    logger.info(f"Final training loss: {history.history['loss'][-1]}")
    logger.info(f"Final validation loss: {history.history['val_loss'][-1]}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train LSTM model for financial prediction')
    parser.add_argument('--ticker', type=str, required=True, help='Ticker symbol')

    args = parser.parse_args()
    params_path = Path(f'{args.ticker}_best_params.json')
    params = load_best_params(args.ticker, params_path)

    if params is None:
        logger.error(f"Parameters file not found: {params_path}")
        logger.info("Starting hyperparameter optimization...")
        params = optimize_hyperparameters(ticker=args.ticker)

    main(ticker=args.ticker, params=params)
