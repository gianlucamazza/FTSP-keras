import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

# Add the project directory to the sys.path
project_dir = Path(__file__).resolve().parent
sys.path.append(str(project_dir))

from model import build_model, prepare_callbacks
import logger as logger
from config import COLUMN_SETS

PARAMETERS = {
    'neurons': 100,
    'dropout': 0.3,
    'additional_layers': 2,
    'bidirectional': True,
    'epochs': 50,
    'batch_size': 32,
    'train_steps': 60,
    'test_steps': 30
}

BASE_DIR = Path(__file__).parent.parent
logger = logger.setup_logger('train_logger', BASE_DIR / 'logs', 'train.log')


class ModelTrainer:
    COLUMN_TO_PREDICT = 'Close'
    DATA_FOLDER = 'data'
    SCALERS_FOLDER = 'scalers'
    MODELS_FOLDER = 'models'

    def __init__(self, ticker='BTC-USD'):
        self.ticker = ticker
        self.data_path = BASE_DIR / f'{self.DATA_FOLDER}/scaled_data_{self.ticker}.csv'
        self.feature_scaler = self.load_scaler()
        self.df = self.load_dataset()
        self.x, self.y = None, None

    def load_scaler(self):
        try:
            return joblib.load(BASE_DIR / f'{self.SCALERS_FOLDER}/feature_scaler_{self.ticker}.pkl')
        except FileNotFoundError as e:
            logger.error(f"Scaler file not found: {e}")
            raise

    def load_dataset(self):
        try:
            df = pd.read_csv(self.data_path, index_col='Date')
            df.ffill(inplace=True)
            return df
        except Exception as e:
            logger.error(f"Error loading dataset: {e}", exc_info=True)
            raise

    def prepare_data(self, parameters):
        missing_columns = set(COLUMN_SETS['to_scale']) - set(self.df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns in the DataFrame: {missing_columns}")

        scaler_columns = COLUMN_SETS['to_scale']
        self.df = self.df.reindex(columns=scaler_columns)
        self.df[scaler_columns] = self.feature_scaler.transform(self.df[scaler_columns])
        self.x, self.y = create_windowed_data(self.df[scaler_columns].values, parameters['train_steps'])


def create_windowed_data(df, steps):
    x, y = [], []
    for i in range(steps, len(df)):
        x.append(df[i - steps:i])
        y.append(df[i, 0])
    return np.array(x), np.array(y)


def train_model(x_train, y_train, x_val, y_val, model_dir, ticker, fold_index, parameters, worker=None):
    model = build_model(
        input_shape=(parameters['train_steps'], len(COLUMN_SETS['to_scale'])),
        neurons=parameters['neurons'],
        dropout=parameters['dropout'],
        additional_layers=parameters['additional_layers'],
        bidirectional=parameters['bidirectional']
    )
    callbacks = prepare_callbacks(model_dir, f"{ticker}_fold_{fold_index}")

    for epoch in range(parameters['epochs']):
        if worker is not None and not worker._is_running:
            logger.info("Training stopped early.")
            return None
        history = model.fit(
            x_train, y_train, epochs=1, batch_size=parameters['batch_size'],
            validation_data=(x_val, y_val), callbacks=callbacks, verbose=1
        )
    
    model_path = Path(model_dir) / f"model_{ticker}_fold_{fold_index}.keras"
    model.save(model_path)
    logger.info(f"Fold {fold_index} model saved at {model_path}")
    return model, history


def calculate_rmse(model, x_test, y_test):
    y_pred = model.predict(x_test)
    return np.sqrt(mean_squared_error(y_test, y_pred))


def main(ticker='BTC-USD', worker=None, parameters=None):
    if parameters is None:
        parameters = PARAMETERS

    trainer = ModelTrainer(ticker)
    trainer.prepare_data(parameters)

    n_splits = max(1, (len(trainer.df) - parameters['train_steps']) // parameters['test_steps'])
    tscv = TimeSeriesSplit(n_splits=n_splits)

    logger.info(f"Number of splits: {n_splits}")

    best_val_loss = np.inf
    best_model = None
    rmse_list = []

    for i, (train_index, test_index) in enumerate(tscv.split(trainer.x)):
        if worker is not None and not worker._is_running:
            logger.info("Training stopped early.")
            return
        percent_complete = (i / tscv.n_splits) * 100
        logger.info(f"Training fold {i + 1}/{tscv.n_splits} ({percent_complete:.2f}% complete)")

        x_train, x_test = trainer.x[train_index], trainer.x[test_index]
        y_train, y_test = trainer.y[train_index], trainer.y[test_index]

        model, history = train_model(
            x_train, y_train, x_test, y_test,
            model_dir=str(BASE_DIR / trainer.MODELS_FOLDER),
            ticker=trainer.ticker,
            fold_index=i,
            parameters=parameters,
            worker=worker
        )

        if model is None:
            return

        current_val_loss = min(history.history['val_loss'])
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_model = model

    if best_model:
        best_model_path = BASE_DIR / trainer.MODELS_FOLDER / f"model_{trainer.ticker}_best.keras"
        best_model.save(best_model_path)
        logger.info(f"Best model saved at {best_model_path}")

    for _, test_index in tscv.split(trainer.x):
        if worker is not None and not worker._is_running:
            logger.info("Evaluation stopped early.")
            return
        x_test = trainer.x[test_index]
        y_test = trainer.y[test_index]
        rmse = calculate_rmse(best_model, x_test, y_test)
        if rmse is not None:
            rmse_list.append(rmse)

    if rmse_list:
        average_rmse = np.mean(rmse_list)
        logger.info(f"Average RMSE across all folds: {average_rmse:.2f}")


if __name__ == '__main__':
    main()
