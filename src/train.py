# train.py
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
from model import build_model, prepare_callbacks
import logger as logger
from data_preparation import COLUMN_SETS

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
        self.x = None
        self.y = None
        self.ticker = ticker
        self.data_path = Path(BASE_DIR / f'{self.DATA_FOLDER}/scaled_data_{self.ticker}.csv')
        self.model_path = Path(BASE_DIR / f'{self.MODELS_FOLDER}/model_{self.ticker}.keras')
        self.feature_scaler = self.load_scaler()
        self.df = self.load_dataset()

    def load_scaler(self):
        return joblib.load(BASE_DIR / f'{self.SCALERS_FOLDER}/feature_scaler_{self.ticker}.pkl')

    def load_dataset(self):
        try:
            return pd.read_csv(self.data_path, index_col='Date')
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
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


def train_model(x_train, y_train, x_val, y_val, model_dir, ticker, parameters):
    build_model_params = {
        'input_shape': (parameters['train_steps'], len(COLUMN_SETS['to_scale'])),
        'neurons': parameters['neurons'],
        'dropout': parameters['dropout'],
        'additional_layers': parameters['additional_layers'],
        'bidirectional': parameters['bidirectional']
    }

    model = build_model(**build_model_params)
    callbacks = prepare_callbacks(model_dir, ticker)
    history = model.fit(
        x_train, y_train, epochs=parameters['epochs'], batch_size=parameters['batch_size'],
        validation_data=(x_val, y_val), callbacks=callbacks, verbose=1
    )
    return model, history


def calculate_rmse(model, x_test, y_test):
    y_pred = model.predict(x_test)
    return np.sqrt(mean_squared_error(y_test, y_pred))


def plot_history(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.legend()
    plt.show()


def main(ticker='BTC-USD', parameters=None):
    if parameters is None:
        parameters = PARAMETERS

    trainer = ModelTrainer(ticker)
    trainer.df.ffill(inplace=True)
    trainer.prepare_data(parameters)

    tscv = TimeSeriesSplit(n_splits=(len(trainer.df) - parameters['train_steps']) // parameters['test_steps'])
    best_val_loss = np.inf
    best_model = None
    rmse_list = []
    history_list = []

    for i, (train_index, test_index) in enumerate(tscv.split(trainer.x)):
        percent_complete = (i / tscv.n_splits) * 100
        logger.info(f"Training fold {i + 1}/{tscv.n_splits} ({percent_complete:.2f}% complete)")

        x_train, x_test = trainer.x[train_index], trainer.x[test_index]
        y_train, y_test = trainer.y[train_index], trainer.y[test_index]

        model, history = train_model(
            x_train, y_train, x_test, y_test,
            model_dir=str(trainer.model_path.parent),
            ticker=trainer.ticker,
            parameters=parameters
        )
        history_list.append(history)  # Aggiungi la storia a history_list

        current_val_loss = min(history.history['val_loss'])
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_model = model

    if best_model:
        best_model_path = BASE_DIR / trainer.model_path
        best_model.save(best_model_path)
        logger.info(f"Best model saved at {best_model_path}")

    for _, test_index in tscv.split(trainer.x):
        x_test = trainer.x[test_index]
        y_test = trainer.y[test_index]
        rmse = calculate_rmse(best_model, x_test, y_test)
        if rmse is not None:
            rmse_list.append(rmse)

    if rmse_list:
        average_rmse = np.mean(rmse_list)
        logger.info(f"Average RMSE across all folds: {average_rmse:.2f}")

    plot_history(np.mean(history_list, axis=0))


if __name__ == '__main__':
    main()
