# train.py
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from model import build_model
from logger import setup_logger
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

logger = setup_logger('train_logger', 'logs', 'train.log')


class ModelTrainer:
    COLUMN_TO_PREDICT = 'Close'
    DATA_FOLDER = 'data'
    SCALERS_FOLDER = 'scalers'
    MODELS_FOLDER = 'models'

    def __init__(self, ticker='BTC-USD'):
        self.ticker = ticker
        self.data_path = Path(f'{self.DATA_FOLDER}/scaled_data_{self.ticker}.csv')
        self.model_path = Path(f'{self.MODELS_FOLDER}/model_{self.ticker}.keras')
        self.feature_scaler, self.close_scaler = self.load_scalers()
        self.df = self.load_dataset()

    def load_scalers(self):
        feature_scaler = joblib.load(f'{self.SCALERS_FOLDER}/feature_scaler_{self.ticker}.pkl')
        close_scaler = joblib.load(f'{self.SCALERS_FOLDER}/close_scaler_{self.ticker}.pkl')
        return feature_scaler, close_scaler

    def load_dataset(self):
        try:
            df = pd.read_csv(self.data_path, index_col='Date')
            df.ffill(inplace=True)
            logger.info(f"Loaded dataset shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

    def prepare_data(self, parameters):
        missing_columns = set(COLUMN_SETS['to_scale']) - set(self.df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns in the DataFrame: {missing_columns}")

        self.df[self.COLUMN_TO_PREDICT] = self.close_scaler.transform(self.df[[self.COLUMN_TO_PREDICT]])

        scaler_columns = [col for col in COLUMN_SETS['to_scale'] if col in self.df.columns]
        self.df = self.df.reindex(columns=scaler_columns)
        self.df[scaler_columns] = self.feature_scaler.transform(self.df[scaler_columns])

        self.x, self.y = create_windowed_data(self.df[scaler_columns].values, parameters['train_steps'])
        self.y = self.y.reshape(-1, 1)


def create_windowed_data(df, steps):
    x, y = [], []
    for i in range(steps, len(df)):
        x.append(df[i - steps:i])
        y.append(df[i, 0])
    return np.array(x), np.array(y)


def train_model(x_train, y_train, x_val, y_val, model_path, parameters):
    build_model_params = {
        'input_shape': (parameters['train_steps'], len(COLUMN_SETS['to_scale'])),
        'neurons': parameters['neurons'],
        'dropout': parameters['dropout'],
        'additional_layers': parameters['additional_layers'],
        'bidirectional': parameters['bidirectional']
    }

    model = build_model(**build_model_params)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    model_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True)
    history = model.fit(x_train, y_train, epochs=parameters['epochs'], batch_size=parameters['batch_size'],
                        validation_data=(x_val, y_val), callbacks=[early_stopping, model_checkpoint], verbose=1)
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
    trainer.prepare_data(parameters)

    tscv = TimeSeriesSplit(n_splits=(len(trainer.df) - parameters['train_steps']) // parameters['test_steps'])
    rmse_list, history = [], None

    for i, (train_index, test_index) in enumerate(tscv.split(trainer.x)):
        percent_complete = (i / tscv.n_splits) * 100
        logger.info(f"Training fold {i + 1}/{tscv.n_splits} ({percent_complete:.2f}% complete)")

        x_train, x_test = trainer.x[train_index], trainer.x[test_index]
        y_train, y_test = trainer.y[train_index], trainer.y[test_index]

        model, history = train_model(x_train, y_train, x_test, y_test, str(trainer.model_path), parameters)
        best_model = load_model(str(trainer.model_path))
        rmse = calculate_rmse(best_model, x_test, y_test)
        rmse_list.append(rmse)
        print(f"RMSE for fold {i + 1}: {rmse:.2f}")

    average_rmse = np.mean(rmse_list)
    logger.info(f"Average RMSE across all folds: {average_rmse:.2f}")
    plot_history(history)


if __name__ == '__main__':
    main()
