import sys
import pandas as pd
import numpy as np
import joblib
import time
import optuna
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.regularizers import l1_l2
from model import build_model, prepare_callbacks
from data_utils import prepare_data
import logger as logger_module
from config import COLUMN_SETS, CLOSE, PARAMETERS

# Add the project directory to the sys.path
project_dir = Path(__file__).resolve().parent
sys.path.append(str(project_dir))

BASE_DIR = Path(__file__).parent.parent
logger = logger_module.setup_logger('train_logger', BASE_DIR / 'logs', 'train.log')


class ModelTrainer:
    COLUMN_TO_PREDICT = CLOSE
    DATA_FOLDER = 'data'
    SCALERS_FOLDER = 'scalers'
    MODELS_FOLDER = 'models'

    def __init__(self, ticker='BTC-USD'):
        self.ticker = ticker
        self.data_path = BASE_DIR / f'{self.DATA_FOLDER}/scaled_data_{self.ticker}.csv'
        self.scaler_path = BASE_DIR / f'{self.SCALERS_FOLDER}/feature_scaler_{self.ticker}.pkl'

        logger.info(f"Initializing ModelTrainer for ticker {ticker}")
        logger.info(f"Data path: {self.data_path}")
        logger.info(f"Scaler path: {self.scaler_path}")

        self.feature_scaler = self.load_scaler()
        self.df = self.load_dataset()
        self.df = prepare_data(self.df, self.feature_scaler)
        self.x, self.y = self.create_windowed_data(self.df, PARAMETERS['train_steps'], self.COLUMN_TO_PREDICT)

    def load_scaler(self):
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

    def load_dataset(self):
        """Load the dataset from disk."""
        logger.info(f"Loading dataset from {self.data_path}")
        try:
            df = pd.read_csv(self.data_path, index_col='Date')
            df.ffill(inplace=True)
            return df
        except FileNotFoundError as e:
            logger.error(f"Dataset file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading dataset: {e}", exc_info=True)
            raise

    @staticmethod
    def create_windowed_data(df, steps, target_column):
        """Create windowed data for LSTM."""
        x, y = [], []
        target_index = df.columns.get_loc(target_column)
        data = df.values
        for i in range(steps, len(data)):
            x.append(data[i - steps:i])
            y.append(data[i, target_index])
        return np.array(x), np.array(y)


def train_model(x_train, y_train, x_val, y_val, model_dir, ticker, fold_index, parameters, worker=None):
    """Train the LSTM model."""
    logger.info(f"Starting training for fold {fold_index}...")
    start_time = time.time()

    model = build_model(
        input_shape=(parameters['train_steps'], len(COLUMN_SETS['to_scale'])),
        neurons=parameters['neurons'],
        dropout=parameters['dropout'],
        additional_layers=parameters['additional_layers'],
        bidirectional=parameters['bidirectional'],
        regularizer=l1_l2(parameters.get('l1_reg', 1e-5), parameters.get('l2_reg', 1e-5))
    )

    # Callbacks
    model_checkpoint = ModelCheckpoint(
        filepath=str(Path(model_dir) / f"model_{ticker}_fold_{fold_index}.keras"),
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=parameters['early_stopping_patience'],
        verbose=1,
        restore_best_weights=True
    )

    tensorboard_callback = TensorBoard(log_dir=str(BASE_DIR / 'logs' / f"fold_{fold_index}"), histogram_freq=1)

    callbacks = [model_checkpoint, early_stopping, tensorboard_callback]

    history = model.fit(
        x_train, y_train,
        epochs=parameters['epochs'],
        batch_size=parameters['batch_size'],
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    logger.info(f"Training completed for fold {fold_index}.")
    model_path = Path(model_dir) / f"model_{ticker}_fold_{fold_index}.keras"
    model.save(model_path)
    logger.info(f"Model saved at {model_path}")

    end_time = time.time()
    logger.info(f"Training for fold {fold_index} completed in {end_time - start_time:.2f} seconds.")
    return model, history


def calculate_metrics(model, x_test, y_test):
    """Calculate evaluation metrics for the model."""
    y_pred = model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    logger.info(f"Evaluation metrics - RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.4f}")
    return rmse, mae, mape


def objective(trial):
    parameters = {
        'neurons': trial.suggest_int('neurons', 50, 200),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'additional_layers': trial.suggest_int('additional_layers', 0, 2),
        'bidirectional': trial.suggest_categorical('bidirectional', [True, False]),
        'l1_reg': trial.suggest_float('l1_reg', 1e-6, 1e-2),
        'l2_reg': trial.suggest_float('l2_reg', 1e-6, 1e-2),
        'epochs': PARAMETERS['epochs'],
        'batch_size': PARAMETERS['batch_size'],
        'train_steps': PARAMETERS['train_steps'],
        'early_stopping_patience': PARAMETERS['early_stopping_patience'],
        'n_folds': PARAMETERS['n_folds']
    }

    trainer = ModelTrainer(ticker='BTC-USD')
    x, y = trainer.x, trainer.y

    tscv = TimeSeriesSplit(n_splits=parameters['n_folds'])
    splits = tscv.split(x)

    scores = []
    for i, (train_index, val_index) in enumerate(splits):
        x_train, x_val = x[train_index], x[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model, history = train_model(
            x_train, y_train, x_val, y_val,
            model_dir=str(BASE_DIR / trainer.MODELS_FOLDER),
            ticker=trainer.ticker,
            fold_index=i,
            parameters=parameters
        )

        y_pred = model.predict(x_val)
        mse = mean_squared_error(y_val, y_pred)
        scores.append(mse)

    return np.mean(scores)


def optimize_hyperparameters(n_trials=20):
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


def main(ticker='BTC-USD', worker=None, parameters=None):
    """Main function to train the model."""
    if parameters is None:
        parameters = PARAMETERS

    logger.info(f"Starting training process for {ticker} with parameters: {parameters}")

    optimize_hyperparameters(20)

    trainer = ModelTrainer(ticker)

    n_splits = parameters['n_folds']
    if n_splits > 1:
        tscv = TimeSeriesSplit(n_splits=n_splits)
        splits = tscv.split(trainer.x)
    else:
        split_index = int(len(trainer.x) * 0.8)
        splits = [(np.arange(split_index), np.arange(split_index, len(trainer.x)))]

    logger.info(f"Number of splits: {n_splits}")

    best_val_loss = np.inf
    best_model = None
    metrics_list = []

    for i, (train_index, test_index) in enumerate(splits):
        if worker is not None and hasattr(worker, 'is_running') and not worker.is_running:
            logger.info("Training stopped early.")
            return

        logger.info(f"Training fold {i + 1}/{n_splits}")

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

        rmse, mae, mape = calculate_metrics(model, x_test, y_test)
        metrics_list.append((rmse, mae, mape))

    if best_model:
        best_model_path = BASE_DIR / trainer.MODELS_FOLDER / f"model_{trainer.ticker}_best.keras"
        best_model.save(best_model_path)
        logger.info(f"Best model saved at {best_model_path}")

    if metrics_list:
        average_rmse = np.mean([m[0] for m in metrics_list])
        average_mae = np.mean([m[1] for m in metrics_list])
        average_mape = np.mean([m[2] for m in metrics_list])
        logger.info(
            f"Average metrics across all folds - RMSE: {average_rmse:.2f}, MAE: {average_mae:.2f}, MAPE: {average_mape:.2f}")

    logger.info("Training process completed.")


if __name__ == '__main__':
    main()
