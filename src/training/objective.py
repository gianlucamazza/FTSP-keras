import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
from optuna.integration.tensorboard import TensorBoardCallback
from sklearn.model_selection import TimeSeriesSplit
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from typing import Tuple, Optional, Dict

# Ensure the project directory is in the sys.path
project_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_dir))

from src.logging.logger import setup_logger
from src.utils import save_to_json

# Setup logger
ROOT_DIR = project_dir
logger = setup_logger('objective_logger', 'logs', 'objective_logger.log')

# Define hyperparameters
HP_NEURONS = hp.HParam('neurons', hp.Discrete([50, 100, 150, 200, 250, 300]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.5))
HP_LAYERS = hp.HParam('additional_layers', hp.IntInterval(0, 3))
HP_BIDIRECTIONAL = hp.HParam('bidirectional', hp.Discrete([True, False]))
HP_L1_REG = hp.HParam('l1_reg', hp.RealInterval(1e-6, 1e-2))
HP_L2_REG = hp.HParam('l2_reg', hp.RealInterval(1e-6, 1e-2))
HP_LEARNING_RATE = hp.HParam('learning_rate', hp.RealInterval(1e-5, 1e-2))
HP_N_FOLDS = hp.HParam('n_folds', hp.IntInterval(3, 10))
HP_EPOCHS = hp.HParam('epochs', hp.IntInterval(50, 200))
HP_BATCH_SIZE = hp.HParam('batch_size', hp.IntInterval(16, 512))
HP_LAG_PARAMETER = hp.HParam('lag_parameter', hp.IntInterval(1, 20))
HP_EARLY_STOPPING_PATIENCE = hp.HParam('early_stopping_patience', hp.IntInterval(5, 20))

METRIC_MSE = 'mse'

from src.training.train_model import train_model, ModelTrainer

def preprocess_data(data_path, lag_parameter):
    df = pd.read_csv(data_path, index_col='Date', parse_dates=True)
    if df.isnull().values.any():
        raise ValueError("Missing values in Dataset")
    x, y = ModelTrainer.create_windowed_data(df, lag_parameter, 'Close')

    num_samples, num_timesteps, num_features = x.shape
    x = x.reshape(-1, num_features)
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    x = x.reshape(num_samples, num_timesteps, num_features)

    return x, y


def objective(trial: optuna.trial.Trial, ticker: str) -> float:
    parameters = {
        'neurons': trial.suggest_int('neurons', 60, 150),
        'dropout': trial.suggest_float('dropout', 0.01, 0.2),
        'additional_layers': trial.suggest_int('additional_layers', 1, 3),
        'bidirectional': trial.suggest_categorical('bidirectional', [True, False]),
        'l1_reg': trial.suggest_float('l1_reg', 1e-6, 1e-2),
        'l2_reg': trial.suggest_float('l2_reg', 1e-6, 1e-2),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'epochs': trial.suggest_int('epochs', 50, 200),
        'batch_size': trial.suggest_int('batch_size', 16, 512),
        'lag_parameter': trial.suggest_int('lag_parameter', 1, 10),
        'early_stopping_patience': trial.suggest_int('early_stopping_patience', 5, 20),
        'n_folds': trial.suggest_int('n_folds', 3, 10)
    }

    logger.info(f"Starting trial {trial.number} with parameters: {parameters}")

    data_path = ROOT_DIR / f'data/scaled_data_{ticker}.csv'
    x, y = preprocess_data(data_path, parameters['lag_parameter'])
    

    input_shape = (x.shape[1], x.shape[2])

    try:
        trainer = ModelTrainer(ticker=ticker, params=parameters)
    except Exception as e:
        logger.error(f"Failed to initialize ModelTrainer: {e}", exc_info=True)
        raise

    x, y = trainer.x, trainer.y
    tscv = TimeSeriesSplit(n_splits=parameters['n_folds'])
    splits = list(tscv.split(x))

    scores = []
    trial_id = trial.number
    last_step = 0

    for i, (train_index, val_index) in enumerate(splits):
        logger.info(f"Starting fold {i} for trial {trial_id}")
        x_train, x_val = x[train_index], x[val_index]
        y_train, y_val = y[train_index], y[val_index]

        try:
            model, history, val_loss = train_model(
                x_train, y_train, x_val, y_val,
                model_dir=str(ROOT_DIR / trainer.MODELS_FOLDER / trainer.ticker),
                ticker=trainer.ticker,
                fold_index=i,
                trial_id=trial_id,
                input_shape=input_shape,
                params=parameters
            )
        except Exception as e:
            logger.error(f"Failed to train model for fold {i} in trial {trial_id}: {e}", exc_info=True)
            continue

        scores.append(val_loss)
        last_step = i

        logger.info(f"Completed fold {i} for trial {trial_id} with MSE: {val_loss}")

    if scores:
        average_score = np.mean(scores)
        trial.report(average_score, step=last_step)
        logger.info(f"Trial {trial_id} completed with average MSE: {average_score}")

        # Log hyperparameters and metrics
        with tf.summary.create_file_writer(str(ROOT_DIR / 'logs' / 'hparams')).as_default():
            hp.hparams({
                HP_NEURONS: parameters['neurons'],
                HP_DROPOUT: parameters['dropout'],
                HP_LAYERS: parameters['additional_layers'],
                HP_BIDIRECTIONAL: parameters['bidirectional'],
                HP_L1_REG: parameters['l1_reg'],
                HP_L2_REG: parameters['l2_reg'],
                HP_LEARNING_RATE: parameters['learning_rate'],
                HP_EPOCHS: parameters['epochs'],
                HP_BATCH_SIZE: parameters['batch_size'],
                HP_LAG_PARAMETER: parameters['lag_parameter'],
                HP_EARLY_STOPPING_PATIENCE: parameters['early_stopping_patience'],
                HP_N_FOLDS: parameters['n_folds']
            })
            tf.summary.scalar(METRIC_MSE, average_score, step=trial_id)

    return float(np.mean(scores)) if scores else float('inf')


def optimize_hyperparameters(ticker: str, n_trials: int = 50) -> Dict:
    best_params_path = ROOT_DIR / f'{ticker}_best_params.json'
    logger.info("Optimizing hyperparameters")
    tensorboard_log_dir = ROOT_DIR / 'logs' / 'optuna'
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    tensorboard_callback = TensorBoardCallback(str(tensorboard_log_dir), metric_name='value')

    study_name = f'{ticker}_study'
    study = optuna.create_study(direction='minimize', study_name=study_name)

    data_path = ROOT_DIR / f'data/scaled_data_{ticker}.csv'
    df = pd.read_csv(data_path, index_col='Date', parse_dates=True)
    train_steps = 30  # FIXME: replace with dynamic value (?)
    x, _ = ModelTrainer.create_windowed_data(df, train_steps, 'Close')
    input_shape = (x.shape[1], x.shape[2])

    study.optimize(lambda t: objective(t, ticker, input_shape), n_trials=n_trials,
                   callbacks=[tensorboard_callback])

    logger.info("Hyperparameter optimization completed")
    logger.info("Best trial:")
    best_trial = study.best_trial
    logger.info(f"  Value: {best_trial.value:.4f}")
    logger.info("  Params: ")
    for key, value in best_trial.params.items():
        logger.info(f"    {key}: {value}")

    # Save the best parameters
    save_to_json(best_trial.params, best_params_path)
    logger.info(f"Best parameters saved at {best_params_path}")

    return best_trial.params


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Search hyperparams')
    parser.add_argument('--ticker', type=str, required=True, help='Ticker')

    args = parser.parse_args()
    params_path = ROOT_DIR / f'{args.ticker}_best_params.json'
    best_params = optimize_hyperparameters(args.ticker, 50)

    save_to_json(best_params, params_path)
