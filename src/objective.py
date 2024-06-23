from pathlib import Path
import numpy as np
import optuna
from optuna.integration.tensorboard import TensorBoardCallback
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

from config import PARAMETERS
from logger import setup_logger
from train_model import train_model, ModelTrainer
from train_utils import save_best_params

BASE_DIR = Path(__file__).parent.parent
logger = setup_logger('objective_logger', BASE_DIR / 'logs', 'objective.log')


def objective(trial: optuna.trial.Trial) -> float:
    parameters = {
        'neurons': trial.suggest_int('neurons', 50, 300),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'additional_layers': trial.suggest_int('additional_layers', 0, 3),
        'bidirectional': trial.suggest_categorical('bidirectional', [True, False]),
        'l1_reg': trial.suggest_float('l1_reg', 1e-6, 1e-2),
        'l2_reg': trial.suggest_float('l2_reg', 1e-6, 1e-2),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
        'epochs': PARAMETERS['epochs'],
        'batch_size': PARAMETERS['batch_size'],
        'train_steps': PARAMETERS['train_steps'],
        'early_stopping_patience': PARAMETERS['early_stopping_patience'],
        'n_folds': PARAMETERS['n_folds']
    }

    try:
        trainer = ModelTrainer(ticker='BTC-USD')
    except Exception as e:
        logger.error(f"Failed to initialize ModelTrainer: {e}", exc_info=True)
        raise

    x, y = trainer.x, trainer.y
    tscv = TimeSeriesSplit(n_splits=parameters['n_folds'])
    splits = tscv.split(x)

    scores = []
    for i, (train_index, val_index) in enumerate(splits):
        x_train, x_val = x[train_index], x[val_index]
        y_train, y_val = y[train_index], y[val_index]

        try:
            model, history = train_model(
                x_train, y_train, x_val, y_val,
                model_dir=str(BASE_DIR / trainer.MODELS_FOLDER),
                ticker=trainer.ticker,
                fold_index=i,
                parameters=parameters
            )
        except Exception as e:
            logger.error(f"Failed to train model for fold {i}: {e}", exc_info=True)
            continue

        y_pred = model.predict(x_val)
        mse = mean_squared_error(y_val, y_pred)
        scores.append(mse)

    # Log metrics to TensorBoard
    trial.report(np.mean(scores), step=i)

    return float(np.mean(scores))


def optimize_hyperparameters(n_trials: int = 50) -> None:
    logger.info("Optimizing hyperparameters")
    tensorboard_log_dir = BASE_DIR / 'logs' / 'optuna'
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_callback = TensorBoardCallback(str(tensorboard_log_dir), metric_name='value')

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, callbacks=[tensorboard_callback])

    logger.info("Best trial:")
    trial = study.best_trial
    logger.info(f"  Value: {trial.value:.4f}")
    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")

    # Save the best parameters
    best_params_path = BASE_DIR / 'best_params.json'
    save_best_params(trial.params, best_params_path)
    logger.info(f"Best parameters saved at {best_params_path}")


if __name__ == '__main__':
    optimize_hyperparameters()
