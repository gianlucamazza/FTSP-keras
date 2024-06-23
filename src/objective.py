import optuna
from optuna.integration.tensorboard import TensorBoardCallback
import numpy as np
from pathlib import Path
from keras.src.losses import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from train_model import train_model, ModelTrainer
from config import PARAMETERS
from train_utils import calculate_metrics, save_best_params
import logger as logger_module

BASE_DIR = Path(__file__).parent.parent
logger = logger_module.setup_logger('objective_logger', BASE_DIR / 'logs', 'objective.log')


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
    tensorboard_log_dir = BASE_DIR / 'logs' / 'optuna'
    tensorboard_callback = TensorBoardCallback(str(tensorboard_log_dir), metric_name='mse')
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, callbacks=[tensorboard_callback])

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Save the best parameters
    best_params_path = BASE_DIR / 'best_params.json'
    save_best_params(trial.params, best_params_path)
    logger.info(f"Best parameters saved at {best_params_path}")
