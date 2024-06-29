from pathlib import Path
import numpy as np
import optuna
from optuna.integration.tensorboard import TensorBoardCallback
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from logger import setup_logger
from train_model import train_model, ModelTrainer
from train_utils import save_best_params

BASE_DIR = Path(__file__).parent.parent
logger = setup_logger('objective_logger', BASE_DIR / 'logs', 'objective.log')

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
HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([16, 32, 64, 128]))
HP_TRAIN_STEPS = hp.HParam('train_steps', hp.IntInterval(30, 180))
HP_EARLY_STOPPING_PATIENCE = hp.HParam('early_stopping_patience', hp.IntInterval(5, 20))

METRIC_MSE = 'mse'


def objective(trial: optuna.trial.Trial) -> float:
    parameters = {
        'neurons': trial.suggest_int('neurons', 50, 300),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'additional_layers': trial.suggest_int('additional_layers', 0, 3),
        'bidirectional': trial.suggest_categorical('bidirectional', [True, False]),
        'l1_reg': trial.suggest_float('l1_reg', 1e-6, 1e-2),
        'l2_reg': trial.suggest_float('l2_reg', 1e-6, 1e-2),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'epochs': trial.suggest_int('epochs', 50, 200),
        'batch_size': trial.suggest_int('batch_size', 16, 128),
        'train_steps': trial.suggest_int('train_steps', 30, 180),
        'early_stopping_patience': trial.suggest_int('early_stopping_patience', 5, 20),
        'n_folds': trial.suggest_int('n_folds', 3, 10)
    }

    logger.info(f"Starting trial {trial.number} with parameters: {parameters}")

    try:
        trainer = ModelTrainer(ticker='BTC-USD', parameters=parameters)
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
            model, history = train_model(
                x_train, y_train, x_val, y_val,
                model_dir=str(BASE_DIR / trainer.MODELS_FOLDER),
                ticker=trainer.ticker,
                fold_index=i,
                trial_id=trial_id,
                parameters=parameters
            )
        except Exception as e:
            logger.error(f"Failed to train model for fold {i} in trial {trial_id}: {e}", exc_info=True)
            continue

        y_pred = model.predict(x_val)
        mse = mean_squared_error(y_val, y_pred)
        scores.append(mse)
        last_step = i

        logger.info(f"Completed fold {i} for trial {trial_id} with MSE: {mse}")

    if scores:
        average_score = np.mean(scores)
        trial.report(average_score, step=last_step)
        logger.info(f"Trial {trial_id} completed with average MSE: {average_score}")

        # Log hyperparameters and metrics
        with tf.summary.create_file_writer(str(BASE_DIR / 'logs' / 'hparams')).as_default():
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
                HP_TRAIN_STEPS: parameters['train_steps'],
                HP_EARLY_STOPPING_PATIENCE: parameters['early_stopping_patience'],
                HP_N_FOLDS: parameters['n_folds']
            })
            tf.summary.scalar(METRIC_MSE, average_score, step=trial_id)

    return float(np.mean(scores)) if scores else float('inf')


def optimize_hyperparameters(n_trials: int = 50) -> None:
    logger.info("Optimizing hyperparameters")
    tensorboard_log_dir = BASE_DIR / 'logs' / 'optuna'
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    tensorboard_callback = TensorBoardCallback(str(tensorboard_log_dir), metric_name='value')

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, callbacks=[tensorboard_callback])

    logger.info("Hyperparameter optimization completed")
    logger.info("Best trial:")
    trial = study.best_trial
    logger.info(f"  Value: {trial.value:.4f}")
    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")

    # Save the best parameters
    best_params_path = BASE_DIR / 'best_params.json'
    save_best_params(trial.params, best_params_path, ticker)
    logger.info(f"Best parameters saved at {best_params_path}")


if __name__ == '__main__':
    optimize_hyperparameters()
