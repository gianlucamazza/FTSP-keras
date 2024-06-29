import sys
from pathlib import Path
from tensorflow.keras import mixed_precision
import numpy as np
import logger as logger_module
from train_utils import load_best_params, save_best_params, calculate_metrics
from train_model import train_model, ModelTrainer
from objective import optimize_hyperparameters
from typing import Any, Optional

# Add the project directory to the sys.path for module resolution
project_dir = Path(__file__).resolve().parent
sys.path.append(str(project_dir))

BASE_DIR = Path(__file__).parent.parent
logger = logger_module.setup_logger('train_logger', BASE_DIR / 'logs', 'train.log')

# Enable mixed precision training for performance improvement
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)


def main(ticker: str = 'BTC-USD', worker: Optional[Any] = None, hyperparameters_file: str = 'best_params.json', trial_id: Optional[int] = None):
    """
    Main function to train the model.

    Args:
        ticker (str): The ticker symbol for the financial instrument.
        worker (optional): A worker object to manage early stopping.
        hyperparameters_file (str): Path to the parameters file.
        trial_id (int, optional): The trial ID from hyperparameter optimization.
    """

    hyperparameters_path = BASE_DIR / hyperparameters_file
    hyperparameters = load_best_params(hyperparameters_path)

    if not hyperparameters:
        logger.info("Optimizing hyperparameters...")
        optimize_hyperparameters(n_trials=50)
        hyperparameters = load_best_params(hyperparameters_path)
        logger.info(f"Optimized hyperparameters: {hyperparameters}")
        save_best_params(hyperparameters, hyperparameters_path)
    else:
        logger.info("Using existing hyperparameters.")

    logger.info(f"Starting training process for {ticker} with hyperparameters: {hyperparameters}")

    # Initialize ModelTrainer with hyperparameters
    trainer = ModelTrainer(ticker, parameters=hyperparameters)

    # Set up cross-validation with TimeSeriesSplit
    n_splits = hyperparameters['n_folds']
    split_ratio = 0.1
    splits = []

    logger.info(f"Number of splits: {n_splits}")

    for i in range(n_splits):
        # Create a rolling window split
        train_size = int((1 - split_ratio) * len(trainer.x))
        test_size = len(trainer.x) - train_size

        train_index = np.arange(0, train_size)
        test_index = np.arange(train_size, len(trainer.x))

        splits.append((train_index, test_index))

    logger.info(f"Number of splits: {n_splits}")

    best_val_loss = np.inf
    best_model = None
    metrics_list = []

    for i, (train_index, test_index) in enumerate(splits):
        if worker is not None and hasattr(worker, 'is_running') and not worker.is_running:
            logger.info("Training stopped early.")
            return

        train_size = len(train_index)
        val_size = len(test_index)
        total_size = train_size + val_size
        train_percentage = (train_size / total_size) * 100
        val_percentage = (val_size / total_size) * 100

        logger.info(
            f"Fold {i + 1}/{n_splits} - Training data: {train_percentage:.2f}%, Validation data: {val_percentage:.2f}%")

        # Log the training and validation dates for each fold
        train_dates = trainer.df.index[train_index]
        val_dates = trainer.df.index[test_index]
        logger.info(
            f"Fold {i + 1}/{n_splits} - Train dates: {train_dates[0]} to {train_dates[-1]}, "
            f"Val dates: {val_dates[0]} to {val_dates[-1]}")

        logger.info(f"Training fold {i + 1}/{n_splits}")

        x_train, x_test = trainer.x[train_index], trainer.x[test_index]
        y_train, y_test = trainer.y[train_index], trainer.y[test_index]

        try:
            model, history = train_model(
                x_train, y_train, x_test, y_test,
                model_dir=str(BASE_DIR / trainer.MODELS_FOLDER),
                ticker=trainer.ticker,
                fold_index=i,
                trial_id=trial_id if trial_id is not None else 0,
                parameters=hyperparameters,
                worker=worker
            )
        except Exception as e:
            logger.error(f"Training failed for fold {i}: {e}", exc_info=True)
            continue

        if model is None:
            logger.error("Model training failed.")
            return

        # Validate the model
        val_loss = min(history.history['val_loss'])
        logger.info(f"Fold {i + 1}/{n_splits} - Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            logger.info(f"New best model found for fold {i + 1} with validation loss {val_loss:.4f}")

        # Calculate and log metrics
        rmse, mae, mape = calculate_metrics(model, x_test, y_test)
        logger.info(f"Fold {i + 1}/{n_splits} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.4f}")
        metrics_list.append((rmse, mae, mape))

    if best_model:
        best_model_path = BASE_DIR / trainer.MODELS_FOLDER / f"model_{trainer.ticker}_best.keras"
        best_model.save(best_model_path)
        logger.info(f"Best model saved at {best_model_path}")

    if metrics_list:
        valid_metrics = [(rmse, mae, mape) for rmse, mae, mape in metrics_list if
                         np.isfinite(rmse) and np.isfinite(mae) and np.isfinite(mape)]
        if valid_metrics:
            average_rmse: float = np.mean([m[0] for m in valid_metrics])
            average_mae: float = np.mean([m[1] for m in valid_metrics])
            average_mape: float = np.mean([m[2] for m in valid_metrics])
            logger.info(
                f"Average metrics across all folds - RMSE: {average_rmse:.4f}, MAE: {average_mae:.4f}, MAPE: {average_mape:.4f}")
        else:
            logger.warning("No valid metrics to average.")

    logger.info("Training process completed.")


if __name__ == '__main__':
    main()