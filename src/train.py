import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import TimeSeriesSplit

import logger as logger_module
from train_utils import load_best_params, calculate_metrics
from train_model import train_model, ModelTrainer
from objective import optimize_hyperparameters

# Add the project directory to the sys.path
project_dir = Path(__file__).resolve().parent
sys.path.append(str(project_dir))

BASE_DIR = Path(__file__).parent.parent
logger = logger_module.setup_logger('train_logger', BASE_DIR / 'logs', 'train.log')


def main(ticker='BTC-USD', worker=None, parameters=None):
    """Main function to train the model."""
    if parameters is None:
        parameters = load_best_params(BASE_DIR / 'best_params.json')

    logger.info(f"Starting training process for {ticker} with parameters: {parameters}")

    # Optimize hyperparameters if not provided
    if parameters is None:
        logger.info("Optimizing hyperparameters...")
        optimize_hyperparameters(20)
        parameters = load_best_params(BASE_DIR / 'best_params.json')
        logger.info(f"Optimized parameters: {parameters}")

    trainer = ModelTrainer(ticker)

    n_splits = parameters['n_folds']
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = tscv.split(trainer.x)

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

        try:
            model, history = train_model(
                x_train, y_train, x_test, y_test,
                model_dir=str(BASE_DIR / trainer.MODELS_FOLDER),
                ticker=trainer.ticker,
                fold_index=i,
                parameters=parameters,
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

        # Calculate and log metrics
        rmse, mae, mape = calculate_metrics(model, x_test, y_test)
        logger.info(f"Fold {i + 1}/{n_splits} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.4f}")
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
            f"Average metrics across all folds - RMSE: {average_rmse:.4f}, MAE: {average_mae:.4f}, MAPE: {average_mape:.4f}")

    logger.info("Training process completed.")


if __name__ == '__main__':
    main()
