import sys
from pathlib import Path
import tensorflow_cloud as tfc
from tensorflow.keras import mixed_precision
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import logger as logger_module
from train_utils import load_best_params, calculate_metrics
from train_model import train_model, ModelTrainer
from objective import optimize_hyperparameters

# Add the project directory to the sys.path for module resolution
project_dir = Path(__file__).resolve().parent
sys.path.append(str(project_dir))

BASE_DIR = Path(__file__).parent.parent
logger = logger_module.setup_logger('train_logger', BASE_DIR / 'logs', 'train.log')

# Enable mixed precision training for performance improvement
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)


def main(ticker='BTC-USD', worker=None, parameters=None, trial_id=None):
    """
    Main function to train the model.

    Args:
        ticker (str): The ticker symbol for the financial instrument.
        worker (optional): A worker object to manage early stopping.
        parameters (dict, optional): Hyperparameters for the model.
        trial_id (int, optional): The trial ID from hyperparameter optimization.
    """
    if parameters is None:
        logger.info("Optimizing hyperparameters...")
        optimize_hyperparameters(n_trials=50)
        parameters = load_best_params(BASE_DIR / 'best_params.json')
        logger.info(f"Optimized parameters: {parameters}")

    logger.info(f"Starting training process for {ticker} with parameters: {parameters}")

    # Initialize ModelTrainer
    trainer = ModelTrainer(ticker)

    # Set up cross-validation with TimeSeriesSplit
    n_splits = parameters['n_folds']
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = list(tscv.split(trainer.x))

    logger.info(f"Number of splits: {n_splits}")

    best_val_loss = np.inf
    best_model = None
    metrics_list = []

    for i, (train_index, test_index) in enumerate(splits):
        if worker is not None and hasattr(worker, 'is_running') and not worker.is_running:
            logger.info("Training stopped early.")
            return

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
        average_rmse = np.mean([m[0] for m in metrics_list])
        average_mae = np.mean([m[1] for m in metrics_list])
        average_mape = np.mean([m[2] for m in metrics_list])
        logger.info(
            f"Average metrics across all folds - RMSE: {average_rmse:.4f}, MAE: {average_mae:.4f}, MAPE: {average_mape:.4f}")

    logger.info("Training process completed.")


if __name__ == '__main__':
    use_cloud = False  # Set to True to use TensorFlow Cloud for training
    if use_cloud:
        TF_GPU_IMAGE = "tensorflow/tensorflow:latest-gpu"
        GCS_BUCKET = 'modeltrainer-bucket'

        run_parameters = {
            'distribution_strategy': 'auto',
            'requirements_txt': 'requirements.txt',
            'docker_config': tfc.DockerConfig(
                parent_image=TF_GPU_IMAGE,
                image_build_bucket=GCS_BUCKET
            ),
            'chief_config': tfc.COMMON_MACHINE_CONFIGS['K80_1X'],
            'worker_config': tfc.COMMON_MACHINE_CONFIGS['K80_1X'],
            'worker_count': 3,
            'job_labels': {'job': "btc_price_prediction"}
        }

        tfc.run(**run_parameters)
    else:
        main()
