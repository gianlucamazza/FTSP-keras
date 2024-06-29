import json
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from src.logging.logger import setup_logger

# Setup logger
project_dir = Path(__file__).resolve().parent.parent.parent
logger = setup_logger('data_preparation_logger', 'logs', 'data_preparation.log')


def save_best_params(params: dict, file_path: Path, ticker: str) -> None:
    """Save the best model parameters to a JSON file."""
    params['ticker'] = ticker
    try:
        with file_path.open('w') as f:
            json.dump(params, f, indent=4)
        logger.info(f"Best parameters saved to {file_path}")
    except IOError as e:
        logger.error(f"Failed to save parameters: {e}")


def load_best_params(path: Path) -> None:
    """Load the best model parameters from a JSON file."""
    try:
        if path.exists():
            logger.info(f"Found parameters file at {path}")
            with path.open('r') as f:
                return json.load(f)
        else:
            logger.warning(f"No parameters file found at {path}")
            return None
    except IOError as e:
        logger.error(f"Failed to load parameters: {e}")
        return None


def calculate_metrics(model, x_test: np.ndarray, y_test: np.ndarray) -> tuple:
    """Calculate and log evaluation metrics for the model."""
    y_pred = model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    logger.info(f"Evaluation metrics - RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.4f}")
    return rmse, mae, mape
