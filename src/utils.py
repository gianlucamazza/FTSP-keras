import json
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from src.logging.logger import setup_logger

# Setup logger
logger = setup_logger('utils_logger', 'logs', 'utils.log')


def save_to_json(params: dict, file_path: Path) -> None:
    """Save parameters to a JSON file."""
    try:
        with file_path.open('w') as f:
            json.dump(params, f, indent=4)
        logger.info(f"Parameters saved to {file_path}")
    except IOError as e:
        logger.error(f"Failed to save parameters: {e}")


def load_from_json(path: Path) -> dict:
    """Load parameters from a JSON file."""
    try:
        if path.exists():
            logger.info(f"Found parameters file at {path}")
            with path.open('r') as f:
                return json.load(f)
        else:
            logger.warning(f"No parameters file found at {path}")
            return {}
    except IOError as e:
        logger.error(f"Failed to load parameters: {e}")
        return {}


def update_json(file_path: Path, new_params: dict) -> None:
    """Update existing JSON file with new parameters."""
    try:
        data = load_from_json(file_path)
        data.update(new_params)
        save_to_json(data, file_path)
        logger.info(f"Parameters updated in {file_path}")
    except Exception as e:
        logger.error(f"Failed to update parameters: {e}")


def calculate_metrics(model, x_test: np.ndarray, y_test: np.ndarray) -> tuple:
    """Calculate and log evaluation metrics for the model."""
    y_pred = model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    logger.info(f"Evaluation metrics - RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.4f}")
    return rmse, mae, mape
