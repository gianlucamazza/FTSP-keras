import json
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

from config import PARAMETERS
import logger as logger_module

BASE_DIR = Path(__file__).parent.parent
logger = logger_module.setup_logger('train_utils_logger', BASE_DIR / 'logs', 'train_utils.log')


def save_best_params(params, path):
    with open(path, 'w') as f:
        json.dump(params, f, indent=4)


def load_best_params(path):
    if path.exists():
        logger.info(f"Found parameters file at {path}")
        with open(path, 'r') as f:
            return json.load(f)
    else:
        return


def calculate_metrics(model, x_test, y_test):
    """Calculate evaluation metrics for the model."""
    y_pred = model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    logger.info(f"Evaluation metrics - RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.4f}")
    return rmse, mae, mape
