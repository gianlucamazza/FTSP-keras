from config import COLUMN_SETS
from technical_indicators import calculate_technical_indicators
import logger as logger_module
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
logger = logger_module.setup_logger('data_utils_logger', BASE_DIR / 'logs', 'data_utils.log')


def prepare_data(df, feature_scaler):
    """Prepare the data for training or prediction."""
    # Ensure all required columns are present
    if not set(COLUMN_SETS['to_scale']).issubset(df.columns):
        df = calculate_technical_indicators(df)

    missing_columns = set(COLUMN_SETS['to_scale']) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns in the DataFrame: {missing_columns}")

    scaler_columns = COLUMN_SETS['to_scale']
    df = df.reindex(columns=scaler_columns)

    try:
        df[scaler_columns] = feature_scaler.transform(df[scaler_columns])
    except ValueError as e:
        logger.error(f"Error during scaling: {e}")
        raise

    return df
