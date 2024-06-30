import sys
from pathlib import Path

from src.config import COLUMN_SETS
from src.logging.logger import setup_logger
from src.data.technical_indicators import calculate_technical_indicators

# Ensure the project directory is in the sys.path
project_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_dir))


ROOT_DIR = Path(__file__).parent.parent
logger = setup_logger('data_utils_logger', 'logs', 'data_utils_logger.log')


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
