import sys
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
from pathlib import Path
import featuretools as ft
import statsmodels.api as sm
import logger as logger_module
from config import COLUMN_SETS, CLOSE
from technical_indicators import calculate_technical_indicators

# Add the project directory to the sys.path
project_dir = Path(__file__).resolve().parent
sys.path.append(str(project_dir))

BASE_DIR = Path(__file__).parent.parent
logger = logger_module.setup_logger('feature_engineering_logger', BASE_DIR / 'logs', 'feature_engineering.log')


def validate_input_data(df: pd.DataFrame, required_columns: list):
    """Validate that all required columns are present in the DataFrame."""
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        error_message = f"Missing columns in input data: {missing_columns}"
        logger.error(error_message)
        raise ValueError(error_message)


def check_data(df: pd.DataFrame, step_description: str):
    """Log the shape and sample of the DataFrame at a given step."""
    logger.info(f"Data after {step_description}: shape = {df.shape}")
    logger.info(f"Sample data:\n{df.head()}")


def clean_data(df: pd.DataFrame):
    """Clean the DataFrame by removing rows with NaN values."""
    initial_shape = df.shape
    df.dropna(inplace=True, how='any')
    logger.info(f"Data cleaned. NaN values handled. {initial_shape[0] - df.shape[0]} rows removed.")
    check_data(df, "cleaning data")


def normalize_features(df: pd.DataFrame, columns_to_normalize: list):
    """Normalize specified features in the DataFrame using MinMaxScaler."""
    logger.info("Normalizing features.")
    scaler = MinMaxScaler()
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    check_data(df, "normalizing features")
    return df, scaler


def save_scaler(scaler: MinMaxScaler, ticker: str):
    """Save the scaler object to disk."""
    path = BASE_DIR / f'scalers/feature_scaler_{ticker}.pkl'
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)
    logger.info(f"Feature scaler saved at {path}")


def fit_arima_model(data: pd.Series):
    """Fit an ARIMA model to the data."""
    try:
        model = sm.tsa.ARIMA(data, order=(1, 1, 1))
        results = model.fit(method_kwargs={'maxiter': 500, 'disp': 0})
        logger.info(f"ARIMA model summary: \n{results.summary()}")
        return results
    except Exception as e:
        logger.error(f"Error fitting ARIMA model: {e}", exc_info=True)
        raise


def process_and_save_features(df: pd.DataFrame, ticker: str):
    """Process features by calculating technical indicators, normalizing, and saving the data and scaler."""
    try:
        logger.info(f"Processing features for {ticker}.")

        # Calculate technical indicators
        df = calculate_technical_indicators(df)
        check_data(df, "calculating technical indicators")

        # Validate input data
        required_columns = COLUMN_SETS['to_scale'] + COLUMN_SETS['required']
        validate_input_data(df, required_columns)

        # Clean data
        clean_data(df)

        # Create an entity set and add the dataframe as an entity
        es = ft.EntitySet(id='financial_data')
        es = es.add_dataframe(dataframe_name='data', dataframe=df, index='Date', make_index=False)

        # Run deep feature synthesis
        feature_matrix, feature_defs = ft.dfs(entityset=es, target_dataframe_name='data', max_depth=1)
        logger.info(f"Featuretools created {len(feature_defs)} features.")
        check_data(feature_matrix, "creating features with Featuretools")

        # Normalize features
        columns_to_normalize = feature_matrix.columns.tolist()
        feature_matrix, feature_scaler = normalize_features(feature_matrix, columns_to_normalize)

        save_scaler(feature_scaler, ticker)

        scaled_data_path = BASE_DIR / f'data/scaled_data_{ticker}.csv'
        if scaled_data_path.exists():
            logger.warning(f"File {scaled_data_path} already exists and will be overwritten.")
        feature_matrix.to_csv(scaled_data_path, index=True)
        logger.info(f"Scaled data saved at {scaled_data_path}")

        # Fit ARIMA model with statsmodels
        fit_arima_model(feature_matrix[CLOSE])

    except Exception as e:
        logger.error(f"Error in process_and_save_features: {e}", exc_info=True)
        raise


def main(ticker: str = 'BTC-USD', worker=None):
    """Main function to start feature engineering process for a given ticker."""
    logger.info(f"Starting feature engineering for {ticker}.")
    file_path = Path(BASE_DIR / f'data/processed_data_{ticker}.csv')
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return
    try:
        df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        process_and_save_features(df, ticker)
        logger.info(f"Feature engineering completed for {ticker}.")
    except Exception as e:
        logger.error(f"Failed to complete feature engineering for {ticker}: {e}")

    if worker and hasattr(worker, 'is_running') and not worker.is_running():
        return


if __name__ == '__main__':
    main(ticker='BTC-USD')
