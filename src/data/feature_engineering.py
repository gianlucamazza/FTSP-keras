import sys
import argparse
from pathlib import Path
import pandas as pd
import pmdarima as pm
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import joblib
from statsmodels.tsa.stattools import adfuller

# Ensure the project directory is in the sys.path
project_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_dir))

from src.config import COLUMN_SETS, CLOSE, ARIMA_ORDER, ARIMA_SEASONAL_ORDER
from src.logging.logger import setup_logger

# Setup logger
ROOT_DIR = project_dir
logger = setup_logger('feature_engineering_logger', 'logs', 'feature_engineering_logger.log')


def validate_input_data(df: pd.DataFrame, required_columns: list) -> None:
    """Validate that all required columns are present in the DataFrame."""
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        logger.error(f"Missing columns in input data: {missing_columns}")
        raise ValueError(f"Missing columns: {missing_columns}")


def check_data(df: pd.DataFrame, step_description: str) -> None:
    """Log the shape and sample of the DataFrame at a given step."""
    logger.info(f"Data after {step_description}: shape = {df.shape}")
    logger.debug(f"Sample data:\n{df.head()}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the DataFrame by removing rows with NaN values."""
    initial_shape = df.shape
    df.dropna(inplace=True)
    logger.info(f"Data cleaned. NaN values handled. {initial_shape[0] - df.shape[0]} rows removed.")
    check_data(df, "cleaning data")
    return df


def normalize_features(df: pd.DataFrame, columns_to_normalize: list, scaler_type: str = 'RobustScaler') -> (pd.DataFrame, object):
    """Normalize specified features in the DataFrame using the specified scaler."""
    logger.info("Normalizing features.")
    scaler = MinMaxScaler(feature_range=(0.1, 0.9)) if scaler_type == 'MinMaxScaler' else RobustScaler()
    df_reset = df.reset_index()
    df_reset[columns_to_normalize] = scaler.fit_transform(df_reset[columns_to_normalize])
    df_normalized = df_reset.set_index('Date')
    check_data(df_normalized, "normalizing features")
    return df_normalized, scaler


def save_scaler(scaler, ticker: str) -> None:
    """Save the scaler object to disk."""
    path = ROOT_DIR / f'scalers/feature_scaler_{ticker}.pkl'
    joblib.dump(scaler, path)
    logger.info(f"Feature scaler saved at {path}")


def optimize_arima(y: pd.Series) -> pm.arima.ARIMA:
    """Optimize ARIMA model using specified parameters."""
    y = y.astype(float)

    # Ensure the frequency is set
    if y.index.freq is None:
        y = y.asfreq('B')  # Assume business days if not set

    result = adfuller(y)
    if isinstance(result, tuple) and len(result) > 1:
        p_value = result[1]
        if p_value > 0.05:
            logger.info("Data is not stationary. Differencing might be required.")
    else:
        logger.error("Unexpected result from adfuller: Expected tuple, got {}".format(type(result)))

    # Fit the ARIMA model with specified parameters and additional settings
    model = pm.ARIMA(order=ARIMA_ORDER, seasonal_order=ARIMA_SEASONAL_ORDER, enforce_stationarity=False, enforce_invertibility=False, maxiter=200)
    model.fit(y)
    logger.info(f"Optimal parameters: {model.order}, seasonal_order: {model.seasonal_order}")
    return model


def process_and_save_features(df: pd.DataFrame, ticker: str, scaler_type: str) -> None:
    """Process features by normalizing and saving the data and scaler."""
    try:
        logger.info(f"Processing features for {ticker}.")
        validate_input_data(df, COLUMN_SETS['to_scale'] + COLUMN_SETS['required'])
        df = clean_data(df)

        if 'Date' not in df.index.names:
            df.set_index('Date', inplace=True)
            logger.info("'Date' column set as index.")
        df.index = pd.to_datetime(df.index)

        # Set the frequency based on the ticker
        if ticker == "BTC-USD":
            df = df.asfreq('D')
        else:
            df = df.asfreq('B')

        # Log the inferred frequency
        logger.info(f"Data frequency set to: {df.index.freq}")

        # Interpolation or handling missing data if necessary
        df.ffill(inplace=True)

        feature_matrix, feature_scaler = normalize_features(df, COLUMN_SETS['to_scale'], scaler_type)
        save_scaler(feature_scaler, ticker)

        scaled_data_path = ROOT_DIR / f'data/scaled_data_{ticker}.csv'
        if scaled_data_path.exists():
            logger.warning(f"File {scaled_data_path} already exists and will be overwritten.")
        feature_matrix.to_csv(scaled_data_path, index=True)
        logger.info(f"Scaled data saved at {scaled_data_path}")

        optimized_model = optimize_arima(feature_matrix[CLOSE])
        logger.info(f"Best ARIMA model: {optimized_model.order}, seasonal_order: {optimized_model.seasonal_order}")

    except Exception as e:
        logger.error(f"Error in process_and_save_features: {e}", exc_info=True)
        raise


def main(ticker: str, scaler_type: str, worker=None) -> None:
    """Main function to start feature engineering process for a given ticker."""
    logger.info(f"Starting feature engineering for {ticker}.")
    file_path = ROOT_DIR / f'data/processed_data_{ticker}.csv'
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return
    try:
        df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        logger.info(f"Data loaded. Shape: {df.shape}")
        logger.info(f"Index: {df.index.name}")
        process_and_save_features(df, ticker, scaler_type)
        logger.info(f"Feature engineering completed for {ticker}.")
    except Exception as e:
        logger.error(f"Failed to complete feature engineering for {ticker}: {e}")

    if worker and hasattr(worker, 'is_running') and not worker.is_running():
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature Engineering')
    parser.add_argument('--ticker', type=str, required=True, help='Ticker')
    parser.add_argument('--scaler', type=str, default='RobustScaler', choices=['MinMaxScaler', 'RobustScaler'],
                        help='Scaler type')
    args = parser.parse_args()
    main(ticker=args.ticker, scaler_type=args.scaler)
