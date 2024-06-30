import sys
import argparse
from pathlib import Path
import pandas as pd

# Ensure the project directory is in the sys.path
project_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_dir))

from src.indicators.technical_indicators import calculate_technical_indicators
from src.logging.logger import setup_logger

# Setup logger
ROOT_DIR = project_dir
logger = setup_logger('calculate_indicators_logger', 'logs', 'calculate_indicators.log')


def process_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators."""
    logger.info("Calculating technical indicators.")
    df = calculate_technical_indicators(df)
    logger.info("Technical indicators calculated.")
    return df


def main(ticker: str) -> None:
    """Main function to calculate technical indicators for a given ticker."""
    logger.info(f"Starting calculation of technical indicators for {ticker}.")
    file_path = ROOT_DIR / f'data/raw_data_{ticker}.csv'
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return
    try:
        df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        logger.info(f"Data loaded. Shape: {df.shape}")
        df = process_indicators(df)
        processed_data_path = ROOT_DIR / f'data/processed_data_{ticker}.csv'
        df.to_csv(processed_data_path, index=True)
        logger.info(f"Processed data saved at {processed_data_path}")
    except Exception as e:
        logger.error(f"Failed to calculate technical indicators for {ticker}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate Technical Indicators')
    parser.add_argument('--ticker', type=str, required=True, help='Ticker')
    args = parser.parse_args()
    main(ticker=args.ticker)
