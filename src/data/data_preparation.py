import argparse
import sys
from pathlib import Path
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

# Ensure the project directory is in the sys.path
project_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_dir))

from src.config import COLUMN_SETS
from src.logging.logger import setup_logger

# Setup logger
ROOT_DIR = project_dir
logger = setup_logger('data_preparation_logger', 'logs', 'data_preparation.log')


def download_financial_data(ticker, start_date, end_date):
    """Download financial data for a given ticker and date range."""
    logger.info(f"Downloading data for {ticker} from {start_date} to {end_date}.")
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            error_message = f"No data returned for ticker {ticker}."
            logger.warning(error_message)
            raise ValueError(error_message)
        return data
    except Exception as e:
        logger.error(f"Error in downloading data for {ticker}: {e}")
        raise


def arrange_and_fill(df, name):
    """Arrange and fill missing data in the DataFrame."""
    logger.info(f"Arranging and filling missing data for {name}.")
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df[COLUMN_SETS['basic']].copy()
    df.dropna(how='all', inplace=True)
    df.sort_index(inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df


def save_df_to_csv(df, relative_path):
    """Save DataFrame to a CSV file."""
    file_path = ROOT_DIR / 'data' / relative_path
    logger.info(f"Saving DataFrame to {file_path}.")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=True)


def get_financial_data(ticker: str, name: str, file_path=None, start_date=None, end_date=None):
    """Download, arrange, fill, and save financial data for a given ticker."""
    try:
        df = download_financial_data(ticker, start_date, end_date)
        df = arrange_and_fill(df, name)
        save_df_to_csv(df, file_path)
        logger.info(f"Data for {name} successfully processed and saved.")
        return df
    except Exception as e:
        logger.error(f"Error in processing data for {name}: {e}")
        raise


def plot_price_history(dates, prices, ticker):
    """Plot the price history for a given ticker."""
    logger.info(f"Plotting price history for {ticker}.")
    plt.figure(figsize=(15, 10))
    plt.plot(dates, prices, label='Close Price')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gcf().autofmt_xdate()
    plt.title(f'{ticker} Price History')
    plt.ylabel('Price (USD)')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main(ticker: str, label: str, start_date=None, end_date=None, worker=None):
    """Main function to prepare data for a given ticker."""
    logger.info(f"Starting data preparation for {label}.")
    raw_data_path = f'raw_data_{label}.csv'
    processed_data_path = f'processed_data_{label}.csv'
    try:
        df = get_financial_data(ticker, file_path=raw_data_path, start_date=start_date, end_date=end_date)
        logger.info(f"Start date: {df.index[0]}, End date: {df.index[-1]}")
        save_df_to_csv(df, processed_data_path)
        logger.info(f'Finished data preparation for {label}.')
        if worker and not worker.is_running():
            return
    except Exception as e:
        logger.error(f"Failed to complete data preparation for {label}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data Preparation')
    parser.add_argument('--yfinance_ticker', type=str, required=True, help='Yearn Finance Symbol')
    parser.add_argument('--ticker', type=str, required=True, help='Ticker label')
    parser.add_argument('--start_date', type=str, required=True, help='Start date')
    parser.add_argument('--end_date', type=str, help='End date')

    args = parser.parse_args()
    main(ticker=args.ticker, label=args.label, start_date=args.start_date, end_date=args.end_date)
