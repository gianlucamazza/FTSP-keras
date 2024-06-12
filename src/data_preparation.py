from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

import logger as logger
from config import COLUMN_SETS

logger = logger.setup_logger('data_preparation_logger', 'logs', 'data_preparation.log')

BASE_DIR = Path(__file__).parent.parent


def download_financial_data(ticker, start_date, end_date):
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


def arrange_and_fill(df, ticker):
    logger.info(f"Arranging and filling missing data for {ticker}.")
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df[COLUMN_SETS['basic']]

    df.dropna(how='all', inplace=True)
    df.sort_index(inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df


def save_df_to_csv(df, relative_path):
    file_path = BASE_DIR / relative_path
    logger.info(f"Saving DataFrame to {file_path}.")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=True)


def get_financial_data(ticker, file_path=None, start_date=None, end_date=None):
    try:
        df = download_financial_data(ticker, start_date, end_date)
        if df.empty:
            error_message = f"No data returned for ticker {ticker}."
            logger.warning(error_message)
            raise ValueError(error_message)
        df = arrange_and_fill(df, ticker)
        save_df_to_csv(df, file_path)
        logger.info(f"Data for {ticker} successfully processed and saved.")
        return df
    except Exception as e:
        logger.error(f"Error in processing data for {ticker}: {e}")


def plot_price_history(dates, prices, ticker):
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


def main(ticker='BTC-USD', start_date=None, end_date=None):
    logger.info(f"Starting data preparation for {ticker}.")
    raw_data_path = f'data/raw_data_{ticker}.csv'
    processed_data_path = f'data/processed_data_{ticker}.csv'
    df = get_financial_data(ticker, file_path=raw_data_path, start_date=start_date, end_date=end_date)
    logger.info(f"Data for {ticker} successfully processed and saved.")
    logger.info(f"Start date: {df.index[0]}, End date: {df.index[-1]}")
    save_df_to_csv(df, processed_data_path)
    logger.info(f'Finished data preparation for {ticker}.')


if __name__ == '__main__':
    main(ticker='BTC-USD', start_date='2014-01-01', end_date=None)
