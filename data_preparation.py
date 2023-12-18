# data_preparation.py
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import logger as logger

logger = logger.setup_logger('data_preparation_logger', 'logs', 'data_preparation.log')

COLUMNS_TO_SCALE = ['Open', 'High', 'Low', 'Adj Close', 'Volume',
                    'MA50', 'MA200', 'Returns', 'Volatility', 'MA20',
                    'Upper', 'Lower', 'RSI', 'MACD']

COLUMNS = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

RSI_WINDOW = 14
MACD_FAST_WINDOW = 12
MACD_SLOW_WINDOW = 26


def download_financial_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            raise ValueError(f"No data returned for ticker {ticker}.")
        return data
    except Exception as e:
        logger.error(f"Error in downloading data for {ticker}: {e}")
        raise


def arrange_and_fill(df):
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df[COLUMNS]

    df.dropna(how='all', inplace=True)

    df.sort_index(inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df


def save_df_to_csv(df, file_path):
    if file_path:
        df.to_csv(file_path, index=True)


def get_financial_data(ticker, file_path=None, start_date=None, end_date=None):
    try:
        df = download_financial_data(ticker, start_date, end_date)
        if df.empty:
            raise ValueError(f"No data returned for ticker {ticker}.")
        df = arrange_and_fill(df)
        save_df_to_csv(df, file_path)
        logger.info(f"Downloaded data for {ticker} from {start_date} to {end_date}.")
    except Exception as e:
        logger.error(f"Error in downloading or processing data for {ticker}: {e}")
        raise


def calculate_rsi(prices, n=RSI_WINDOW):
    deltas = prices.diff()
    loss = deltas.where(deltas < 0, 0)
    gain = deltas.where(deltas > 0, 0)
    avg_gain = gain.rolling(window=n, min_periods=1).mean()
    avg_loss = -loss.rolling(window=n, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def calculate_macd(prices, n_fast=MACD_FAST_WINDOW, n_slow=MACD_SLOW_WINDOW):
    ema_fast = prices.ewm(span=n_fast, min_periods=n_slow).mean()
    ema_slow = prices.ewm(span=n_slow, min_periods=n_slow).mean()
    return ema_fast - ema_slow


def calculate_bollinger_bands(df):
    df['MA20'] = df['Close'].rolling(20).mean()
    df['Volatility'] = df['Returns'].rolling(20).std()
    df['Upper'] = df['MA20'] + 2 * df['Volatility']
    df['Lower'] = df['MA20'] - 2 * df['Volatility']
    return df


def calculate_technical_indicators(df):
    required_columns = ['Close']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain the following columns: {required_columns}")
    df['MA50'] = df['Close'].rolling(50).mean()
    df['MA200'] = df['Close'].rolling(200).mean()
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(50).std()
    df = calculate_bollinger_bands(df)
    df['RSI'] = calculate_rsi(df['Close'])
    df['MACD'] = calculate_macd(df['Close'])
    return df


def plot_price_history(dates, prices, ticker):
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
    get_financial_data(ticker, file_path=f'data/raw_data_{ticker}.csv', start_date=start_date, end_date=end_date)
    df = pd.read_csv(f'data/raw_data_{ticker}.csv', index_col='Date', parse_dates=True)
    df = calculate_technical_indicators(df)

    if df.isnull().any().any():
        logger.error("DataFrame still contains NaN values after preprocessing.")
        df.dropna(inplace=True)
        logger.info("NaN values removed from DataFrame.")

    df.to_csv(f'data/processed_data_{ticker}.csv', index=True)
    logger.info(f"Processed data shape: {df.shape}")
    logger.info(f"Processed data columns: {df.columns.tolist()}")


if __name__ == '__main__':
    main(ticker='BTC-USD', start_date='2023-01-01', end_date=None)
