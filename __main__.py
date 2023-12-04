import argparse
import os
import subprocess
from logger import setup_logger

logger = setup_logger('main_logger', 'logs', 'main.log')

tickers = ["BTC-USD", "ETH-USD"]


def clean_data(_ticker="BTC-USD"):
    files_to_remove = [
        f"data/processed_data_{_ticker}.csv",
        f"data/scaled_data_{_ticker}.csv",
        f"scalers/scaler_{_ticker}.pkl",
        f"scalers/close_scaler_{_ticker}.pkl"
    ]

    for file in files_to_remove:
        try:
            if os.path.exists(file):
                os.remove(file)
                logger.info(f"Removed {file}")
        except Exception as e:
            logger.error(f"Error removing {file}: {e}")


def run_script(script_name, _ticker):
    try:
        subprocess.run(["python", script_name, _ticker], check=True)
        logger.info(f"Script {script_name} executed successfully for {_ticker}.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing {script_name} for {_ticker}: {e}")


def main(_args):
    scripts = []
    if _args.train:
        scripts += ["data_preparation.py", "feature_engineering.py", "model.py", "train.py"]
    if _args.predict:
        scripts += ["predict.py"]

    for ticker in tickers:
        clean_data(ticker)
        for script in scripts:
            run_script(script, ticker)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training and prediction scripts.")
    parser.add_argument("--train", action="store_true", help="Run the training scripts.")
    parser.add_argument("--predict", action="store_true", help="Run the prediction script.")
    args = parser.parse_args()
    main(args)
