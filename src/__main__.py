import argparse
import subprocess
from src.logger import setup_logger
from pathlib import Path

logger = setup_logger('main_logger', 'logs', 'main.log')

tickers = ["BTC-USD"]

BASE_DIR = Path(__file__).parent.parent


def clean_data(_ticker="BTC-USD"):
    files_to_remove = [
        Path(f"data/processed_data_{_ticker}.csv"),
        Path(f"data/scaled_data_{_ticker}.csv"),
        Path(f"scalers/scaler_{_ticker}.pkl"),
        Path(f"scalers/close_scaler_{_ticker}.pkl")
    ]

    for file_path in files_to_remove:
        try:
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Removed {file_path}")
        except Exception as e:
            logger.error(f"Error removing {file_path}: {e}")


def run_script(script_name, _ticker):
    script_path = BASE_DIR / 'src' / script_name
    try:
        subprocess.run(["python", str(script_path), _ticker], check=True)
        logger.info(f"Script {script_name} executed successfully for {_ticker}.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing {script_name} for {_ticker}: {e}")


def main(_args):
    if _args.train:
        scripts = ["data_preparation.py", "feature_engineering.py", "train.py"]
        for ticker in tickers:
            clean_data(ticker)
            for script in scripts:
                run_script(script, ticker)
    if _args.predict:
        scripts = ["predict.py"]
        for ticker in tickers:
            for script in scripts:
                run_script(script, ticker)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training and prediction scripts.")
    parser.add_argument("--train", action="store_true", help="Run the training scripts.")
    parser.add_argument("--predict", action="store_true", help="Run the prediction script.")
    args = parser.parse_args()
    main(args)
