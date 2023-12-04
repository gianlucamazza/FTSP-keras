import subprocess

tickers = ["BTC-USD", "ETH-USD"]


def clean_data(_ticker="BTC-USD"):
    try:
        subprocess.run(["rm", "-f", f"data/processed_data_{_ticker}.csv"])
        subprocess.run(["rm", "-f", f"data/scaled_data_{_ticker}.csv"])
        subprocess.run(["rm", "-f", f"scalers/scaler_{_ticker}.pkl"])
        subprocess.run(["rm", "-f", f"scalers/close_scaler_{_ticker}.pkl"])
        print(f"Data for {_ticker} cleaned successfully.")
    except subprocess.CalledProcessError:
        print(f"An error occurred while cleaning the data for {_ticker}.")


def run_script(script_name, _ticker):
    try:
        subprocess.run(["python", script_name, _ticker], check=True)
        print(f"Script {script_name} executed successfully for {_ticker}.")
    except subprocess.CalledProcessError:
        print(f"An error occurred while executing {script_name} for {_ticker}.")


if __name__ == "__main__":
    scripts = [
        "data_preparation.py",
        "feature_engineering.py",
        "model.py",
        "train.py",
        "predict.py"
    ]

    for ticker in tickers:
        clean_data(ticker)
        for script in scripts:
            try:
                run_script(script, ticker)
            except subprocess.CalledProcessError:
                print(f"An error occurred while executing {script} for {ticker}.")
                break
