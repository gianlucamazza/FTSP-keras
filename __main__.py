import subprocess

PROCESSED_DATA_PATH = "data/processed_data.csv"
SCALED_DATA_PATH = "data/scaled_data.csv"
MODEL_PATH = "models/bitcoin_prediction_model.keras"
SCALER_PATH = "models/scaler.pkl"
BTC_CSV_PATH = "data/BTC-USD.csv"


def clean_data():
    try:
        # Remove the CSV files
        subprocess.run(["rm", PROCESSED_DATA_PATH], check=True)
        subprocess.run(["rm", SCALED_DATA_PATH], check=True)
        # Remove all the files in the models directory
        print("Data cleaned successfully.")
    except subprocess.CalledProcessError:
        print("An error occurred while cleaning the data.")


def run_script(script_name, args=None):
    command = ["python", script_name]
    if args:
        command += args
    try:
        subprocess.run(command, check=True)
        print(f"Script {script_name} executed successfully.")
    except subprocess.CalledProcessError:
        print(f"An error occurred while executing {script_name}.")


if __name__ == "__main__":
    # Clean the data
    clean_data()

    scripts = [
        ("data_preparation.py", ["--file_path", BTC_CSV_PATH]),
        ("feature_engineering.py", ["--data", PROCESSED_DATA_PATH, "--output", SCALED_DATA_PATH]),
        ("model.py", ["--input_shape", "50", "15"]),
        ("train.py", ["--data_path", SCALED_DATA_PATH, "--model_path", MODEL_PATH, "--scaler_path", SCALER_PATH]),
        ("predict.py", ["--data", SCALED_DATA_PATH, "--model", MODEL_PATH])
    ]

    for script, args in scripts:
        run_script(script, args)