# Bitcoin Price Prediction Using LSTM

## Project Overview
This project uses a Long Short-Term Memory (LSTM) model to predict Bitcoin prices. It covers data preparation, feature engineering, model building, training, and predictions using historical price data.

## Motivation
The aim is to apply machine learning techniques to financial data, developing a model that accurately predicts Bitcoin prices, potentially aiding in investment decisions.

## Structure
The repository includes scripts for each stage of the machine learning pipeline and a main script to run the entire process:

- `data_preparation.py`: Processes raw Bitcoin price data.
- `feature_engineering.py`: Enhances data with technical indicators and normalizes it.
- `model.py`: Defines the LSTM neural network architecture.
- `train.py`: Trains and evaluates the model with historical data.
- `predict.py`: Uses the trained model for future price predictions.
- `main.py`: Automates the running of scripts for training and prediction.

## Data
Data is downloaded from Yahoo Finance using `yfinance`. The raw data is stored in `data/raw_data.csv`.

Processed data files:
- `processed_data.csv`: Data with added technical indicators.
- `scaled_data.csv`: Normalized data for model training.

## Usage
Run the entire process with `main.py` or individual scripts:

1. **Using main.py** (automated process):
   ```bash
   python main.py --train   # For training
   python main.py --predict # For predictions
    ```
2. **Using individual scripts**:
    ```bash
    python data_preparation.py
    python feature_engineering.py
    python train.py
    python predict.py
    ```
### Optional Arguments
**Run individual scripts with different ticker symbols**:
    ```bash
    python data_preparation.py --ticker ETH-USD
    python feature_engineering.py --ticker ETH-USD
    python train.py --ticker ETH-USD
    python predict.py --ticker ETH-USD
    ```

## Logging
The project uses Python's logging module to output logs to both the terminal and log files, aiding in monitoring and debugging.

## Requirements
```
pandas~=2.1.3
numpy~=1.26.2
tensorflow~=2.15.0
matplotlib~=3.8.2
scikit-learn~=1.3.2
keras~=2.15.0
joblib~=1.3.2
yfinance~=0.2.32
```

## Install Dependencies
```bash
pip install -r requirements.txt
```

## Contributing
Contributions to improve the model or add new features are welcome. Please use the standard pull request process.

## License
[MIT](https://choosealicense.com/licenses/mit/)