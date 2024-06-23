# Bitcoin Price Prediction Using LSTM

## Project Overview
This project uses a Long Short-Term Memory (LSTM) model to predict Bitcoin prices. It encompasses data preparation, feature engineering, model building, training, and predictions using historical price data.

## Motivation
The aim is to apply machine learning techniques to financial data, developing a model that accurately predicts Bitcoin prices, potentially aiding in investment decisions.

## Structure

- `data_preparation.py`: Processes raw Bitcoin price data.
- `data_utils.py`: Utilities for data processing.
- `technical_indicators.py`: Adds technical indicators to the data.
- `feature_engineering.py`: Enhances data with technical indicators and normalizes it.
- `logger.py`: Sets up logging for the project.
- `objective.py`: Defines the objective function for hyperparameter optimization.
- `model.py`: Defines the LSTM neural network architecture.
- `train.py`: Trains and evaluates the model with historical data.
- `train_model.py`: Contains functions for model training.
- `train_utils.py`: Utility functions for training.
- `predict.py`: Uses the trained model for future price predictions.

## Data
Data is downloaded from Yahoo Finance using `yfinance`. The raw data is stored in `data/raw_data.csv`.

Processed data files:
- `data/processed_data_{TICKER}.csv`: Cleaned up data.
- `data/raw_data_{TICKER}.csv`: Raw data.
- `data/scaled_data_{TICKER}.csv`: Normalized data with technical indicators for model training.

## Usage

### Training
Run the training script to train the model:
```bash
sh train.sh
```

Run training scripts individually
```bash
# Prepare the data
python3 src/data_preparation.py

# Setup the feature engineering
python3 src/feature_engineering.py

# Train the model
python3 src/train.py
```

### Predict
Run the prediction script to make predictions:
```bash
sh predict.sh
```

```bash
# Predict the test data
python3 src/predict.py
```

## Logging
The project uses Python's logging module to output logs to both the terminal and log files, aiding in monitoring and debugging.

## Requirements
```
pandas~=2.1.3
numpy~=1.26.2
tensorflow~=2.16.1
featuretools~=1.31.0
statsmodels~=0.14.2
matplotlib~=3.9.0
scikit-learn~=1.3.2
keras~=3.3.3
joblib~=1.3.2
yfinance~=0.2.32
PyQt5~=5.15.10
optuna~=3.6.1
optuna-integration~=3.6.0
```

## Install Dependencies
```bash
pip install -r requirements.txt
```

## Contributors
- Gianluca Mazza
- Matteo Garbelli

## License
This project is licensed under the MIT License - see the LICENSE file for details.