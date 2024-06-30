# Financial Time Series Prediction Using LSTM

## Project Overview
This project employs a Long Short-Term Memory (LSTM) model to predict financial time series data. It encompasses data preparation, feature engineering, model optimization, training, and predictions using historical financial data.

## Motivation
The goal is to apply machine learning techniques to financial data, developing a model that accurately predicts prices, aiding in investment and trading decisions.

## Data
Data is sourced from Yahoo Finance using `yfinance`. The raw data is stored in `data/raw_data.csv`.

Processed data files:
- `data/processed_data_{TICKER}.csv`: Cleaned data.
- `data/raw_data_{TICKER}.csv`: Raw data.
- `data/scaled_data_{TICKER}.csv`: Normalized data with technical indicators for model training.

## Usage

```bash
# Prepare the data
python3 src/data/data_preparation.py --ticker {TICKER} --start_date=YYYY-MM-DD

# Setup the feature engineering
python3 src/feature_engineering.py --ticker {TICKER} --scaler RobustScaler

# Train the model
python3 src/training/train_model.py --ticker {TICKER} --params best_params.json 
