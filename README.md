# Bitcoin Price Prediction Using LSTM

## Project Overview
This project uses a Long Short-Term Memory (LSTM) model to predict Bitcoin prices. It encompasses data preparation, feature engineering, model building, training, and predictions using historical price data.

## Motivation
The aim is to apply machine learning techniques to financial data, developing a model that accurately predicts Bitcoin prices, potentially aiding in investment decisions.

## Data
Data is downloaded from Yahoo Finance using `yfinance`. The raw data is stored in `data/raw_data.csv`.

Processed data files:
- `data/processed_data_{TICKER}.csv`: Cleaned up data.
- `data/raw_data_{TICKER}.csv`: Raw data.
- `data/scaled_data_{TICKER}.csv`: Normalized data with technical indicators for model training.

## Usage

```bash
# Prepare the data
python3 src/data/data_preparation.py --ticker BTC --start_date=2020-01-01

# Setup the feature engineering
python3 src/feature_engineering.py --ticker BTC --scaler RobustScaler

# Train the model
python3 src/training/train_model.py --ticker BTC --params BTC_best_params.json 
```

### Predict
Run the prediction script to make predictions:
```bash
sh predict.sh
```

## Install
```bash
pip install .
```

## Contributors
- Gianluca Mazza
- Matteo Garbelli

## License
This project is licensed under the MIT License - see the LICENSE file for details.