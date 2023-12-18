import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
from data_preparation import COLUMNS_TO_SCALE as FEATURES
from keras.models import load_model
from logger import setup_logger

logger = setup_logger('predict_logger', 'logs', 'predict.log')


def load_dataset(file_path):
    try:
        df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        logger.info(f"Dataset loaded successfully from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading dataset from {file_path}: {e}")
        raise


def select_features(df, feature_columns):
    return df[feature_columns]


def reshape_data(data, steps, features):
    return data[-steps:].reshape(1, steps, features)


def predict_price(model, reshaped_data):
    prediction = model.predict(reshaped_data)
    return prediction[0, 0]


def predict_next_days(model, initial_data, feature_scaler, close_scaler, days=30, steps=60):
    future_predictions = []
    input_data = initial_data[-steps:].copy()

    for _ in range(days):
        scaled_input = feature_scaler.transform(input_data[:, :-1])
        model_input = reshape_data(scaled_input, steps, len(FEATURES) - 1)
        predicted_scaled_price = predict_price(model, model_input)

        predicted_price = close_scaler.inverse_transform([[predicted_scaled_price]])[0, 0]
        future_predictions.append(predicted_price)

        new_row = np.zeros_like(input_data[0])
        new_row[-1] = predicted_scaled_price
        input_data = np.vstack((input_data[1:], [new_row]))

    return future_predictions


def plot_predictions(dates, historical_prices, future_prices):
    plt.figure(figsize=(12, 6))
    plt.plot(dates, historical_prices, color='blue', label='Historical Close')
    plt.plot(dates, future_prices, color='red', label='Predicted Close')
    plt.legend()
    plt.title('Future Price Predictions')
    plt.show()


def main(ticker='BTC-USD'):
    paths = {
        'model': f'models/model_{ticker}.keras',
        'data': f'data/scaled_data_{ticker}.csv',
        'feature_scaler': f'scalers/feature_scaler_{ticker}.pkl',
        'close_scaler': f'scalers/close_scaler_{ticker}.pkl'
    }

    # Load and preprocess dataset
    df = load_dataset(paths['data'])
    if 'Close' not in df.columns:
        raise ValueError("Column 'Close' not found in the DataFrame.")

    # Include 'Close' in the features to select
    all_features = FEATURES + ['Close']
    df = select_features(df, all_features)

    # Load scalers
    feature_scaler = joblib.load(paths['feature_scaler'])
    close_scaler = joblib.load(paths['close_scaler'])

    # Apply scalers
    df[FEATURES] = feature_scaler.transform(df[FEATURES])
    df['Close'] = close_scaler.transform(df[['Close']])

    # Load model
    model = load_model(paths['model'])

    # Predict future prices
    future_prices = predict_next_days(model, df.values, feature_scaler, close_scaler)

    # Plot predictions
    dates = pd.date_range(start=df.index[-1], periods=len(future_prices) + 1)[1:]
    plot_predictions(dates, df['Close'].tail(30).values, future_prices)

    plt.savefig(f'predictions/{ticker}_predictions.png')
    logger.info("Prediction plot saved successfully.")


if __name__ == '__main__':
    main(ticker='BTC-USD')
