import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
from data_preparation import columns_to_scale as columns
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


def reshape_data(last_window, steps, features):
    return last_window.values.reshape(1, steps, features)


def predict_price(model, data, scaler):
    prediction = model.predict(data)
    return scaler.inverse_transform(prediction)[0, 0]


def predict_next_days(model, initial_data, scaler, days=30, steps=60):
    future_predictions = []
    input_data = initial_data.copy()

    for _ in range(days):
        model_input = reshape_data(input_data[-steps:], steps, len(columns))
        predicted_price = predict_price(model, model_input, scaler)
        future_predictions.append(predicted_price)
        new_row = np.append(input_data[-1, 1:], predicted_price)
        input_data = np.vstack((input_data, new_row))

    return future_predictions


def plot_future_predictions(dates, historical_prices, future_dates, future_predictions):
    plt.figure(figsize=(12, 6))
    plt.plot(dates, historical_prices, label='Historical Close')
    plt.plot(future_dates, future_predictions, 'ro-', label='Predicted Close')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Future Price Predictions')
    plt.show()


def main(ticker='BTC-USD'):
    paths = {
        'best_model_path': f'models/model_{ticker}_best.keras',
        'data': f'data/processed_data_{ticker}.csv',
        'scaler': f'scalers/scaler_{ticker}.pkl'
    }

    parameters = {
        'steps': 50,
        'features': len(columns),
        'columns': columns
    }

    dataset = load_dataset(paths['data'])
    feature_data = select_features(dataset, parameters['columns'])

    model_input = feature_data.iloc[-parameters['steps']:]
    reshaped_input = reshape_data(model_input, parameters['steps'], parameters['features'])

    model = load_model(paths['best_model_path'])
    scaler = joblib.load(paths['scaler'])

    historical_closing_prices = scaler.inverse_transform(
        feature_data['Close'][-parameters['steps']:].values.reshape(-1, 1)
    ).flatten()

    historical_dates = dataset.index[-parameters['steps']:]
    future_dates = pd.date_range(start=historical_dates[-1], periods=31, closed='right')

    try:
        predicted_prices = predict_next_days(model, reshaped_input, scaler, days=30, steps=parameters['steps'])
    except Exception as e:
        logger.error(f"Error predicting next 30 days: {e}")
        raise

    plot_future_predictions(historical_dates, historical_closing_prices, future_dates, predicted_prices)

    plt.savefig(f'predictions/{ticker}_predictions.png')
    logger.info("Prediction plot saved successfully.")


if __name__ == '__main__':
    main(ticker='BTC-USD')
