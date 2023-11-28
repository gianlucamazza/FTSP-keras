# train.py
import pandas as pd
from keras.models import load_model
from keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np
import matplotlib.pyplot as plt
import joblib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_data(filename, split_date='2020-01-01'):
    """
    Loads Bitcoin price data from a CSV file.

    Parameters:
    file_path (str): Path to the CSV file containing the data.

    Returns:
    train_data (pandas.DataFrame): Dataframe containing the training data.
    test_data (pandas.DataFrame): Dataframe containing the test data.
    """
    df = pd.read_csv(filename)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.dropna()
    train_data = df[df.index < split_date]
    test_data = df[df.index >= split_date]
    return train_data, test_data


def create_dataset(df, feature_columns, target_column, time_steps=1):
    """
    Creates a dataset for training and testing.

    Parameters:
    df (pandas.DataFrame): Dataframe containing Bitcoin price data.
    feature_column (str): Name of the column containing the feature data.
    target_column (str): Name of the column containing the target data.
    time_steps (int): Number of time steps to look back.

    Returns:
    keras.preprocessing.sequence.TimeseriesGenerator: A generator that generates batches of temporal data.
    """
    X = df[feature_columns].values
    y = df[target_column].values
    return TimeseriesGenerator(X, y, length=time_steps, batch_size=batch_size)


def get_numeric_columns(df):
    """
    Returns a list of column names containing numeric data.

    Parameters:
    df (pandas.DataFrame): Dataframe containing Bitcoin price data.

    Returns:
    list: List of column names containing numeric data.
    """
    return df.select_dtypes(include=['number']).columns


def load_scaler(path):
    """
    Loads a pre-fitted MinMaxScaler from a file.

    Parameters:
    path (str): Path to the saved scaler file.

    Returns:
    MinMaxScaler: The loaded scaler.
    """
    return joblib.load(path)


def train_model(epochs=10, batch_size=32):
    try:
        train_data, test_data = load_data('data/scaled_data.csv')
        feature_columns = ['Close', 'MA50', 'MA200', 'Returns', 'Volatility', 'MA20', 'Upper', 'Lower', 'RSI', 'MACD']
        target_column = 'Close'

        model = load_model('models/bitcoin_prediction_model.keras')

        time_steps = 50
        train_generator = create_dataset(train_data, feature_columns, target_column, time_steps, batch_size)
        test_generator = create_dataset(test_data, feature_columns, target_column, time_steps, batch_size)

        model.fit(train_generator, epochs=epochs, validation_data=test_generator, steps_per_epoch=len(train_generator), validation_steps=len(test_generator))

        loss = model.evaluate(test_generator)
        logging.info(f'Test Loss: {loss}')

        model.save('models/bitcoin_prediction_model_trained.keras')
    except Exception as e:
        logging.error(f"An error occurred during training: {e}")


def predict_future_prices(model, last_data, future_steps=30, feature_columns=None):
    """
    Predicts future prices for a given number of days.

    Parameters:
    model (keras.Sequential): Trained LSTM model.
    last_data (pandas.DataFrame): Dataframe containing the last data point.
    future_steps (int): Number of days to predict.
    feature_columns (list): List of feature column names.

    Returns:
    dict: Dictionary containing the predicted prices and dates.
    """
    future_predictions = {}
    last_input = last_data[feature_columns].tail(1).values.reshape(1, -1)

    for steps in range(1, future_steps + 1):
        predicted_price = model.predict(last_input)[0][0]
        last_input = np.roll(last_input, -1)
        last_input[0, -1] = predicted_price
        future_predictions[last_data.index[-1] + pd.Timedelta(days=steps)] = predicted_price

    return future_predictions


def visualize(test_data, future_predictions):
    plt.figure(figsize=(12, 6))
    plt.plot(test_data.index, test_data['Close'], label='Actual')
    prediction_dates = list(future_predictions.keys())
    prediction_values = list(future_predictions.values())
    plt.plot(prediction_dates, prediction_values, label='Predicted')
    plt.legend()
    plt.show()


def evaluate():
    try:
        test_data = load_data('data/scaled_data.csv')[1]
        if test_data.empty:
            raise ValueError("Test data is empty.")

        model = load_model('models/bitcoin_prediction_model_trained.keras')

        feature_columns = ['Close', 'MA50', 'MA200', 'Returns', 'Volatility', 'MA20', 'Upper', 'Lower', 'RSI', 'MACD']
        target_column = 'Close'
        time_steps = 50
        test_generator = create_dataset(test_data, feature_columns, target_column, time_steps)

        loss = model.evaluate(test_generator)
        logging.info(f'Test Loss: {loss}')
    except Exception as e:
        logging.error(f"An error occurred during evaluation: {e}")


if __name__ == "__main__":
    train_model()

    # Load the model and data
    model = load_model('models/bitcoin_prediction_model_trained.keras')
    test_data, _ = load_data('data/scaled_data.csv')

    # Predict future prices
    future_steps = 30  # Number of days to predict
    feature_columns = ['Close', 'MA50', 'MA200', 'Returns', 'Volatility', 'MA20', 'Upper', 'Lower', 'RSI', 'MACD']
    future_predictions = predict_future_prices(model, test_data, future_steps, feature_columns)

    # Visualize the predictions
    visualize(test_data, future_predictions)

    # Evaluate the model
    evaluate()

