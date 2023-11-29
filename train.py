# train.py
from typing import Tuple, Any

import pandas as pd
import joblib
import logging
import os

from keras import Sequential
from keras.models import load_model
from keras.preprocessing.sequence import TimeseriesGenerator
from model import build_model
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_data(filename: str, split_date: str = '2020-01-01') -> (pd.DataFrame, pd.DataFrame):
    """
    Load Bitcoin price data from a CSV file and split into training and test sets.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file {filename} does not exist.")

    df = pd.read_csv(filename, parse_dates=['Date'], index_col='Date')
    df = df.sort_index()  # Ensure the data is sorted by date

    if split_date not in df.index:
        raise ValueError(f"The split date {split_date} is not in the data range.")

    train_data = df.loc[df.index < split_date]
    test_data = df.loc[df.index >= split_date]

    if train_data.empty or test_data.empty:
        raise ValueError("Training or testing set is empty after split. Check your split date and data.")

    return train_data, test_data


def create_dataset(df: pd.DataFrame, feature_columns: list, target_column: str, time_steps: int = 1,
                   batch_size: int = 32) -> TimeseriesGenerator:
    """
    Create a dataset for model training and evaluation using a time series generator.
    """
    if len(df) < time_steps:
        raise ValueError(f"The DataFrame is too small for the given time_steps: {time_steps}")

    X = df[feature_columns].values
    y = df[[target_column]].values
    return TimeseriesGenerator(X, y, length=time_steps, batch_size=batch_size)


def get_numeric_columns(df: pd.DataFrame) -> list:
    """
    Retrieve column names for numeric data types within the DataFrame.
    """
    return df.select_dtypes(include=['number']).columns.tolist()


def load_scaler(path: str) -> MinMaxScaler:
    """
    Load a pre-fitted MinMaxScaler from a file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"The scaler file {path} does not exist.")
    return joblib.load(path)


def create_or_load_model(input_shape: tuple, model_path: str = 'models/bitcoin_prediction_model.keras') -> 'Sequential':
    """
    Load an existing model or create a new one if it doesn't exist.
    """
    if not os.path.exists(model_path):
        logging.info("No model found. Creating a new model.")
        model = build_model(input_shape)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
    else:
        model = load_model(model_path)
        logging.info("Model loaded successfully.")
    return model


def train_model(model: 'Sequential', train_generator: TimeseriesGenerator, test_generator: TimeseriesGenerator,
                epochs: int = 10) -> tuple[Sequential, Any]:
    """
    Train the LSTM model.
    """
    history = model.fit(train_generator, epochs=epochs, validation_data=test_generator,
                        steps_per_epoch=len(train_generator), validation_steps=len(test_generator))
    return model, history


def evaluate(model: 'Sequential', test_generator: TimeseriesGenerator) -> float:
    """
    Evaluates the model on the test data.
    """
    loss, metric = model.evaluate(test_generator)
    logging.info(f'Test Loss: {loss}, Test Metric: {metric}')
    return loss


if __name__ == "__main__":
    # Main execution sequence
    data_path = 'data/scaled_data.csv'
    model_path = 'models/bitcoin_prediction_model.keras'
    scaler_path = 'models/scaler.pkl'

    # Load and prepare data
    train_data, test_data = load_data(data_path)
    feature_columns = get_numeric_columns(train_data)
    target_column = 'Close'

    # Model creation or loading
    input_shape = (50, len(feature_columns))
    model = create_or_load_model(input_shape, model_path)

    # Data generators
    train_generator = create_dataset(train_data, feature_columns, target_column)
    test_generator = create_dataset(test_data, feature_columns, target_column)

    # Model training and evaluation
    model, history = train_model(model, train_generator, test_generator)
    evaluate(model, test_generator)