# train.py
import argparse
import datetime
import pandas as pd
import joblib
import logging
import os
from keras.models import load_model
from keras.preprocessing.sequence import TimeseriesGenerator
from model import build_model
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import gc

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def build_model(input_shape, neurons1=64, neurons2=8, dropout=0.1, optimizer='adam', loss='mean_squared_error', metrics=['mae']):
    model = Sequential()
    model.add(LSTM(neurons1, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(LSTM(neurons2, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def create_dataset(df: pd.DataFrame, features: list, target: str, time_steps: int = 50,
                   batch_size: int = 32) -> TimeseriesGenerator:
    """
    Create a dataset for model training and evaluation using a time series generator.
    """
    X = df[features].values
    y = df[[target]].values
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


def train_model(model: Sequential, train_generator: TimeseriesGenerator, test_generator: TimeseriesGenerator, model_path: str, epochs: int = 10) -> Sequential:
    """
    Train the LSTM model.
    """
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3),
        ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', mode='min')
    ]
    history = model.fit(train_generator, epochs=epochs, validation_data=test_generator, callbacks=callbacks)
    return model, history


def evaluate(model: 'Sequential', test_generator: TimeseriesGenerator) -> float:
    """
    Evaluates the model on the test data.
    """
    loss, metric = model.evaluate(test_generator)
    logging.info(f'Test Loss: {loss}, Test Metric: {metric}')
    return loss


def generate_data_splits(df, start_date, end_date, k=5):
    total_days = (end_date - start_date).days
    fold_size = total_days // k

    for i in range(k):
        train_end = start_date + datetime.timedelta(days=fold_size * i)
        test_end = start_date + datetime.timedelta(days=fold_size * (i + 1))

        if i == k - 1:
            test_end = end_date

        train_df = df.loc[start_date:train_end]
        test_df = df.loc[train_end + datetime.timedelta(days=1):test_end]

        yield train_df, test_df


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='data/scaled_data.csv')
    parser.add_argument("--model_path", type=str, default='models/bitcoin_prediction_model.keras')
    parser.add_argument("--scaler_path", type=str, default='models/scaler.pkl')

    args = parser.parse_args()

    # Load and prepare the data
    df = pd.read_csv(args.data_path, parse_dates=['Date'], index_col='Date')
    df.sort_index(inplace=True)
    df.fillna(method='ffill', inplace=True)

    # Load the scaler if it exists
    if args.scaler_path and os.path.exists(args.scaler_path):
        scaler = joblib.load(args.scaler_path)
        df[df.columns] = scaler.transform(df)

    start_date = datetime.datetime(2014, 9, 17)
    end_date = datetime.datetime(2023, 11, 27)

    performance_metrics = []

    for train_df, test_df in generate_data_splits(df, start_date, end_date):
        feature_columns = train_df.select_dtypes(include=['number']).columns.tolist()
        target_column = 'Close'

        if len(train_df) < 50:
            logging.info("Skipping fold due to insufficient data.")
            continue

        input_shape = (50, len(feature_columns))
        train_generator = create_dataset(train_df, feature_columns, target_column)
        test_generator = create_dataset(test_df, feature_columns, target_column)

        model = build_model(input_shape)

        callbacks = [EarlyStopping(monitor='val_loss', patience=3)]
        history = model.fit(train_generator, epochs=10, validation_data=test_generator, callbacks=callbacks)

        loss, metric = model.evaluate(test_generator)
        logging.info(f'Test Loss: {loss}, Test Metric: {metric}')
        performance_metrics.append((loss, metric))

        del model
        gc.collect()

    logging.info("Cross-validation complete. Performance metrics: {}".format(performance_metrics))
