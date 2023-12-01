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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def build_model(input_shape, neurons1=64, neurons2=8, dropout=0.1, optimizer='adam', loss='mean_squared_error',
                metrics=['mae']):
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
    y = df[[target]].values # The model will predict the target column: example: 'Close'
    return TimeseriesGenerator(X, y, length=time_steps, batch_size=batch_size)


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


def train_model(model: Sequential, train_generator: TimeseriesGenerator, test_generator: TimeseriesGenerator,
                model_path: str, epochs: int = 100) -> Sequential:
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


def generate_data_splits(df, start_date, end_date, overlap=10, k=50):
    total_days = (end_date - start_date).days
    fold_size = total_days // k

    for i in range(k):
        train_end = start_date + datetime.timedelta(days=fold_size * i)
        test_end = start_date + datetime.timedelta(days=fold_size * (i + 1))
        test_end = test_end if i < k - 1 else end_date

        train_df = df.loc[start_date:train_end]
        test_df = df.loc[train_end - datetime.timedelta(days=overlap):test_end]

        yield train_df, test_df


def generate_combined_dataset(df, start_date, end_date, overlap, folds, feature_columns, target):
    all_train_df = pd.DataFrame()
    all_test_df = pd.DataFrame()

    for train_df, test_df in generate_data_splits(df, start_date, end_date, overlap, folds):
        all_train_df = pd.concat([all_train_df, train_df])
        all_test_df = pd.concat([all_test_df, test_df])

    train_generator = create_dataset(all_train_df, feature_columns, target)
    test_generator = create_dataset(all_test_df, feature_columns, target)

    return train_generator, test_generator


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='data/scaled_data.csv')
    parser.add_argument("--model_path", type=str, default='models/bitcoin_prediction_model.keras')
    # parser.add_argument("--scaler_path", type=str, default='models/scaler.pkl')

    args = parser.parse_args()

    # Load and prepare the data
    df = pd.read_csv(args.data_path, parse_dates=['Date'], index_col='Date')
    df.sort_index(inplace=True)
    df.fillna(method='ffill', inplace=True)

    feature_columns = df.select_dtypes(include=['number']).columns.tolist()
    folds = 50
    input_shape = (50, len(feature_columns))
    model = create_or_load_model(input_shape, args.model_path)

    start_date = pd.to_datetime("2015-01-01")
    end_date = pd.to_datetime(df.index.max())

    # Generate data splits
    feature_columns = df.select_dtypes(include=['number']).columns.tolist()
    input_shape = (50, len(feature_columns))
    model = create_or_load_model(input_shape, args.model_path)

    # Genera un unico dataset di training e test
    train_generator, test_generator = generate_combined_dataset(df, start_date, end_date, 10, folds, feature_columns,
                                                                'Close')

    # Addestra il modello
    logging.info("Starting model training...")
    model, history = train_model(model, train_generator, test_generator, args.model_path)

    # Valuta il modello
    evaluate(model, test_generator)

    logging.info("Training completed successfully.")
    model.save(args.model_path)
