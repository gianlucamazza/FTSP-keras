# model.py
import argparse

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout


def build_model(input_shape, neurons1=64, neurons2=8,dropout=0.1, optimizer='adam', loss='mean_squared_error', metrics=None):
    """
    Builds a LSTM model.

    Parameters:
    input_shape (tuple): Shape of the input data.
    neurons (int): Number of neurons.
    dropout (float): Dropout rate.
    optimizer (str): Optimizer to use.
    loss (str): Loss function to use.
    metrics (list): List of metrics to use.

    Returns:
    keras.Sequential: Compiled Keras model.
    """
    if metrics is None:
        metrics = ['mae']
    model = Sequential()
    model.add(LSTM(neurons1, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(LSTM(neurons2, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def main(input_shape, model_path):
    model = build_model(input_shape)
    model.save(model_path)
    print("Model built and saved successfully.")
    print(model.summary())


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_shape", nargs=2, type=int, default=[50, 15])
    parser.add_argument("--model_path", type=str, default='models/bitcoin_prediction_model.keras')
    args = parser.parse_args()
    input_shape = tuple(args.input_shape)
    main(input_shape, args.model_path)
