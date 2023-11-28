# model.py
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout


def build_model(input_shape, neurons=50, dropout=0.2, optimizer='adam', loss='mean_squared_error', metrics=None):
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
    model.add(LSTM(neurons, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(LSTM(neurons, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def main():
    input_shape = (50, 15)
    model = build_model(input_shape)
    model.save('models/bitcoin_prediction_model.keras')


if __name__ == "__main__":
    main()
