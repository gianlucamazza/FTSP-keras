from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout


def build_model(input_shape, neurons=50, dropout=0.2, optimizer='adam', loss='mean_squared_error', metrics=['mae']):
    """
    Build a LSTM model with 2 layers and 50 neurons

    Parameters:
    input_shape (tuple): Shape of input data.

    Returns:
    keras.Sequential: LSTM model.
    """
    model = Sequential()
    model.add(LSTM(neurons, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(LSTM(neurons, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def main():
    input_shape = (50, 10)
    model = build_model(input_shape)
    model.save('models/bitcoin_prediction_model.keras')


if __name__ == "__main__":
    main()
