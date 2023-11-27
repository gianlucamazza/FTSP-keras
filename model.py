from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout


def build_model(input_shape):
    """
    Build a LSTM model with 2 layers and 50 neurons

    Parameters:
    input_shape (tuple): Shape of input data.

    Returns:
    keras.Sequential: LSTM model.
    """
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model


def main():
    input_shape = (50, 10)
    model = build_model(input_shape)
    model.save('models/bitcoin_prediction_model.keras')


if __name__ == "__main__":
    main()
