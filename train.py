# train.py
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from model import build_model


def load_dataset(file_path):
    df = pd.read_csv(file_path, index_col='Date')
    print(f"Loaded dataset shape: {df.shape}")  # Debugging print
    return df


def split_data(df, train_window, val_window, overlap, input_timesteps):
    train_data = []
    val_data = []

    start = 0
    while start + train_window <= len(df):
        end_train = start + train_window
        end_val = min(end_train + val_window, len(df))

        # Append training data
        if end_train - start >= input_timesteps:
            train_data.append(df.iloc[start:end_train].values)

        # Append validation data if there's enough data left
        if len(df) - end_train >= input_timesteps:
            val_data.append(df.iloc[end_train:end_val].values)
            print(f"Validation window added: start={end_train}, end={end_val}, length={len(df.iloc[end_train:end_val].values)}")  # Debugging print

        start += train_window - overlap

    print(f"Split data - Train samples: {len(train_data)}, Validation samples: {len(val_data)}")
    return np.array(train_data, dtype=object), np.array(val_data, dtype=object)


def prepare_data(data, input_timesteps, num_features):
    X, y = [], []
    for window in data:
        if len(window) >= input_timesteps:  # Ensure window length meets the minimum requirement
            for i in range(len(window) - input_timesteps):
                X.append(window[i:i + input_timesteps, :num_features])
                y.append(window[i + input_timesteps - 1, -1])
    X, y = np.array(X), np.array(y)
    print(f"Prepared data - X shape: {X.shape}, y shape: {y.shape}")  # Debugging print
    return X, y


def main():
    df = load_dataset('data/scaled_data.csv')

    # Define the window sizes and overlap
    train_window, val_window, overlap, timesteps = 60, 30, 30, 50

    # Split the data
    train_data, val_data = split_data(df, train_window, val_window, overlap, timesteps)

    # Prepare the data for the LSTM model
    num_features = df.shape[1] - 1
    X_train, y_train = prepare_data(train_data, timesteps, num_features)
    X_val, y_val = prepare_data(val_data, timesteps, num_features)

    # Ensure there is data to train on
    if X_train.size == 0 or y_train.size == 0:
        raise ValueError("Training data is empty.")
    if X_val.size == 0 or y_val.size == 0:
        raise ValueError("Validation data is empty.")

    # Build and train the model
    model = build_model((timesteps, num_features), neurons=50, dropout=0.2)
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

    # Evaluate the model
    predictions = model.predict(X_val)
    mse = mean_squared_error(y_val, predictions)
    print(f'Mean Squared Error: {mse}')

    # Plot predictions
    plt.plot(y_val, label='True Values')
    plt.plot(predictions, label='Predictions')
    plt.title('Model Predictions vs True Values')
    plt.legend()
    plt.show()

    # Save the model
    model.save('models/trained_model.h5')


if __name__ == '__main__':
    main()
