# train.py
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from model import build_model


def load_dataset(file_path):
    df = pd.read_csv(file_path, index_col='Date')
    print(f"Loaded dataset shape: {df.shape}")
    df.dropna(inplace=True)
    print(f"Cleaned dataset shape: {df.shape}")
    return df


def split_data(df, train_window, overlap, input_timesteps, validation_split):
    split_idx = int(len(df) * (1 - validation_split))
    df_train = df.iloc[:split_idx]
    df_val = df.iloc[split_idx:]

    train_data = []
    val_data = []

    start = 0
    while start + train_window <= len(df_train):
        end_train = start + train_window

        if end_train - start >= input_timesteps:
            train_data.append(df_train.iloc[start:end_train].values)

        start += train_window - overlap

    start = 0
    while start + train_window <= len(df_val):
        end_train = start + train_window

        if end_train - start >= input_timesteps:
            val_data.append(df_val.iloc[start:end_train].values)

        start += train_window - overlap

    return np.array(train_data, dtype=object), np.array(val_data, dtype=object)


def prepare_data(data, input_timesteps, num_features):
    X, y = [], []
    for window in data:
        if len(window) >= input_timesteps:
            for i in range(len(window) - input_timesteps):
                X.append(window[i:i + input_timesteps, :num_features])
                y.append(window[i + input_timesteps - 1, -1])
    X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    print(f"Prepared data - X shape: {X.shape}, y shape: {y.shape}")
    return X, y


def main():
    df = load_dataset('data/scaled_data.csv')

    # Split the data
    train_window, overlap, timesteps, validation_split = 60, 30, 50, 0.2
    train_data, val_data = split_data(df, train_window, overlap, timesteps, validation_split)

    # Prepare the data for the LSTM model
    num_features = 15
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
    plt.figure(figsize=(12, 6))
    plt.plot(y_val, color='blue', label='True Values')
    plt.plot(predictions, color='red', label='Predictions')
    plt.title('Bitcoin Price Prediction Model Performance')
    plt.xlabel('Time Steps')
    plt.ylabel('Bitcoin Price')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Save the model
    model.save('models/bitcoin_prediction_model.keras')


if __name__ == '__main__':
    main()
