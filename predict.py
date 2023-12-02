import pandas as pd
import numpy as np
import tensorflow as tf
from model import build_model


def load_dataset(file_path):
    df = pd.read_csv(file_path, index_col='Date')
    df.dropna(inplace=True)
    return df


def prepare_data_for_prediction(df, input_timesteps, num_features):
    if len(df) < input_timesteps:
        raise ValueError(f"Not enough data to create a window of {input_timesteps} timesteps")

    last_window = df.iloc[-input_timesteps:]

    if last_window.shape[1] != num_features:
        raise ValueError(f"Expected {num_features} features, but got {last_window.shape[1]}")

    X = last_window.values.reshape(1, input_timesteps, num_features)
    return X


def predict(model_path, data_path, input_timesteps, num_features):
    model = tf.keras.models.load_model(model_path)

    df = load_dataset(data_path)
    X = prepare_data_for_prediction(df, input_timesteps, num_features)

    prediction = model.predict(X)
    return prediction


if __name__ == '__main__':
    model_path = 'models/bitcoin_prediction_model.keras'
    data_path = 'data/scaled_data.csv'
    input_timesteps = 50
    num_features = 14

    prediction = predict(model_path, data_path, input_timesteps, num_features)
    print("Predicted Value:", prediction)
