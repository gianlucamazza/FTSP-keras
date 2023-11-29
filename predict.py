# predict.py
import argparse

import pandas as pd
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import os


def load_data(file_path='data/scaled_data.csv'):
    if not os.path.exists(file_path):
        raise FileNotFoundError("File not found.")
    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError("Dataframe is empty.")
    df.dropna(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df


def load_model_and_scaler(model_path='models/bitcoin_prediction_model.keras'):
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not found.")

    model = load_model(model_path)
    return model


def generate_predictions(df, model, feature_columns, time_steps, future_steps):
    if len(df) < time_steps:
        raise ValueError(f"Not enough data to generate input sequence. Required: {time_steps}, available: {len(df)}")

    future_predictions = {}
    close_index = feature_columns.index('Close')
    input_sequence_df = df[feature_columns].tail(time_steps)

    for step in range(1, future_steps + 1):
        input_sequence = input_sequence_df.values.reshape(1, time_steps, -1)
        prediction = model.predict(input_sequence)[0][0]

        future_date = df.index[-1] + pd.DateOffset(days=step)
        future_predictions[future_date] = prediction

        next_row = pd.DataFrame(np.zeros((1, len(feature_columns))), columns=feature_columns)
        next_row.iloc[0, close_index] = prediction
        input_sequence_df = pd.concat([input_sequence_df.iloc[1:], next_row])

    return future_predictions


def plot_predictions(df, future_predictions):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index[-365:], df['Close'].tail(365), label='Actual') # Plot the last year of actual data
    prediction_dates = list(future_predictions.keys())
    prediction_values = list(future_predictions.values())
    plt.plot(prediction_dates, prediction_values, label='Predicted')
    plt.title('Bitcoin Price Prediction')
    plt.legend()
    plt.show()


def main(data='data/scaled_data.csv', model='models/bitcoin_prediction_model.keras', scaler='models/scaler.pkl'):
    df = load_data(data)
    model = load_model_and_scaler(model)

    feature_columns = [
        'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'MA50', 'MA200',
        'Returns', 'Volatility', 'MA20', 'Upper', 'Lower', 'RSI', 'MACD'
    ]

    time_steps = 50
    future_steps = 30
    future_predictions = generate_predictions(df, model, feature_columns, time_steps, future_steps)
    plot_predictions(df, future_predictions)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='data/scaled_data.csv')
    parser.add_argument("--model", type=str, default='models/bitcoin_prediction_model.keras')
    args = parser.parse_args()

    main(args.data, args.model)

