# predict.py
import pandas as pd
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings


def main():
    df = pd.read_csv('data/scaled_data.csv')
    if df.empty:
        raise ValueError("Dataframe is empty.")
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    model = load_model('models/bitcoin_prediction_model.h5')
    scaler = joblib.load('models/scaler.pkl')
    feature_columns = [
        'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'MA50', 'MA200',
        'Returns', 'Volatility', 'MA20', 'Upper', 'Lower', 'RSI', 'MACD'
    ]

    time_steps = 50
    future_steps = 30
    future_predictions = {}
    close_index = feature_columns.index('Close')  # Index of 'Close' feature

    # Prepare the initial input for prediction: last 'time_steps' entries from the DataFrame
    input_sequence_df = df[feature_columns].tail(time_steps)

    for step in range(1, future_steps + 1):
        # Suppress warnings from the scaler about the feature names
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            input_sequence_scaled = scaler.transform(input_sequence_df).reshape(1, time_steps, -1)

        # Predict the next 'Close' value
        prediction_scaled = model.predict(input_sequence_scaled)[0][0]

        # Prepare the dummy input for inverse scaling
        dummy_input_scaled = np.zeros((1, len(feature_columns)))
        dummy_input_scaled[0, close_index] = prediction_scaled

        # Convert the dummy input to a DataFrame to inverse transform and get the actual 'Close' value
        dummy_input_df = pd.DataFrame(dummy_input_scaled, columns=feature_columns)
        prediction = scaler.inverse_transform(dummy_input_df)[0, close_index]

        # Record the prediction with the corresponding future date
        future_date = df.index[-1] + pd.DateOffset(days=step)
        future_predictions[future_date] = prediction

        # Update the input sequence with the predicted 'Close' value
        next_row = pd.DataFrame(np.zeros((1, len(feature_columns))), columns=feature_columns)
        next_row.iloc[0, close_index] = prediction_scaled  # Use the scaled predicted value

        # Append the predicted 'Close' value to the input sequence and discard the oldest entry
        input_sequence_df = pd.concat([input_sequence_df.iloc[1:], next_row])

    # Visualize the predictions with the actual values from the last part of the dataset and the predicted values
    plt.figure(figsize=(12, 6))
    plt.plot(df.index[-50:], df['Close'].tail(50), label='Actual')
    prediction_dates = list(future_predictions.keys())
    prediction_values = list(future_predictions.values())
    plt.plot(prediction_dates, prediction_values, label='Predicted')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
