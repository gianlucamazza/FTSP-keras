# predict.py
import pandas as pd
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import joblib


def main():
    df = pd.read_csv('data/scaled_data.csv')
    if df.empty:
        raise ValueError("Dataframe is empty.")
    df['Date'] = pd.to_datetime(df['Date'])
    model = load_model('models/bitcoin_prediction_model_trained.keras')

    feature_columns = ['Close', 'MA50', 'MA200', 'Returns', 'Volatility', 'MA20', 'Upper', 'Lower', 'RSI', 'MACD']
    time_steps = 50
    scaler = joblib.load('models/scaler.pkl')
    last_input = df[feature_columns].values[-time_steps:].reshape(1, time_steps, -1)
    scaled_prediction = np.zeros((1, len(feature_columns)))

    future_steps = [1, 7, 30]
    future_predictions = {}

    last_date = df['Date'].iloc[-1]

    for steps in future_steps:
        future_input = last_input.copy()
        predictions = []
        future_dates = [last_date + pd.Timedelta(days=i) for i in range(steps)]
        for _ in range(steps):
            prediction_scaled = model.predict(future_input)[0][0]
            scaled_prediction = np.zeros((1, scaler.n_features_in_))
            scaled_prediction[0, feature_columns.index('Close')] = prediction_scaled
            future_input = np.roll(future_input, -1, axis=1)
            future_input[0, -1, :] = scaler.inverse_transform(scaled_prediction)[0, :len(feature_columns)]
            prediction = scaler.inverse_transform(scaled_prediction)[0, feature_columns.index('Close')]
            predictions.append(prediction)
        future_predictions[steps] = (future_dates, predictions)

    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Close'], label='Actual')
    for steps, (dates, predictions) in future_predictions.items():
        plt.plot(dates, predictions, label=f'Predicted {steps} days')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()