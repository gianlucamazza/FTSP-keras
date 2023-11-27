import pandas as pd
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt


def load_model_and_predict(model, input_data):
    """
    Loads a model and predicts the output for a given input.
    """
    prediction = model.predict(input_data)
    return prediction


def main():
    df = pd.read_csv('data/scaled_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    model = load_model('bitcoin_prediction_model_trained.h5')

    feature_columns = ['Close', 'MA50', 'MA200', 'Returns', 'Volatility', 'MA20', 'Upper', 'Lower', 'RSI', 'MACD']
    time_steps = 50

    last_input = df[feature_columns].values[-time_steps:].reshape(1, time_steps, -1)

    future_steps = [1, 7, 30]
    future_predictions = {}

    last_date = df['Date'].iloc[-1]  # Ultima data nel DataFrame

    for steps in future_steps:
        future_input = last_input.copy()
        predictions = []
        future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, steps + 1)]
        for _ in range(steps):
            prediction = load_model_and_predict(model, future_input)[0][0]
            predictions.append(prediction)

            new_input = np.roll(future_input, -1, axis=1)
            new_input[0, -1, 0] = prediction
            future_input = new_input

        future_predictions[steps] = (future_dates, predictions)

    plt.figure(figsize=(12, 6))
    for steps, (dates, predictions) in future_predictions.items():
        plt.plot(dates, predictions, label=f'Predicted {steps} days')
    plt.plot(df['Date'][-(time_steps + 30):], df['Close'].values[-(time_steps + 30):], label='Actual')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
