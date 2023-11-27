import pandas as pd
from keras.models import load_model
from keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np
import matplotlib.pyplot as plt
import joblib


def load_data(filename):
    """
    Loads Bitcoin price data from a CSV file.

    Parameters:
    file_path (str): Path to the CSV file containing the data.

    Returns:
    train_data (pandas.DataFrame): Dataframe containing the training data.
    test_data (pandas.DataFrame): Dataframe containing the test data.
    """
    df = pd.read_csv(filename)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.dropna()
    train_data = df[df.index < '2020-01-01']
    test_data = df[df.index >= '2020-01-01']
    return train_data, test_data


def create_dataset(df, feature_column, target_column, time_steps=1):
    """
    Creates a dataset for training and testing.

    Parameters:
    df (pandas.DataFrame): Dataframe containing Bitcoin price data.
    feature_column (str): Name of the column containing the feature data.
    target_column (str): Name of the column containing the target data.
    time_steps (int): Number of time steps to look back.

    Returns:
    keras.preprocessing.sequence.TimeseriesGenerator: A generator that generates batches of temporal data.
    """
    data = df[feature_column].values
    target = df[target_column].values
    return TimeseriesGenerator(data, target, length=time_steps, batch_size=1)


def load_scaler(path):
    """
    Loads a pre-fitted MinMaxScaler from a file.

    Parameters:
    path (str): Path to the saved scaler file.

    Returns:
    MinMaxScaler: The loaded scaler.
    """
    return joblib.load(path)


if __name__ == "__main__":
    train_data, test_data = load_data('data/scaled_data.csv')

    feature_columns = ['Close', 'MA50', 'MA200', 'Returns', 'Volatility', 'MA20', 'Upper', 'Lower', 'RSI', 'MACD']
    target_column = 'Close'

    model = load_model('models/bitcoin_prediction_model.keras')

    time_steps = 50
    train_generator = create_dataset(train_data, feature_columns, target_column, time_steps)
    test_generator = create_dataset(test_data, feature_columns, target_column, time_steps)

    model.fit(train_generator, epochs=10)

    loss = model.evaluate(test_generator)
    print(f'Test Loss: {loss}')

    model.save('models/bitcoin_prediction_model_trained.keras')

    last_input = test_data[feature_columns].values[-time_steps:].reshape(1, time_steps, -1)

    future_steps = [1, 7, 30]
    future_predictions = {}

    last_date = test_data.index[-1]  # Ultima data nel DataFrame

    scaler = load_scaler('models/scaler.pkl')

    for steps in future_steps:
        future_input = last_input.copy()
        predictions = []
        future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, steps + 1)]
        for _ in range(steps):
            prediction_scaled = model.predict(future_input)[0][0]
            prediction = scaler.inverse_transform(prediction_scaled.reshape(-1, 1)).flatten()[0]
            predictions.append(prediction)

            new_input = np.roll(future_input, -1, axis=1)
            new_input[0, -1, 0] = prediction
            future_input = new_input

        future_predictions[steps] = (future_dates, predictions)

    # Plot the predictions
    plt.figure(figsize=(12, 6))
    for steps, (dates, predictions) in future_predictions.items():
        plt.plot(dates, predictions, label=f'Predicted {steps} days')
    plt.plot(test_data.index[-(time_steps + 30):], test_data['Close'].values[-(time_steps + 30):], label='Actual')
    plt.legend()
    plt.show()

