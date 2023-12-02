import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt


def load_dataset(file_path):
    df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    df.dropna(inplace=True)
    return df


def prepare_data_for_prediction(df, input_timesteps, num_features):
    # columns: Date,Open,High,Low,Close,Adj Close,Volume,MA50,MA200,Returns,Volatility,MA20,Upper,Lower,RSI,MACD
    df = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'MA50', 'MA200', 'Returns', 'Volatility',
             'MA20', 'Upper', 'Lower', 'RSI', 'MACD']]

    if len(df) < input_timesteps:
        raise ValueError(f"Not enough data to create a window of {input_timesteps} timesteps")

    last_window = df.iloc[-input_timesteps:]

    if last_window.shape[1] != num_features:
        raise ValueError(f"Expected {num_features} features, but got {last_window.shape[1]}")

    X = last_window.values.reshape(1, input_timesteps, num_features)
    return X


def predict(model_path, data_path, scaler_path, input_timesteps, num_features):
    df = load_dataset(data_path)
    X = prepare_data_for_prediction(df, input_timesteps, num_features)
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)

    prediction = model.predict(X)

    dummy_input = np.zeros((1, num_features))
    dummy_input[0, 3] = prediction[0, 0]  # Assuming index 3 is for 'Close'
    predicted_price = scaler.inverse_transform(dummy_input)[0, 3]

    dates = df.index[-input_timesteps:]

    dummy_array = np.zeros((input_timesteps, num_features))
    dummy_array[:, 3] = df['Close'].iloc[-input_timesteps:].values  # Assuming index 3 is for 'Close'

    historical_close_inversed = scaler.inverse_transform(dummy_array)[:, 3]

    return dates, historical_close_inversed, predicted_price


def plot_prediction(dates, historical_close_inversed, prediction):
    plt.figure(figsize=(12, 6))
    plt.plot(dates, historical_close_inversed, label='Historical Close')
    plt.plot([dates[-1], dates[-1] + pd.Timedelta(days=1)], [historical_close_inversed[-1], prediction], 'ro-', label='Predicted Close')
    plt.legend()

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Bitcoin Price Prediction')
    plt.show()


if __name__ == '__main__':
    model_path = 'models/bitcoin_prediction_model.keras'
    data_path = 'data/scaled_data.csv'
    scaler_path = 'models/scaler.pkl'
    input_timesteps = 50
    num_features = 15
    dates, historical_close_inversed, prediction = predict(model_path, data_path, scaler_path, input_timesteps, num_features)
    plot_prediction(dates, historical_close_inversed, prediction)
    print(f"Predicted price: {prediction:.2f}")