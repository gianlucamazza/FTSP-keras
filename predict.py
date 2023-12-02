import pandas as pd
import tensorflow as tf
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


def predict_and_plot(model_path, data_path, input_timesteps, num_features):
    model = tf.keras.models.load_model(model_path)

    df = load_dataset(data_path)
    X = prepare_data_for_prediction(df, input_timesteps, num_features)

    prediction = model.predict(X)
    print(prediction)

    plt.figure(figsize=(10, 6))
    plt.plot(df.index[-input_timesteps:], df['Close'].iloc[-input_timesteps:], label='Historical Data')
    predicted_date = df.index[-1] + pd.Timedelta(days=1)
    plt.plot([predicted_date], prediction[0], 'ro', label='Predicted Price')
    plt.title('Bitcoin Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

    return prediction


if __name__ == '__main__':
    model_path = 'models/bitcoin_prediction_model.keras'
    data_path = 'data/scaled_data.csv'
    input_timesteps = 50
    num_features = 15

    prediction = predict_and_plot(model_path, data_path, input_timesteps, num_features)
    print("Predicted Value:", prediction)