import pandas as pd
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
from data_preparation import columns_to_scale as columns


def load_dataset(file_path):
    df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    return df


def select_features(df, feature_columns):
    return df[feature_columns]


def reshape_data(last_window, timesteps, features):
    return last_window.values.reshape(1, timesteps, features)


def predict_price(model, data, scaler):
    prediction = model.predict(data)
    return scaler.inverse_transform(prediction)[0, 0]


def plot_prediction(ticker, dates, historical_prices, predicted_value, future_predict_date):
    plt.figure(figsize=(12, 6))
    plt.plot(dates, historical_prices, label='Historical Close')

    future_predict_date = pd.bdate_range(start=dates[-1], periods=2)[
        1] if future_predict_date.weekday() >= 5 else future_predict_date

    plt.plot([future_predict_date], [predicted_value], 'ro-', label='Predicted Close')

    plt.xlim([dates[0], future_predict_date + pd.Timedelta(days=1)])
    plt.annotate(
        f'{predicted_value:.2f}',
        xy=(future_predict_date, predicted_value),
        xytext=(future_predict_date + pd.Timedelta(days=1), predicted_value),
        arrowprops=dict(facecolor='black', arrowstyle='->'),
    )

    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'{ticker} Price Prediction')
    plt.show()


def main(ticker='BTC-USD'):
    paths = {
        'model': f'models/model_{ticker}.keras',
        'data': f'data/processed_data_{ticker}.csv',
        'scaler': f'scalers/scaler_{ticker}.pkl'
    }

    parameters = {
        'timesteps': 50,
        'features': len(columns),
        'columns': columns
    }

    dataset = load_dataset(paths['data'])
    feature_data = select_features(dataset, parameters['columns'])

    model_input = reshape_data(feature_data.iloc[-parameters['timesteps']:], parameters['timesteps'],
                               parameters['features'])

    model = tf.keras.models.load_model(paths['model'])
    scaler = joblib.load(paths['scaler'])
    price_prediction = predict_price(model, model_input, scaler)

    historical_closing_prices = scaler.inverse_transform(
        feature_data['Close'][-parameters['timesteps']:].values.reshape(-1, 1)
    ).flatten()

    historical_dates = dataset.index[-parameters['timesteps']:]
    next_business_day = pd.bdate_range(start=historical_dates[-1], periods=2)[1] \
        if historical_dates[-1].weekday() >= 5 else historical_dates[-1]

    plot_prediction(historical_dates, historical_closing_prices, price_prediction, next_business_day)

    print(f"Predicted price for {next_business_day.date()}: {price_prediction:.2f}")


if __name__ == '__main__':
    main(ticker='BTC-USD')
