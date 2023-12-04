# train.py
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint
from model import build_model
from data_preparation import columns_to_scale as columns
from keras.models import load_model
from sklearn.model_selection import TimeSeriesSplit


def load_dataset(file_path):
    df = pd.read_csv(file_path, index_col='Date')
    print(f"Loaded dataset shape: {df.shape}")
    df.ffill(inplace=True)
    return df


def create_windowed_data(df, timesteps):
    X, y = [], []
    for i in range(timesteps, len(df)):
        X.append(df[i - timesteps:i])
        y.append(df[i, 0])
    return np.array(X), np.array(y)


def calculate_rmse(model, x_test, y_test):
    y_pred = model.predict(x_test)
    return np.sqrt(mean_squared_error(y_test, y_pred))


def train_model(x_train, y_train, x_val, y_val, model_path, parameters):
    model = build_model((parameters['train_timesteps'], parameters['features']), neurons=100, dropout=0.3,
                        additional_layers=2, bidirectional=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    model_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True)
    history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_val, y_val),
                        callbacks=[early_stopping, model_checkpoint], verbose=2)
    return model, history


def plot_history(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.legend()
    plt.show()


def main(ticker='BTC-USD'):
    paths = {'model': f'models/model_{ticker}.keras', 'data': f'data/scaled_data_{ticker}.csv'}
    parameters = {'train_timesteps': 60, 'test_timesteps': 30, 'features': len(columns), 'columns': columns}
    df = load_dataset(paths['data'])
    df = df[parameters['columns']]
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)
    X, y = create_windowed_data(df_scaled, parameters['train_timesteps'])
    y = y.reshape(-1, 1)
    n_splits = (len(df_scaled) - parameters['train_timesteps']) // parameters['test_timesteps']
    if n_splits < 1:
        raise ValueError("Not enough data for even one split!")
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmse_list = []
    history = None
    for i, (train_index, test_index) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model, history = train_model(X_train, y_train, X_test, y_test, paths['model'], parameters)
        best_model = load_model(paths['model'])
        rmse = calculate_rmse(best_model, X_test, y_test)
        rmse_list.append(rmse)
        print(f"RMSE for fold {i + 1}: {rmse:.2f}")
    print(f"Average RMSE across all folds: {np.mean(rmse_list):.2f}")
    plot_history(history)


if __name__ == '__main__':
    main(ticker='BTC-USD')
