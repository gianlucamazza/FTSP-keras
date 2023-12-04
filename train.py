# train.py
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint
from model import build_model
from model import prepare_callbacks
from data_preparation import columns_to_scale as columns
from keras.models import load_model
from sklearn.model_selection import TimeSeriesSplit


def load_dataset(file_path):
    df = pd.read_csv(file_path, index_col='Date')
    print(f"Loaded dataset shape: {df.shape}")
    df.ffill(inplace=True)
    return df


def create_windowed_data(df, start_index, end_index, timesteps):
    X, y = [], []
    for i in range(start_index + timesteps, end_index):
        X.append(df.iloc[i - timesteps:i].values)
        y.append(df.iloc[i, 0])
    return np.array(X), np.array(y)


def evaluate_model(model, x_test, y_test, scaler):
    y_pred = model.predict(x_test)

    target_scaler = MinMaxScaler()
    target_scaler.min_, target_scaler.scale_ = scaler.min_[0], scaler.scale_[0]
    target_scaler.data_min_, target_scaler.data_max_ = scaler.data_min_[0], scaler.data_max_[0]
    target_scaler.data_range_ = scaler.data_range_[0]

    y_pred = target_scaler.inverse_transform(y_pred)
    y_test = target_scaler.inverse_transform(y_test.reshape(-1, 1))

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE: {rmse:.2f}")
    return y_pred, rmse


def plot_results(y_test, y_pred):
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.legend()
    plt.show()


def main(ticker='BTC-USD', n_splits=5):

    paths = {
        'model': f'models/model_{ticker}.keras',
        'data': f'data/scaled_data_{ticker}.csv',
        'scaler': f'scalers/feature_scaler_{ticker}.pkl'
    }

    parameters = {
        'train_timesteps': 60,
        'test_timesteps': 30,
        'features': len(columns),
        'columns': columns
    }

    df = load_dataset(paths['data'])
    df = df[parameters['columns']]
    scaler = joblib.load(paths['scaler'])

    df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)

    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    best_model_path = f"{paths['model']}_best.keras"
    model_checkpoint = ModelCheckpoint(filepath=best_model_path, monitor='val_loss', save_best_only=True)

    total_length = len(df_scaled)
    X, y = create_windowed_data(df_scaled, 0, total_length, parameters['train_timesteps'])

    y = y.reshape(-1, 1)  # Reshape y to match the shape of the output layer

    tscv = TimeSeriesSplit(n_splits=n_splits)
    model = build_model((parameters['train_timesteps'], parameters['features']), neurons=100, dropout=0.3,
                        additional_layers=2, bidirectional=True)

    history_list = []

    for i, (train_index, test_index) in enumerate(tscv.split(X)):
        print(f"Training on fold {i + 1}/{n_splits}...")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # No need to fit the scaler again, just transform the data
        X_train_scaled = X_train
        X_test_scaled = X_test

        history = model.fit(
            X_train_scaled, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test_scaled, y_test),
            callbacks=[early_stopping, model_checkpoint, prepare_callbacks(ticker, epoch=10, val_loss='val_loss')],
            verbose=2
        )
        history_list.append(history)

        # Load the best model once after all folds have been trained
    best_model = load_model(best_model_path)

    # Evaluate the best model
    rmse_list = []

    # Use either a separate test set or the last fold as the test set for final evaluation
    X_test_scaled = X[-1]  # Assuming the last fold is for testing
    y_test_scaled = y[-1]

    y_pred_scaled = best_model.predict(X_test_scaled)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_test = scaler.inverse_transform(y_test_scaled.reshape(-1, 1))

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE: {rmse:.2f}")
    rmse_list.append(rmse)

    plot_results(y_test, y_pred)

    average_rmse = np.mean(rmse_list)
    print(f"Average RMSE: {average_rmse:.2f}")


if __name__ == '__main__':
    main(ticker='BTC-USD')
