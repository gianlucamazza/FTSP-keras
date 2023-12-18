# model.py
import datetime
import os
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.regularizers import l1_l2
from keras.layers import Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from data_preparation import COLUMNS_TO_SCALE as columns


def build_model(input_shape, neurons=50, dropout=0.2, optimizer='adam', learning_rate=0.001, loss='mean_squared_error', metrics=None, l1_reg=0.0, l2_reg=0.0, additional_layers=0, bidirectional=False):
    if metrics is None:
        metrics = ['mae']

    model = Sequential()

    lstm_layer = LSTM(neurons, return_sequences=True, kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg), input_shape=input_shape)
    layer_to_add = Bidirectional(lstm_layer, merge_mode='concat') if bidirectional else lstm_layer
    model.add(layer_to_add)

    model.add(Dropout(dropout))
    model.add(BatchNormalization())

    for _ in range(additional_layers):
        lstm_layer = LSTM(neurons, return_sequences=True, kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))
        layer_to_add = Bidirectional(lstm_layer, merge_mode='concat') if bidirectional else lstm_layer
        model.add(layer_to_add)
        model.add(Dropout(dropout))
        model.add(BatchNormalization())

    lstm_layer = LSTM(neurons, return_sequences=False)
    final_layer = Bidirectional(lstm_layer, merge_mode='concat') if bidirectional else lstm_layer
    model.add(final_layer)

    model.add(Dropout(dropout))
    model.add(BatchNormalization())
    model.add(Dense(1))

    opt = Adam(learning_rate=learning_rate) if optimizer == 'adam' else optimizer
    model.compile(optimizer=opt, loss=loss, metrics=metrics)

    return model


def prepare_callbacks(ticker, epoch, val_loss):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir = f'models/{ticker}'
    log_dir = f'logs/{ticker}/{timestamp}'
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    callbacks = [
        EarlyStopping(monitor=val_loss, patience=10, verbose=1, restore_best_weights=True),
        ModelCheckpoint(filepath=f'{model_dir}/model_{epoch:02d}-{val_loss}.keras', verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor=val_loss, factor=0.1, patience=5, verbose=1),
        TensorBoard(log_dir=log_dir, histogram_freq=1)
    ]
    return callbacks


def main(ticker='BTC-USD'):
    input_shape = (50, len(columns))
    model = build_model(input_shape, additional_layers=1, bidirectional=True)
    model.build(input_shape=(None, *input_shape))
    model.save(f'models/model_{ticker}.keras')
    model.summary()


if __name__ == "__main__":
    main()
