import datetime
from pathlib import Path
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional, Input
from keras.optimizers import Adam
from keras.regularizers import l1_l2
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import logger as logger_module

BASE_DIR = Path(__file__).resolve().parent.parent
logger = logger_module.setup_logger('model_logger', BASE_DIR / 'logs', 'model.log')

def build_model(input_shape, neurons=50, dropout=0.2, optimizer='adam',
                learning_rate=0.001, loss='mean_squared_error', metrics=None,
                l1_reg=0.0, l2_reg=0.0, additional_layers=0, bidirectional=False):
    """
    Build and compile an LSTM model with the given parameters.
    """
    logger.info(f"Building the model with parameters:")
    logger.info(f"  - input_shape: {input_shape}")
    logger.info(f"  - neurons: {neurons}")
    logger.info(f"  - dropout: {dropout}")
    logger.info(f"  - optimizer: {optimizer}")
    logger.info(f"  - learning_rate: {learning_rate}")
    logger.info(f"  - loss: {loss}")
    logger.info(f"  - metrics: {metrics}")
    logger.info(f"  - l1_reg: {l1_reg}")
    logger.info(f"  - l2_reg: {l2_reg}")
    logger.info(f"  - additional_layers: {additional_layers}")
    logger.info(f"  - bidirectional: {bidirectional}")

    if metrics is None:
        metrics = ['mae']

    model = Sequential()

    # Input layer
    model.add(Input(shape=input_shape))

    # LSTM layers
    lstm_layer = LSTM(neurons, return_sequences=True, kernel_regularizer=l1_l2(l1_reg, l2_reg))
    if bidirectional:
        model.add(Bidirectional(lstm_layer))
    else:
        model.add(lstm_layer)

    model.add(Dropout(dropout))
    model.add(BatchNormalization())

    # Additional LSTM layers
    for _ in range(additional_layers):
        lstm_layer = LSTM(neurons, return_sequences=True, kernel_regularizer=l1_l2(l1_reg, l2_reg))
        if bidirectional:
            model.add(Bidirectional(lstm_layer))
        else:
            model.add(lstm_layer)

        model.add(Dropout(dropout))
        model.add(BatchNormalization())

    # Final LSTM layer
    lstm_layer = LSTM(neurons)
    if bidirectional:
        model.add(Bidirectional(lstm_layer))
    else:
        model.add(lstm_layer)

    model.add(Dropout(dropout))
    model.add(BatchNormalization())

    # Output layer
    model.add(Dense(1))

    # Compile model
    opt = Adam(learning_rate=learning_rate) if optimizer == 'adam' else optimizer
    model.compile(optimizer=opt, loss=loss, metrics=metrics)

    return model

def prepare_callbacks(model_dir, ticker, monitor='val_loss', epoch=0):
    """
    Prepare callbacks for training the model.
    """
    logger.info(f"Preparing callbacks for {ticker}.")
    logger.info(f"  - model_dir: {model_dir}")
    logger.info(f"  - ticker: {ticker}")
    logger.info(f"  - monitor: {monitor}")
    logger.info(f"  - epoch: {epoch}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = BASE_DIR / f'logs/{ticker}/{timestamp}'
    model_dir = BASE_DIR / model_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    filepath = model_dir / f'model_{epoch:02d}-{monitor}.keras'
    callbacks = [
        EarlyStopping(monitor=monitor, patience=10, verbose=1, restore_best_weights=True),
        ModelCheckpoint(filepath=str(filepath), verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor=monitor, factor=0.1, patience=5, verbose=1),
        TensorBoard(log_dir=str(log_dir), histogram_freq=1)
    ]
    return callbacks
