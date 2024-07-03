import sys
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from src.logging.logger import setup_logger

# Ensure the project directory is in the sys.path
project_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_dir))

# Setup logger
ROOT_DIR = project_dir
logger = setup_logger('model_builder_logger', 'logs', 'model_builder.log')


def build_model(input_shape, neurons=50, dropout=0.2, optimizer='adam',
                learning_rate=0.001, loss='mean_squared_error', metrics=None,
                l1_reg=0.0, l2_reg=0.0, additional_layers=0, bidirectional=False):
    """
    Build and compile an LSTM model with the given parameters.
    """
    logger.info("Building the model with parameters:")
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
    model.add(Bidirectional(lstm_layer) if bidirectional else lstm_layer)
    model.add(Dropout(dropout))
    model.add(BatchNormalization())

    # Additional LSTM layers
    for _ in range(additional_layers):
        lstm_layer = LSTM(neurons, return_sequences=True, kernel_regularizer=l1_l2(l1_reg, l2_reg))
        model.add(Bidirectional(lstm_layer) if bidirectional else lstm_layer)
        model.add(Dropout(dropout))
        model.add(BatchNormalization())

    # Final LSTM layer
    lstm_layer = LSTM(neurons)
    model.add(Bidirectional(lstm_layer) if bidirectional else lstm_layer)
    model.add(Dropout(dropout))
    model.add(BatchNormalization())

    # Output layer
    model.add(Dense(1))

   # Compile model
    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    else:
        opt = optimizer
    model.compile(optimizer=opt, loss=loss, metrics=metrics)

    return model
