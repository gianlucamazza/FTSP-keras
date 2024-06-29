import sys
import datetime
from pathlib import Path
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

# Ensure the project directory is in the sys.path
project_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_dir))

from src.logging.logger import setup_logger

ROOT_DIR = project_dir
logger = setup_logger('callback_logger', 'logs', 'callback_logger.log')


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
    log_dir = ROOT_DIR / f'logs/{ticker}/{timestamp}'
    model_dir = ROOT_DIR / model_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    filepath = model_dir / f'model_{epoch:02d}-{monitor}.keras'
    callbacks = [
        EarlyStopping(monitor=monitor, patience=10, verbose=1, restore_best_weights=True),
        ModelCheckpoint(filepath=str(filepath), verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor=monitor, factor=0.1, patience=5, verbose=1),
        TensorBoard(log_dir=str(log_dir), histogram_freq=1)
    ]
    return callbacks
