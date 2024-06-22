import logging
from pathlib import Path


def setup_logger(name, relative_log_folder, log_file, level=logging.INFO):
    base_dir = Path(__file__).parent.parent
    log_folder = base_dir / relative_log_folder
    log_folder.mkdir(parents=True, exist_ok=True)

    log_path = log_folder / log_file

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Check if logger already has handlers
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
