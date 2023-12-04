import logging
import os


def setup_logger(name, log_folder, log_file, level=logging.INFO):
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    log_path = os.path.join(log_folder, log_file)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = logging.FileHandler(log_path)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
