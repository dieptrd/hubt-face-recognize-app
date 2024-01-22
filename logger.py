from datetime import datetime
import logging
import os

def _setup_logger():
    # Create a custom logger
    logger = logging.getLogger(__name__)

    # Set the level of logger
    logger.setLevel(logging.DEBUG)
    

    # Check if the subfolder './logs' exists, if not, create it
    logs_folder = './logs'
    if not os.path.exists(logs_folder):
        os.makedirs(logs_folder)

    # Create handler
    file_handler = logging.FileHandler('./logs/logging_{}.log'.format(datetime.now().strftime("%Y%m%d-%H%M%S")))

    # Set level of handler
    file_handler.setLevel(logging.DEBUG)

    # Create formatter and add it to handler
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)

    # Add handler to the logger
    logger.addHandler(file_handler)

    return logger

logger = _setup_logger()