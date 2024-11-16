import logging
import numpy as np
import os
import random
import torch

from logging import DEBUG

def create_logger(log_file, logger_name:str, log_level:str="INFO"):
    log_format = '%(asctime)s.%(msecs)03d [%(levelname)s]: %(message)s'
    date_format = '%H:%M:%S'
    level = logging.getLevelNamesMapping()[log_level]
    assert level is not None, f"log_level name '{log_level}', not valid"

    logger = logging.getLogger(logger_name)
    logger.setLevel(DEBUG) # logger minimum level, each handler can have a different level

    # formatter
    log_format = logging.Formatter(log_format, datefmt=date_format)
    # handlers
    fh  = logging.FileHandler(log_file, 'w')
    fh.setLevel(DEBUG)
    fh.setFormatter(log_format)
    logger.addHandler(fh)
    
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(log_format)
    logger.addHandler(console)

    return logger

def copy_logger(logger:logging.Logger, new_name:str):
    new_logger = logging.getLogger(new_name)
    new_logger.handlers = logger.handlers
    new_logger.setLevel(logger.level)
    return new_logger

def get_unique_filename(filename):
    """
    Check if a file exists and append a numeric suffix to the filename 
    until an unused filename is found.
    """
    base, ext = os.path.splitext(filename)  # Separate base name and extension
    counter = 1

    new_filename = filename
    while os.path.exists(new_filename):
        new_filename = f"{base}_{counter}{ext}"
        counter += 1

    return new_filename