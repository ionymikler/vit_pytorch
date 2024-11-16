import logging
import numpy as np
import os
import random
import torch

from typing import Union

def create_logger(log_file, logger_name:str, log_level:str="INFO"):
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    level = logging.getLevelNamesMapping()[log_level]
    assert level is not None, f"log_level name '{log_level}', not valid"

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # handlers
    fh  = logging.FileHandler(log_file, 'w')
    fh.setLevel(level)
    fh.setFormatter(log_format)
    logger.addHandler(fh)
    
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console)

    return logger

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