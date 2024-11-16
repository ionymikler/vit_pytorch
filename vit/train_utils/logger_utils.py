import logging
import numpy as np
import os
import random
import torch

def create_logger(log_file, rank=0):
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    logging.basicConfig(level=logging.INFO if rank == 0 else 'ERROR',
                        format=log_format,
                        filename=log_file)
    
    console = logging.StreamHandler()
    console.setLevel(logging.INFO if rank == 0 else 'ERROR')
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)


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