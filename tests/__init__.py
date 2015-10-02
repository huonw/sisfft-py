import numpy as np
import logging, os

np.random.seed(1)

try:
    name = os.environ['SISFFT_LOG'].upper()
    if name:
        logging.basicConfig(level = getattr(logging, name))
except KeyError:
    pass
    # logging.basicConfig(level = logging.INFO)
