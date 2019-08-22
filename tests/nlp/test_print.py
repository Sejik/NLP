
from argparse import Namespace
import copy
import json

import jsbeautifier
import numpy as np
import random
import torch


def set_global_seed(seed=21):
    # Tensorflow
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # NumPy
    np.random.seed(seed)

    # Python
    random.seed(seed)
