import logging
import os
import random
import numpy as np
import tensorflow as tf

_SEED = 42


def set_seeds(seed=_SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


def set_global_determinism(seed=_SEED, full_determinism=False, cpu_only=False):
    """
        Enable 100% reproducibility on operations related to tensor and randomness.
        Parameters:
        seed (int): seed value for global randomness
        full_determinism (bool): whether to achieve efficiency at the cost of determinism/reproducibility
    """
    set_seeds(seed=seed)

    if cpu_only:
        tf.config.set_visible_devices([], 'GPU')

    if not full_determinism:
        return

    logging.warning("*******************************************************************************")
    logging.warning("*** set_global_determinism is called,setting full determinism, will be slow ***")
    logging.warning("*******************************************************************************")

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    # https://www.tensorflow.org/api_docs/python/tf/config/threading/set_inter_op_parallelism_threads
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    try:
        from tfdeterminism import patch
        patch()
    except Exception as e:
        logging.warning(e)
