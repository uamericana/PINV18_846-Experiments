import tensorflow as tf
import pandas as pd
import numpy as np

from lib import classification, dataset

np.set_printoptions(precision=2)
pd.options.display.float_format = "{:,.2f}".format

PROJECT_ROOT = '.'

IMG_SIZE = (160, 160)
IMG_SHAPE = IMG_SIZE + (3,)
BATCH_SIZE = 32

SPLITS = [0.65, 0.25, 0.15]
SPLITS_NAMES = ["Train", "Validation", "Test"]


def dataset_defaults(splits=SPLITS,
                     remap_with=dataset.RetinopathyV2.mappings['c2'],
                     shuffle_batches=True,
                     reshuffle=False):
    tf.random.set_seed(42)
    datasets, class_names, images_df = dataset.build_dataset(
        dataset.RetinopathyV2,
        PROJECT_ROOT,
        IMG_SIZE,
        BATCH_SIZE,
        splits,
        remap_classes=remap_with,
        shuffle_split=True,
        shuffle_batches=shuffle_batches,
        reshuffle=reshuffle
    )

    return datasets, class_names, images_df


datasets, class_names, images_df = dataset_defaults()
