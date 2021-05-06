from typing import Type, List

import pandas as pd

from lib import data, model
from lib.datasets.info import DatasetInfo, DatasetMapping
from lib.model import MLParams

DEFAULT_BATCH_SIZE = 32


class DataParams(MLParams):
    def __init__(self,
                 dataset: Type[DatasetInfo],
                 splits: List[float],
                 image_size: int,
                 batch_size=DEFAULT_BATCH_SIZE,
                 remap: Type[DatasetMapping] = None):
        self.dataset = dataset
        self.splits = splits
        self.image_size = image_size
        self.batch_size = batch_size
        self.remap = remap


def dataset_defaults(
        project_root: str,
        data_params: DataParams,
        shuffle_batches=True,
        reshuffle=False):
    img_shape = (data_params.image_size, data_params.image_size)

    return data.build_dataset(
        data_params.dataset,
        project_root,
        img_shape,
        data_params.batch_size,
        data_params.splits,
        remap_classes=data_params.remap,
        shuffle_split=True,
        shuffle_batches=shuffle_batches,
        reshuffle=reshuffle
    )


def execute_experiment(datasets, class_names, model_params, training_params):
    train_dataset, validation_dataset, test_dataset = datasets
    retinopathy_model = model.RetinopathyModel(model_params)

    metrics, reports = model.transfer_and_fine_tune(
        retinopathy_model, training_params,
        train_dataset, validation_dataset, test_dataset, class_names,
        verbose=1)

    total_time = metrics.pop('total_time')
    metrics_df = pd.DataFrame(metrics)

    return metrics_df, reports, retinopathy_model
