from enum import Enum
from typing import Type, List

import pandas as pd

from lib import data, model
from lib.datasets.retinopathyv2 import RetinopathyV2
from lib.datasets.retinopathyv2a import RetinopathyV2a
from lib.datasets.retinopathyv2b import RetinopathyV2b
from lib.datasets.retinopathyv3 import RetinopathyV3
from lib.model import MLParams

DEFAULT_BATCH_SIZE = 32

available_datasets = {
    RetinopathyV2a.name: RetinopathyV2a,
    RetinopathyV2b.name: RetinopathyV2b,
    RetinopathyV2.name: RetinopathyV2,
    RetinopathyV3.name: RetinopathyV3
}


def stop_criteria(fitness_history, patience=5):
    last_history = fitness_history[-patience:]
    if len(last_history) < patience:
        return False

    diffs = sum([abs(a - b) for a, b in zip(last_history, last_history[1:])])

    return diffs == 0


class DataParams(MLParams):
    def __init__(self,
                 dataset: str,
                 splits: List[float],
                 image_size: int,
                 batch_size=DEFAULT_BATCH_SIZE,
                 remap: str = None):
        self.dataset = dataset
        self.splits = splits
        self.image_size = image_size
        self.batch_size = batch_size
        self.remap = remap

    def get_dataset_and_mapping(self):
        try:
            dataset = available_datasets[self.dataset]
        except KeyError:
            raise Exception(f"Invalid dataset {self.dataset}. Must be one of {list(available_datasets.keys())}")

        mapping = None
        if self.remap is not None:
            try:
                mapping = dataset.mapping[self.remap]
            except KeyError:
                raise Exception(f"Ivalid mapping {self.remap}. Must be one of {list(dataset.mapping)}")

        return dataset, mapping

    def as_name(self) -> str:
        return f"ds{self.dataset}-" \
               f"sp{'_'.join([str(s) for s in self.splits])}-" \
               f"is{self.image_size}-" \
               f"bs{self.batch_size}-" \
               f"rm{self.remap}"


def dataset_defaults(
        project_root: str,
        data_params: DataParams,
        shuffle_batches=True,
        reshuffle=False):
    img_shape = (data_params.image_size, data_params.image_size)

    dataset, mapping = data_params.get_dataset_and_mapping()
    remap_classes = None if mapping is None else mapping.value

    return data.build_dataset(
        dataset,
        project_root,
        img_shape,
        data_params.batch_size,
        data_params.splits,
        remap_classes=remap_classes,
        shuffle_split=True,
        shuffle_batches=shuffle_batches,
        reshuffle=reshuffle
    )


def execute_experiment(datasets, class_names, model_params, training_params, verbose=2):
    train_dataset, validation_dataset, test_dataset = datasets
    retinopathy_model = model.RetinopathyModel(model_params)

    metrics, reports = model.transfer_and_fine_tune(
        retinopathy_model, training_params,
        train_dataset, validation_dataset, test_dataset, class_names,
        verbose=verbose)

    total_time = metrics.pop('total_time')

    return metrics, reports, retinopathy_model
