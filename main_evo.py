from typing import List, Type

import pandas as pd

from lib import model, data, determ, classification
from lib.datasets.info import DatasetInfo, DatasetMapping
from lib.datasets.retinopathyv2a import RetinopathyV2a

import tensorflow as tf

from lib.model import MLParams

DEFAULT_BATCH_SIZE = 32
SPLITS_NAMES = ["Train", "Validation", "Test"]
PROJECT_ROOT = '.'


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
        data_params: DataParams,
        shuffle_batches=True,
        reshuffle=False):
    img_shape = (data_params.image_size, data_params.image_size)

    return data.build_dataset(
        data_params.dataset,
        PROJECT_ROOT,
        img_shape,
        data_params.batch_size,
        data_params.splits,
        remap_classes=data_params.remap,
        shuffle_split=True,
        shuffle_batches=shuffle_batches,
        reshuffle=reshuffle
    )


def execute_experiment(splits):
    determ.set_global_determinism(42)

    data_params = DataParams(
        dataset=RetinopathyV2a,
        remap=RetinopathyV2a.mapping.c2,
        image_size=160,
        batch_size=32,
        splits=splits
    )

    model_params = model.ModelParams(
        base_model=model.BaseModel.RESNET50_v2,
        image_size=160,
        num_classes=2,
        dropout=0.2,
        global_pooling=model.GlobalPooling.AVG_POOLING,
        use_data_augmentation=True
    )

    training_params = model.TrainingParams(
        tl_learning_rate=0.0001,
        tl_epochs=10,
        fine_learning_rate=0.00001,
        fine_epochs=5,
        fine_layers=30
    )

    datasets, class_names, images_df = dataset_defaults(data_params)
    summary_df = data.summarize_batched_datasets(datasets, SPLITS_NAMES, class_names)
    train_dataset, validation_dataset, test_dataset = datasets

    retinopathy_resnet50 = model.RetinopathyModel(model_params)
    metrics, reports = model.transfer_and_fine_tune(
        retinopathy_resnet50, training_params,
        train_dataset, validation_dataset, test_dataset, class_names,
        verbose=1)
    total_time = metrics.pop('total_time')
    metrics_df = pd.DataFrame(metrics)

    return metrics_df, reports, summary_df


if __name__ == '__main__':
    experiment1 = execute_experiment(splits=[0.65, 0.25, 0.15])
    # experiment2 = execute_experiment(splits=[0.7, 0.2, 0.1])
