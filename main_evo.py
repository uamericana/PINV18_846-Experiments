from typing import List

from lib import model, data, determ, classification
from lib.datasets.retinopathyv2a import RetinopathyV2a

import tensorflow as tf

BATCH_SIZE = 32
SPLITS_NAMES = ["Train", "Validation", "Test"]
PROJECT_ROOT = '.'


def dataset_defaults(
        img_size: int,
        splits: List[float],
        remap_with=RetinopathyV2a.mapping.c2,
        shuffle_batches=True,
        reshuffle=False):
    img_size = (img_size, img_size)

    return data.build_dataset(
        RetinopathyV2a,
        PROJECT_ROOT,
        img_size,
        BATCH_SIZE,
        splits,
        remap_classes=remap_with,
        shuffle_split=True,
        shuffle_batches=shuffle_batches,
        reshuffle=reshuffle
    )


def classification_metrics(trained: tf.keras.Model, test_dataset: tf.data.Dataset, class_names: List[str]):
    report = classification.model_classification_report(trained, test_dataset, class_names)

    return report


def execute_experiment(splits):
    determ.set_global_determinism(42)

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
        tl_epochs=20,
        fine_learning_rate=0.00001,
        fine_epochs=10,
        fine_layers=30
    )

    datasets, class_names, images_df = dataset_defaults(img_size=model_params.image_size, splits=splits)
    summary_df = data.summarize_batched_datasets(datasets, SPLITS_NAMES, class_names)
    train_dataset, validation_dataset, test_dataset = datasets

    def metrics_callback(trained: tf.keras.models.Model):
        return classification_metrics(trained, test_dataset, class_names)

    retinopathy_resnet50 = model.RetinopathyModel(model_params)
    results = model.transfer_and_fine_tune(
        retinopathy_resnet50, training_params,
        train_dataset, validation_dataset, test_dataset,
        metrics_callback=metrics_callback,
        verbose=1)

    return results, summary_df


if __name__ == '__main__':
    experiment1 = execute_experiment(splits=[0.65, 0.25, 0.15])
    experiment2 = execute_experiment(splits=[0.7, 0.2, 0.1])
