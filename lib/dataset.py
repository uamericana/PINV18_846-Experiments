import os
import pathlib
import re
from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf

from lib.info import DatasetInfo, retinopathy_v2

RetinopathyV2 = retinopathy_v2()


def download_dataset(dataset_info: DatasetInfo, dest_dir: str):
    """
    Download dataset
    :param dataset_info: dataset info
    :param dest_dir: destination directory
    :return directory of the extracted zip
    """
    save_dir = os.path.join(dest_dir, dataset_info.name)
    os.makedirs(save_dir, exist_ok=True)
    url = dataset_info.url

    zip_path = tf.keras.utils.get_file(f"{dataset_info.name}.zip", url, extract=False)

    import zipfile

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(save_dir)

    return save_dir


def image_df(directory: str):
    """
    Read images and labels
    :param directory: where to find the images
    :return: a DataFrame with the image's path and label
    """
    images = []
    for file in pathlib.Path(directory).glob("**/*.jpg"):
        directory = os.path.basename(file.parent)
        label = re.sub(r"\d.*\.\s*", "", directory).strip()
        images.append({"Image": str(file), "dir": directory, "Status": label})
    return pd.DataFrame(images)


def reassign_labels(images_df: pd.DataFrame, classes: dict):
    """
    Remap labels from one class to another
    :param classes: target remap classes
    :param images_df: a dataframe with a Status column
    :return: the dataframe with a Label column with the remapped labels
    """
    x = images_df.Status.str.lower()
    target_values = [[lab.lower() for lab in labels] for labels in classes.values()]
    conditions = [x.isin(labels) for labels in target_values]
    choices = list(classes.keys())
    images_df["Label"] = np.select(conditions, choices)
    return images_df


def image_dataset(images_df: pd.DataFrame, img_size: tuple, remap_classes: dict = None):
    """
    Create an image dataset
    :param images_df: the image dataframe
    :param img_size: image size in pixels. shape (1, 1)
    :param remap_classes: dict to remap classes
    :return: tf.data.Dataset with the images and labels
    """
    if remap_classes is not None:
        images_df = reassign_labels(images_df, remap_classes)

    class_names = images_df.Label.unique()
    images_df = tf.data.Dataset.from_tensor_slices(
        (
            images_df.Image,
            images_df.Label,
        )
    )

    # class_names used in closures
    def get_label(label):
        one_hot = label == class_names
        return tf.argmax(one_hot)

    def parse_image(filename, label):
        image = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image)
        image = tf.image.resize(image, img_size, method="bilinear")

        return image, get_label(label)

    images_ds = images_df.map(parse_image)

    return images_ds, class_names


def split_dataset(ds: tf.data.Dataset, splits: List[float], shuffle=False):
    """
    Split dataset into subsets
    Splits can be of length 2 or 3 (e.g. (0.7, 0.3) or (0.6, 0.2, 0.1))
    The last split will always have the remaining observations

    :param ds: image dataset
    :param splits: array of splits
    :param shuffle: should shuffle before splitting
    :return: array of datasets corresponding to each split
    """
    img_count = ds.cardinality().numpy()
    if shuffle:
        ds = ds.shuffle(img_count, reshuffle_each_iteration=False)
    remainder = ds
    datasets = []

    for fraction in splits[:-1]:
        size = round(img_count * fraction, 0)
        subset = remainder.take(size)
        remainder = remainder.skip(size)
        datasets.append(subset)
    datasets.append(remainder)

    return datasets


def configure_for_performance(ds: tf.data.Dataset, batch_size: int, shuffle=False, reshuffle=False):
    """
    Configure dataset for performance (cache, batch, prefetch, shuffle)
    :param ds: dataset
    :param batch_size: batch size
    :param shuffle: should shuffle before batching
    :param reshuffle: reshuffle after each epoch
    :return: batched Dataset
    """
    ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(buffer_size=1000, reshuffle_each_iteration=reshuffle)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return ds


def configure_for_performance_list(datasets: List[tf.data.Dataset], batch_size: int, shuffle=False, reshuffle=False):
    """
    Configure for performance all datasets in the list
    :param datasets: list of image datasets
    :param batch_size: batch size
    :param shuffle: shuffle datasets
    :param reshuffle: reshuffle after epochs
    :return: list of shuffled datasets
    """
    return [configure_for_performance(ds, batch_size, shuffle, reshuffle) for ds in datasets]


def split_and_performance(ds: tf.data.Dataset, splits: List[float], batch_size: int, shuffle=False, reshuffle=False):
    """
    Performs split and configures for performance
    :param ds: images dataset
    :param splits: list of splits
    :param batch_size: number of observations per batch
    :param shuffle: should shuffle the datasets
    :param reshuffle: reshuffle after epoch
    :return: list of split and configured datasets
    """
    datasets = split_dataset(ds, splits, shuffle=shuffle)
    datasets = configure_for_performance_list(datasets, batch_size, shuffle, reshuffle)
    return datasets


def introspect_batched_dataset(dataset: tf.data.Dataset, name: str, class_names: List[str]):
    """
    Count the number of observations per class
    :param dataset: image dataset
    :param name: name of the dataset
    :param class_names: dataset labels
    :return: name of the dataset, total observations, counts per class
    """
    counts = {name: 0 for name in class_names}
    total = 0
    for images, labels in dataset.take(-1):
        for i in range(len(images)):
            idx = int(labels[i])
            counts[class_names[idx]] = 1
            total = 1

    return name, total, counts


def summarize_batched_datasets(datasets: tf.data.Dataset, names: List[str], class_names: List[str]):
    """
    Summarize the observations per class
    :param datasets: list of datasets
    :param names: names of the datasets
    :param class_names: class names
    :return: DataFrame with the summary
    """
    summary = []

    for [d, name] in zip(datasets, names):
        [name, total, counts] = introspect_batched_dataset(d, name, class_names)
        summary.append({"name": name, "total": total, **counts})

    summary_df = pd.DataFrame(summary)
    return summary_df


def build_dataset(
        dataset_info: DatasetInfo,
        project_dir: str,
        img_size: tuple,
        batch_size,
        splits,
        remap_classes=None,
        shuffle_split=True,
        shuffle_batches=False,
        reshuffle=False
):
    """
    Builds the retinopathy dataset

    :param dataset_info: dataset info
    :param project_dir: where to store the images
    :param img_size: image size. shape (1,1)
    :param batch_size: batch size
    :param splits: list of splits
    :param remap_classes: remap classes dict
    :param shuffle_split: shuffle before splitting
    :param shuffle_batches: shuffle each split
    :param reshuffle: reshuffle split after each epoch
    :return: split datasets, class names, images DataFrame (useful for debugging)
    """
    data_dir = os.path.join(project_dir, "data")
    download_dataset(dataset_info, data_dir)

    images_df = image_df(data_dir)

    (images_ds, class_names) = image_dataset(
        images_df, img_size=img_size, remap_classes=remap_classes
    )

    datasets = split_dataset(images_ds, splits=splits, shuffle=shuffle_split)
    datasets = configure_for_performance_list(
        datasets, batch_size=batch_size, shuffle=shuffle_batches, reshuffle=reshuffle
    )
    return datasets, class_names, images_df
