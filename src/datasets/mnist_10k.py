__all__ = ["get_MNIST10K"]

import tensorflow_datasets as tfds

from .utils import dataset_func_chain

"""
Function to load MNIST from tensorflow_datasets as tf.data.Dataset, apply transformations and filter 10000 elements
"""


def get_MNIST10K(path, train=True, transform=[]):
    split = "train" if train else "test"
    transform_func = dataset_func_chain(transform)
    dataset = tfds.load(
        name="mnist",
        data_dir=path,
        split=split,
        download=True,
        as_supervised=False,
    )
    dataset = dataset.map(transform_func)
    if train:
        dataset = dataset.take(10000)
    return dataset
