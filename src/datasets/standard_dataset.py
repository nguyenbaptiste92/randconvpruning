__all__ = ['get_tfds_dataset']

import tensorflow as tf
import tensorflow_datasets as tfds

from .utils import dataset_func_chain

list_dataset=["mnist","mnist_corrupted","svhn_cropped"]

"""
Function to load datasets found in tensorflow_datasets as tf.data.Dataset: https://www.tensorflow.org/datasets/catalog/overview
and apply transformations
"""

def get_tfds_dataset(name,path,train=True,transform=[]):
    
    assert name in list_dataset
    split="train" if train else "test"
    transform_func=dataset_func_chain(transform)
    dataset=tfds.load(name = name,data_dir=path,split=split,download=True,as_supervised=False)
    return dataset.map(transform_func)