__all__ = ['get_USPS']

import tensorflow as tf
import numpy as np
import datasetops as do
from torchvision.datasets import USPS

from .utils import dataset_func_chain

"""
Function to load USPS dataset from torchvision.datasets as tf.data.Dataset and apply transformations
"""

def get_USPS(path,train=True,transform=[]):

    transform_func=dataset_func_chain(transform)
    dataset = do.from_pytorch(USPS(path, download=True, train=train))
    dataset = dataset.transform(
                lambda x: {
                    "image": np.expand_dims(np.array(x[0]), axis=2),
                    "label": x[1],
                }
            ).to_tensorflow()
    if train:
        dataset = dataset.apply(tf.data.experimental.assert_cardinality(7291))
    else:
        dataset = dataset.apply(tf.data.experimental.assert_cardinality(2007))
    return dataset.map(transform_func)