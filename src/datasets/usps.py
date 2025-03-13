__all__ = ['get_USPS']

import tensorflow as tf
import numpy as np
import gdown
import bz2
from pathlib import Path

from .utils import dataset_func_chain

"""
Function to load USPS dataset from torchvision.datasets as tf.data.Dataset and apply transformations
"""

USPS_DOWNLOAD_URL = {
    "train": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.bz2",
    "test": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.t.bz2",
}

def download_usps(folder_path: Path, image_set: str) -> None:
    """
    Summary: Download USPS from url

    :param folder_path: path of folder to save dataset
    :type folder_path: Path
    :param image_set: train or test
    :type image_set: str
    """
    folder_path.mkdir(parents=True, exist_ok=True)

    # URL to download the file
    url = USPS_DOWNLOAD_URL[image_set]

    # Directory to download and extract the file
    file_path = folder_path / Path(url).name
    # Download the file
    gdown.download(url, str(file_path), quiet=False)

def is_downloaded(folder_path: Path, image_set: str = "train") -> bool:
    """
    Summary : Check if dataset is downloaded

    :param folder_path: path of folder to save dataset
    :type folder_path: Path
    :param image_set: train or test dataset, defaults to "train"
    :type image_set: str, optional
    :return: True if already downloded
    :rtype: bool
    """
    file_path = folder_path / Path(USPS_DOWNLOAD_URL[image_set]).name
    exist = file_path.exists()
    return exist

def get_USPS(path,train=True,transform=[]):

    image_set = "train" if train else "test"

    downloaded = is_downloaded(path, image_set)
    if not downloaded:
        download_usps(path, image_set)

    file_path = path / Path(USPS_DOWNLOAD_URL[image_set]).name

    with bz2.open(file_path) as fp:
        raw_data = [line.decode().split() for line in fp.readlines()]
        tmp_list = [[x.split(":")[-1] for x in data[1:]] for data in raw_data]
        imgs = np.asarray(tmp_list, dtype=np.float32).reshape((-1, 16, 16))
        imgs = ((imgs + 1) / 2 * 255).astype(dtype=np.uint8)
        targets = [int(d[0]) - 1 for d in raw_data]

    images = np.array(imgs)
    images = np.reshape(images, (-1, 16, 16, 1))
    labels = np.array(targets)
    labels = np.reshape(labels, (-1, 1))

    # Reshape images to add a channel dimension
    images = np.reshape(images, (-1, 16, 16, 1))
    # Create a TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))

    # Map the function to the dataset
    dataset = dataset.map(lambda x, y: {"image": x, "label": y})

    if train:
        dataset = dataset.apply(tf.data.experimental.assert_cardinality(7291))
    else:
        dataset = dataset.apply(tf.data.experimental.assert_cardinality(2007))
    transform_func=dataset_func_chain(transform)
    return dataset.map(transform_func)
