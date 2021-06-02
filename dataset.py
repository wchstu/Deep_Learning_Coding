"""Common datasets"""

import gzip
import os
import pickle
import sys
import struct
import tarfile

import numpy as np

from tinynn.utils.downloader import download_url


def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]


def mnist(data_dir, one_hot=False):
    """
    return: train_set, valid_set, test_set
    train_set size: (50000, 784), (50000,)
    valid_set size: (10000, 784), (10000,)
    test_set size: (10000, 784), (10000,)
    feature: numerical in range [0, 1]
    target: categorical from 0 to 9
    """
    url = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
    checksum = "a02cd19f81d51c426d7ca14024243ce9"

    save_path = os.path.join(data_dir, url.split("/")[-1])
    print("Preparing MNIST dataset ...")
    try:
        download_url(url, save_path, checksum)
    except Exception as e:
        print("Error downloading dataset: %s" % str(e))
        sys.exit(1)

    # load the dataset
    with gzip.open(save_path, "rb") as f:
        train_set, valid_set, test_set = pickle.load(f, encoding="latin1")

    if one_hot:
        train_set = (train_set[0], get_one_hot(train_set[1], 10))
        valid_set = (valid_set[0], get_one_hot(valid_set[1], 10))
        test_set = (test_set[0], get_one_hot(test_set[1], 10))

    return train_set, valid_set, test_set