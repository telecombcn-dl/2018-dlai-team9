import os
import sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), *[os.path.pardir]))
from utils.mappings import inverse_h
from skimage.transform import resize


def mapped_batch(image_batch):
    """
    Given a batch of images, it returns two lists: one containing the lightness and another one containing the
    color in the Q mapping.
    :param image_batch: Batch of images and labels
    :type image_batch: list
    :return: inputs, labels
    """
    inputs, labels = list(), list()
    for i, image in enumerate(image_batch):
        inputs.append(np.expand_dims(image[:, :, 0], axis=2))
        labels.append(resize(inverse_h(image[:, :, 1:]), (256, 256)))
    return np.array(inputs), np.array(labels)
