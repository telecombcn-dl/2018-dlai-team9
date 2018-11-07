# This file contains the code to pre-process the data (RGB to CIE Lab) plus inverse prior color probability mapping
# and also the calculation of the aforementioned prior probabilities.
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), *[os.path.pardir]))
from utils.mappings import inverse_h


def mapped_batch(image_batch):
    """
    Given a batch of images, it returns two lists: one containing the lightness and another one containing the
    color in the Q mapping.
    :param image_batch: Batch of images
    :type image_batch: list
    :return: inputs, labels
    """
    inputs, labels = list(), list()
    for image in image_batch:
        inputs.append(image[:, :, 0])
        labels.append(inverse_h(image[:, :, 1:]))
    return inputs, labels
