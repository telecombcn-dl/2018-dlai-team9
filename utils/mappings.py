import os
import socket
import numpy as np
import utils.helpers as helpers
from skimage.transform import resize
from sklearn.neighbors import NearestNeighbors

# On import run:
if socket.gethostname() == 'rimmek-XPS-15':
    path = '/home/rimmek/MATT/DLAI/2018-dlai-team9/data/'
else:
    path = '/imatge/pvidal/2018-dlai-team9/data/'

pts_in_hull = np.load(os.path.join(path, 'pts_in_hull.npy'))
len_Q = pts_in_hull.shape[0]
nn = NearestNeighbors(algorithm='ball_tree').fit(pts_in_hull)


def inverse_h(y, neighbors=10, sigma=5.0):
    """
    Returns the inverse H mapping. That is, from Y to Q.
    :param y: Array of H*W*2 dimensions
    :param neighbors: Number of neighbors to use
    :type neighbors: int
    :param sigma: Sigma
    :type sigma: float
    :return: Array of H*W*Q dimensions
    """
    y_flat = helpers.flatten_nd_array(y)
    len_y = y_flat.shape[0]
    q_flat = np.zeros((len_y, len_Q))
    q_indices = np.arange(0, len_y, dtype='int')[:, np.newaxis]
    (d, i) = nn.kneighbors(y_flat, n_neighbors=neighbors)
    weights = np.exp(-d ** 2. / (2. * sigma ** 2.))
    weights = weights / np.sum(weights, axis=1)[:, np.newaxis]
    q_flat[q_indices, i] = weights
    q = helpers.unflatten_2d_array(q_flat, y)
    return q


def h(q, temp=1.0):
    """
    Returns the H mapping. That is, from Q to Y.
    :param q: Array of H*W*Q dimensions
    :param temp: Temperature
    :type temp: float
    :return: Array of H*W*2 dimensions
    """
    q = np.power(q, 1.0 / temp) / np.sum(np.power(q, 1.0 / temp), axis=2)[:, :, None]
    y = np.tensordot(q, pts_in_hull, axes=(2, 0))
    return y


def mapped_batch(image_batch):
    """
    Given a batch of images, it returns two lists: one containing the lightness and another one containing the
    color in the Q mapping.
    :param image_batch: Batch of images and labels
    :type image_batch: list
    :return: inputs, labels
    """
    inputs, labels = list(), list()
    for image in image_batch:
        inputs.append(np.expand_dims(image[:, :, 0], axis=2))
        labels.append(inverse_h(resize(image[:, :, 1:], (64, 64), mode='constant', anti_aliasing=True)))
    return np.array(inputs), np.array(labels)
