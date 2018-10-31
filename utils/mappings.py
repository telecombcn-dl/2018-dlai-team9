import numpy as np
import utils.helpers as helpers
from sklearn.neighbors import NearestNeighbors

# On import run:
prior_probs = np.load('../data/prior_probs.npy')
pts_in_hull = np.load('../data/pts_in_hull.npy')
len_Q = pts_in_hull.shape[0]
nn = NearestNeighbors(algorithm='ball_tree').fit(pts_in_hull)


def inverse_h(y, neighbors=5, sigma=5):
    """
    Returns the inverse H mapping. That is from Y to Q.
    :param y: Image Y of dimensions H*W*2
    :param neighbors:
    :param sigma:
    :return: Image Q of dimensions H*W*Q
    """
    y_flat = helpers.flatten_nd_array(y)
    len_y = y_flat.shape[0]
    q_flat = np.zeros((len_y, len_Q))
    q_indices = np.arange(0, len_y, dtype='int')[:, np.newaxis]
    (d, i) = nn.kneighbors(y_flat, n_neighbors=neighbors)
    weights = np.exp(-d ** 2 / (2 * sigma ** 2))
    weights = weights / np.sum(weights, axis=1)[:, np.newaxis]
    q_flat[q_indices, i] = weights
    q = helpers.unflatten_2d_array(q_flat, y)
    return q

