# This file contains the custom loss used in Colorful Image Colorization
import tensorflow as tf
import keras.backend as K
import numpy as np


def categorical_crossentropy(z_true, z_predicted):
    """
    Computes categorical cross-entropy loss for a softmax distribution in a hot-encoded 2D array
    with shape (num_samples, num_classes, dim1, dim2)

    Parameters
    ----------
    z_true : Keras placeholder [batches, dim0,dim1, num_classes]
        Placeholder for data holding the ground-truth labels encoded in a one-hot representation
    z_predicted : Keras placeholder [batches,dim0,dim1, num_classes]
        Placeholder for data holding the softmax distribution over classes

    Returns
    -------
    scalar
        Categorical cross-entropy loss value
    """
    z_true_flatten = K.flatten(z_true)
    z_pred_flatten = K.flatten(z_predicted)
    z_pred_flatten_log = -K.log(z_pred_flatten + K.epsilon())
    num_total_elements = K.sum(z_true_flatten)
    cross_entropy = tf.reduce_sum(tf.multiply(z_true_flatten, z_pred_flatten_log))
    mean_cross_entropy = cross_entropy / (num_total_elements + K.epsilon())
    return mean_cross_entropy


def calculate_weights_maps(z_true, len_q=313, _lambda=0.5):
    """
    Calculates the weight maps
    :param z_true: [batch, dim0, dim1, num_classes]
    :param p: smoothed empirical distribution
    :param len_q: 313 quantized levels
    :param _lambda: 0.5
    :return: weights maps [batch, dim0, dim1]
    """
    input_shape = z_true.shape
    batch_size = input_shape[0]
    weights_maps = np.zeros(shape=[input_shape[0], input_shape[1], input_shape[2]])

    p = np.load('/Users/clarabonnin/Documents/MET/DLAI/2018-dlai-team9/data/prior_probs.npy')

    w = 1/((1 - _lambda) * p + _lambda / len_q)
    q = np.argmax(z_true, axis=3)  # [batch, dim0, dim1]

    for b in range(batch_size):
        weights_maps[b] = w[q[b]]

    return weights_maps


def categorical_crossentropy_weighted(weights_maps):
    """
    Computes weighted categorical cross-entropy loss for a softmax distribution in a hot-encoded 2D array
    with shape (num_samples, num_classes, dim0, dim1)
    :param weights: [batch, dim0, dim1]
    :return: categorical xentropy function
    """

    # define weights

    def categorical_crossentropy(y_true, y_predicted):
        """

        :param y_true: keras.placeholder [batches, dim0,dim1, num_classes]
               Placeholder for data holding the ground-truth labels encoded in a one-hot representation
        :param y_predicted: keras.placeholder [batches,dim0,dim1, num_classes]
             Placeholder for data holding the softmax distribution over classes
             num_classes= Q quantized levels
        :return: scalar
             Categorical cross-entropy loss value
        """

        weights_flatten = K.flatten(weights_maps)
        y_true_flatten = K.flatten(y_true)
        y_pred_flatten = K.flatten(y_predicted)
        y_pred_flatten_log = -K.log(y_pred_flatten + K.epsilon())
        cross_entropy = tf.multiply(y_true_flatten, y_pred_flatten_log)
        cross_entropy = tf.reduce_sum(cross_entropy, axis=3)
        multinomial_cross_entropy = tf.multiply(cross_entropy, weights_flatten)

        return multinomial_cross_entropy

    return categorical_crossentropy
