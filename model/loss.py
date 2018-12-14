# This file contains the custom loss used in Colorful Image Colorization
import tensorflow as tf
import keras.backend as K
from keras.losses import categorical_crossentropy
import numpy as np


def calculate_weights_maps(z_true, prior_probs, input_shape, len_q=313, _lambda=0.5):
    """
    Calculates the weight maps
    :param z_true: [batch, dim0, dim1, num_classes]
    :param p: smoothed empirical distribution [313,]
    :param len_q: 313 quantized levels
    :param _lambda: 0.5
    :return: weights maps [batch, dim0, dim1]
    """
    batch_size = input_shape[0]

    weights = 1 / ((1 - _lambda) * prior_probs + _lambda / len_q)
    q = tf.argmax(z_true, axis=3)  # [batch, dim0, dim1]
    # q = K.flatten(q)

    weights_maps = tf.gather(weights, q)  # [batch, dim0, dim1]

    weights_maps = tf.reshape(weights_maps, [10, 64, 64])

    return weights_maps


def categorical_crossentropy_weighted(prior_probs, input_shape):
    """
    Computes weighted categorical cross-entropy loss for a softmax distribution in a hot-encoded 2D array
    with shape (num_samples, num_classes, dim0, dim1)
    :param
    :return: categorical xentropy function
    """

    def _categorical_crossentropy(y_true, y_predicted):
        """

        :param y_true: keras.placeholder [batches, dim0,dim1, num_classes]
               Placeholder for data holding the ground-truth labels encoded in a one-hot representation
        :param y_predicted: keras.placeholder [batches,num_channels, dim0,dim1]
             Placeholder for data holding the softmax distribution over classes
             num_classes= Q quantized levels
        :return: scalar
             Categorical cross-entropy loss value
        """

        original_input_shape = [10, 64, 64, 313]
        weights_maps = calculate_weights_maps(z_true=y_true, prior_probs=prior_probs, input_shape=input_shape)
        weights_flatten = K.flatten(weights_maps)
        y_true_flatten = K.flatten(y_true)
        y_pred_flatten = K.flatten(y_predicted)

        y_pred_flatten = tf.Print(y_pred_flatten, [y_pred_flatten], message="This is y_pred_flatten: ")
        y_true_flatten = tf.Print(y_true_flatten, [y_true_flatten], message="This is y_true_flatten: ")

        y_pred_flatten_log = -K.log(y_pred_flatten + K.epsilon())
        cross_entropy = tf.multiply(y_true_flatten, y_pred_flatten_log)

        cross_entropy = tf.Print(cross_entropy, [cross_entropy], message="This is cross_entropy: ")

        cross_entropy = tf.reshape(cross_entropy, original_input_shape)
        cross_entropy = tf.reduce_sum(cross_entropy, axis=3)

        weights_flatten = tf.Print(weights_flatten, [weights_flatten], message="This is weights_flatten: ")

        weighted_cross_entropy = tf.multiply(K.flatten(cross_entropy), tf.cast(weights_flatten, tf.float32))
        weighted_cross_entropy = tf.reduce_sum(weighted_cross_entropy)

        return weighted_cross_entropy

    return _categorical_crossentropy
