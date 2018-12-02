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


def calculate_weights_maps(z_true, prior_probs, input_shape,  len_q=313, _lambda=0.5):
    """
    Calculates the weight maps
    :param z_true: [batch, dim0, dim1, num_classes]
    :param p: smoothed empirical distribution [313,]
    :param len_q: 313 quantized levels
    :param _lambda: 0.5
    :return: weights maps [batch, dim0, dim1]
    """
    batch_size = input_shape[0]

    weights = 1/((1 - _lambda) * prior_probs + _lambda / len_q)
    q = tf.argmax(z_true, axis=3)  # [batch, dim0, dim1]
    # q = K.flatten(q)

    weights_maps = tf.gather(weights, q)  # [batch, dim0, dim1]
    
    # weights_maps = tf.reshape(weights_maps, input_shape)

    return weights_maps


def categorical_crossentropy_weighted(prior_probs, input_shape):
    """
    Computes weighted categorical cross-entropy loss for a softmax distribution in a hot-encoded 2D array
    with shape (num_samples, num_classes, dim0, dim1)
    :param
    :return: categorical xentropy function
    """

    def categorical_crossentropy(y_true, y_predicted):
        """

        :param y_true: keras.placeholder [batches, dim0,dim1, num_classes]
               Placeholder for data holding the ground-truth labels encoded in a one-hot representation
        :param y_predicted: keras.placeholder [batches,num_channels, dim0,dim1]
             Placeholder for data holding the softmax distribution over classes
             num_classes= Q quantized levels
        :return: scalar
             Categorical cross-entropy loss value
        """
        original_input_shape = [input_shape[0], input_shape[1], input_shape[2], 313]
        weights_maps = calculate_weights_maps(z_true=y_true, prior_probs=prior_probs, input_shape=input_shape)
        weights_flatten = K.flatten(weights_maps)
        y_true_flatten = K.flatten(y_true)
        y_pred_flatten = K.flatten(y_predicted)
        y_pred_flatten_log = -K.log(y_pred_flatten + K.epsilon())
        cross_entropy = tf.multiply(y_true_flatten, y_pred_flatten_log)
        cross_entropy = tf.reshape(cross_entropy, original_input_shape)
        cross_entropy = tf.reduce_sum(cross_entropy, axis=3)
        weighted_cross_entropy = tf.multiply(cross_entropy, tf.cast(weights_flatten, tf.float32))
        multinomial_cross_entropy = tf.reduce_sum(weighted_cross_entropy)

        return multinomial_cross_entropy

    return categorical_crossentropy
