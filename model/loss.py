# This file contains the custom loss used in Colorful Image Colorization
import tensorflow as tf
import keras.backend as K


def calculate_weights_maps(z_true, prior_probs):
    """
    Calculates the weight maps
    :param z_true: [dim0, dim1, num_classes]
    :param prior_probs: probability of each color in ImageNet
    :return: weights maps [dim0, dim1]
    """
    weights = 1. / (0.5 * prior_probs + 0.5 / 313.) / 101.3784919
    q = tf.argmax(z_true, axis=3)
    weights_maps = tf.gather(weights, q)  # [batch, dim0, dim1]
    return weights_maps


def categorical_crossentropy_weighted(prior_probs):
    """
    Returns the weighted categorical cross-entropy loss function
    :param prior_probs: probability of each color in ImageNet
    :return: categorical xentropy function
    """

    def categorical_crossentropy(y_true, y_pred):
        """
        Computes weighted categorical cross-entropy loss for a softmax distribution in a hot-encoded 2D array
        with shape (num_samples, num_classes, dim0, dim1)
        :param y_true: keras.placeholder [batches, dim0,dim1, num_classes]
               Placeholder for data holding the ground-truth labels encoded in a one-hot representation
        :param y_pred: keras.placeholder [batches,num_channels, dim0,dim1]
             Placeholder for data holding the softmax distribution over classes
             num_classes= Q quantized levels
        :return: Categorical cross-entropy loss value
        """
        weights_maps = calculate_weights_maps(z_true=y_true, prior_probs=prior_probs)
        y_pred_log = -K.log(y_pred + K.epsilon())
        cross_entropy = tf.multiply(y_true, y_pred_log)
        cross_entropy = tf.reduce_sum(cross_entropy, axis=3)
        weighted_cross_entropy = tf.multiply(cross_entropy, tf.cast(weights_maps, tf.float32))
        weighted_cross_entropy = tf.reduce_sum(weighted_cross_entropy)
        return weighted_cross_entropy / 1282048.0  # (64 * 64 * 313)

    return categorical_crossentropy
