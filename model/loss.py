# # This file contains the custom loss used in Colorful Image Colorization
# import tensorflow as tf
# import keras.backend as K
# import numpy as np
#
#
# def categorical_crossentropy(z_true, z_predicted):
#     """
#     Computes categorical cross-entropy loss for a softmax distribution in a hot-encoded 2D array
#     with shape (num_samples, num_classes, dim1, dim2)
#
#     Parameters
#     ----------
#     z_true : Keras placeholder [batches, dim0,dim1, num_classes]
#         Placeholder for data holding the ground-truth labels encoded in a one-hot representation
#     z_predicted : Keras placeholder [batches,dim0,dim1, num_classes]
#         Placeholder for data holding the softmax distribution over classes
#
#     Returns
#     -------
#     scalar
#         Categorical cross-entropy loss value
#     """
#     z_true_flatten = K.flatten(z_true)
#     z_pred_flatten = K.flatten(z_predicted)
#     z_pred_flatten_log = -K.log(z_pred_flatten + K.epsilon())
#     num_total_elements = K.sum(z_true_flatten)
#     cross_entropy = tf.reduce_sum(tf.multiply(z_true_flatten, z_pred_flatten_log))
#     mean_cross_entropy = cross_entropy / (num_total_elements + K.epsilon())
#     return mean_cross_entropy
#
#
# def calculate_weights_maps(z_true, prior_probs, input_shape,  len_q=313, _lambda=0.5):
#     """
#     Calculates the weight maps
#     :param z_true: [batch, dim0, dim1, num_classes]
#     :param p: smoothed empirical distribution [313,]
#     :param len_q: 313 quantized levels
#     :param _lambda: 0.5
#     :return: weights maps [batch, dim0, dim1]
#     """
#
#     weights = 1/((1 - _lambda) * prior_probs + _lambda / len_q)
#     q = tf.argmax(z_true, axis=3)  # [batch, dim0, dim1]
#     # q = K.flatten(q)
#
#     weights_maps = tf.gather(weights, q)  # [batch, dim0, dim1]
#
#     # weights_maps = tf.reshape(weights_maps, input_shape)
#
#     return weights_maps
#
#
# def categorical_crossentropy_weighted(prior_probs, input_shape):
#     """
#     Computes weighted categorical cross-entropy loss for a softmax distribution in a hot-encoded 2D array
#     with shape (num_samples, num_classes, dim0, dim1)
#     :param
#     :return: categorical xentropy function
#     """
#
#     def categorical_crossentropy(y_true, y_predicted):
#         """
#
#         :param y_true: keras.placeholder [batches, dim0,dim1, num_classes]
#                Placeholder for data holding the ground-truth labels encoded in a one-hot representation
#         :param y_predicted: keras.placeholder [batches,num_channels, dim0,dim1]
#              Placeholder for data holding the softmax distribution over classes
#              num_classes= Q quantized levels
#         :return: scalar
#              Categorical cross-entropy loss value
#         """
#         original_input_shape = [64,64,313]
#         weights_maps = calculate_weights_maps(z_true=y_true, prior_probs=prior_probs, input_shape=input_shape)
#         weights_flatten = K.flatten(weights_maps)
#         y_true_flatten = K.flatten(y_true)
#         y_pred_flatten = K.flatten(y_predicted)
#         y_pred_flatten_log = -K.log(y_pred_flatten + K.epsilon())
#         cross_entropy = tf.multiply(y_true_flatten, y_pred_flatten_log)
#         cross_entropy = tf.reshape(cross_entropy, original_input_shape)
#         cross_entropy = tf.reduce_sum(cross_entropy, axis=3)
#         weighted_cross_entropy = tf.multiply(cross_entropy, tf.cast(weights_flatten, tf.float32))
#         multinomial_cross_entropy = tf.reduce_sum(weighted_cross_entropy)
#
#         return multinomial_cross_entropy
#
#     return categorical_crossentropy
#
# # --------------------------------------------------
# This file contains the custom loss used in Colorful Image Colorization
import tensorflow as tf
import keras.backend as K
import numpy as np


#
# def categorical_crossentropy(z_true, z_predicted):
#     """
#     Computes categorical cross-entropy loss for a softmax distribution in a hot-encoded 2D array
#     with shape (num_samples, num_classes, dim1, dim2)
#
#     Parameters
#     ----------
#     z_true : Keras placeholder [batches, dim0,dim1, num_classes]
#         Placeholder for data holding the ground-truth labels encoded in a one-hot representation
#     z_predicted : Keras placeholder [batches,dim0,dim1, num_classes]
#         Placeholder for data holding the softmax distribution over classes
#
#     Returns
#     -------
#     scalar
#         Categorical cross-entropy loss value
#     """
#     z_true_flatten = K.flatten(z_true)
#     z_pred_flatten = K.flatten(z_predicted)
#     z_pred_flatten_log = -K.log(z_pred_flatten + K.epsilon())
#     num_total_elements = K.sum(z_true_flatten)
#     cross_entropy = tf.reduce_sum(tf.multiply(z_true_flatten, z_pred_flatten_log))
#     mean_cross_entropy = cross_entropy / (num_total_elements + K.epsilon())
#     return mean_cross_entropy
#
#
# def calculate_weights_maps(z_true, prior_probs, input_shape, len_q=313, _lambda=0.5):
#     """
#     Calculates the weight maps
#     :param z_true: [batch, dim0, dim1, num_classes]
#     :param p: smoothed empirical distribution [313,]
#     :param len_q: 313 quantized levels
#     :param _lambda: 0.5
#     :return: weights maps [batch, dim0, dim1]
#     """
#     batch_size = input_shape[0]
#
#     weights = 1 / ((1 - _lambda) * prior_probs + _lambda / len_q)
#     q = tf.argmax(z_true, axis=3)  # [batch, dim0, dim1]
#     # q = K.flatten(q)
#
#     weights_maps = tf.gather(weights, q)  # [batch, dim0, dim1]
#
#     weights_maps = tf.reshape(weights_maps, [10, 64, 64])
#
#     return weights_maps
#
#
# def categorical_crossentropy_weighted(prior_probs, input_shape):
#     """
#     Computes weighted categorical cross-entropy loss for a softmax distribution in a hot-encoded 2D array
#     with shape (num_samples, num_classes, dim0, dim1)
#     :param
#     :return: categorical xentropy function
#     """
#     print('entered crossentropy weighted')
#
#     def categorical_crossentropy(y_true, y_predicted):
#         """
#
#         :param y_true: keras.placeholder [batches, dim0,dim1, num_classes]
#                Placeholder for data holding the ground-truth labels encoded in a one-hot representation
#         :param y_predicted: keras.placeholder [batches,num_channels, dim0,dim1]
#              Placeholder for data holding the softmax distribution over classes
#              num_classes= Q quantized levels
#         :return: scalar
#              Categorical cross-entropy loss value
#         """
#         original_input_shape = [10, 64, 64, 313]
#         weights_maps = calculate_weights_maps(z_true=y_true, prior_probs=prior_probs, input_shape=input_shape)
#         print(weights_maps.shape)
#         weights_flatten = K.flatten(weights_maps)
#         print(weights_flatten.shape)
#         y_true_flatten = K.flatten(y_true)
#         print(y_true_flatten.shape)
#         y_pred_flatten = K.flatten(y_predicted)
#         print(y_pred_flatten.shape)
#         y_pred_flatten_log = -K.log(y_pred_flatten + K.epsilon())
#         print(y_pred_flatten_log.shape)
#         cross_entropy = tf.multiply(y_true_flatten, y_pred_flatten_log)
#         print(cross_entropy.shape)
#         cross_entropy = tf.reshape(cross_entropy, original_input_shape)
#         print(cross_entropy.shape)
#         cross_entropy = tf.reduce_sum(cross_entropy, axis=3)
#         print(cross_entropy.shape)
#         cross_entropy = K.flatten(cross_entropy)
#         print(cross_entropy.shape)
#         weighted_cross_entropy = tf.multiply(tf.cast(cross_entropy, tf.float32), tf.cast(weights_flatten, tf.float32))
#         print(weighted_cross_entropy.shape)
#         # weighted_cross_entropy = tf.multiply(cross_entropy, tf.cast(weights_flatten, tf.float32))
#         # weighted_cross_entropy = cross_entropy
#         multinomial_cross_entropy = tf.reduce_sum(weighted_cross_entropy)
#         print(multinomial_cross_entropy.shape)
#
#         return multinomial_cross_entropy
#
#     return categorical_crossentropy
#
# # --------------------------------------------------


def calculate_weights_maps(z_true, prior_probs, len_q=313., _lambda=0.5):
    """
    Calculates the weight maps
    :param z_true: [dim0, dim1, num_classes]
    :param p: smoothed empirical distribution [313,]
    :param len_q: 313 quantized levels
    :param _lambda: 0.5
    :return: weights maps [dim0, dim1]
    """
    weights = 1 / ((1 - _lambda) * prior_probs + _lambda / len_q)
    q = tf.argmax(z_true, axis=3)
    weights_maps = tf.gather(weights, q)  # [batch, dim0, dim1]
    return weights_maps


def softmax_per_pixel(y):
    softmax = tf.divide(tf.exp(y), tf.expand_dims(tf.reduce_sum(tf.exp(y), axis=3), 3))
    return softmax


def categorical_crossentropy_weighted(prior_probs, input_shape):
    """
    Computes weighted categorical cross-entropy loss for a softmax distribution in a hot-encoded 2D array
    with shape (num_samples, num_classes, dim0, dim1)
    :param
    :return: categorical xentropy function
    """
    print('entered crossentropy weighted')

    def categorical_crossentropy(y_true, y_pred):
        """

        :param y_true: keras.placeholder [batches, dim0,dim1, num_classes]
               Placeholder for data holding the ground-truth labels encoded in a one-hot representation
        :param y_predicted: keras.placeholder [batches,num_channels, dim0,dim1]
             Placeholder for data holding the softmax distribution over classes
             num_classes= Q quantized levels
        :return: scalar
             Categorical cross-entropy loss value
        """
        y_pred = tf.Print(y_pred, [tf.reduce_min(y_pred)], 'y_pred min: ')
        y_pred = tf.Print(y_pred, [tf.reduce_max(y_pred)], 'y_pred max: ')
        y_true = tf.Print(y_true, [tf.reduce_min(y_true)], 'y_trues min: ')
        y_true = tf.Print(y_true, [tf.reduce_max(y_true)], 'y_trues max: ')
        weights_maps = calculate_weights_maps(z_true=y_true, prior_probs=prior_probs)
        y_pred_log = -K.log(y_pred + K.epsilon())
        cross_entropy = tf.multiply(y_true, y_pred_log)
        cross_entropy = tf.reduce_sum(cross_entropy, axis=3)
        weighted_cross_entropy = tf.multiply(cross_entropy, tf.cast(weights_maps, tf.float32))
        weighted_cross_entropy = tf.reduce_sum(weighted_cross_entropy)
        weighted_cross_entropy = tf.Print(weighted_cross_entropy, [weighted_cross_entropy], 'weighted cross_entropy: ')
        return weighted_cross_entropy/tf.reduce_sum(y_true + K.epsilon())

    return categorical_crossentropy


# if __name__ == '__main__':
#     y_true = np.random.rand(10, 64, 64, 313)
#     y_pred = np.random.rand(10, 64, 64, 313)
#     prior_probs = np.load('/home/adribarja/Documents/COLE/Q1/DLAI/2018-dlai-team9/data/prior_probs.npy')
#     # print(prior_probs)
#     # print(prior_probs.shape)
#     # print(softmax_per_pixel(y_pred))
#     sa_loss = categorical_crossentropy_weighted(prior_probs, [64, 64, 313])
#     print(sa_loss(y_true, y_pred))
#     print('ola')

    # y = tf.convert_to_tensor(np.ones((10, 64, 64, 313)))
    # sess = tf.InteractiveSession()
    # softmax = tf.reduce_sum(tf.divide(tf.exp(y), tf.expand_dims(tf.reduce_sum(tf.exp(y), axis=3), 3)), axis=3)
    # print (sess.run(softmax))
