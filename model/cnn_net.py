# This file contains the CNN definition
from keras.models import Model
from keras.optimizers import Adam
from keras.activations import softmax
from keras.losses import categorical_crossentropy
from model.loss import categorical_crossentropy_weighted
from keras.metrics import mse, categorical_crossentropy as cc
from keras.layers import Input, Activation, ZeroPadding2D, BatchNormalization, Conv2D, Deconv2D


def graph(input_shape):
    """
    This function defines the graph
    :param input_shape: shape of the input
    :return: a model instance in Keras
    """
    x_input = Input(input_shape)

    # ***** conv1 *****
    x = ZeroPadding2D((1, 1))(x_input)
    x = Conv2D(64, (3, 3), strides=(1, 1), name='bw_conv1_1')(x)
    x = Activation('relu')(x)  # relu1_1
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(64, (3, 3), strides=(2, 2), name='conv1_2')(x)
    x = Activation('relu')(x)  # relu1_2
    x = BatchNormalization(axis=3, name='conv1_2norm')(x)

    # ***** conv2 *****
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(128, (3, 3), strides=(1, 1), name='conv2_1')(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(128, (3, 3), strides=(2, 2), name='conv2_2')(x)
    x = Activation('relu')(x)
    x = BatchNormalization(axis=3, name='conv2_2norm')(x)

    # ***** conv3 *****
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), name='conv3_1')(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), name='conv3_2')(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(256, (3, 3), strides=(2, 2), name='conv3_3')(x)
    x = Activation('relu')(x)
    x = BatchNormalization(axis=3, name='conv3_3norm')(x)

    # ***** conv4 *****
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), dilation_rate=(1, 1), name='conv4_1')(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), dilation_rate=(1, 1), name='conv4_2')(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), dilation_rate=(1, 1), name='conv4_3')(x)
    x = Activation('relu')(x)
    x = BatchNormalization(axis=3, name='conv4_3norm')(x)

    # ***** conv5 *****
    x = ZeroPadding2D((2, 2))(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), dilation_rate=(2, 2), name='conv5_1')(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D((2, 2))(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), dilation_rate=(2, 2), name='conv5_2')(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D((2, 2))(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), dilation_rate=(2, 2), name='conv5_3')(x)
    x = Activation('relu')(x)
    x = BatchNormalization(axis=3, name='conv5_3norm')(x)

    # ***** conv6 *****
    x = ZeroPadding2D((2, 2))(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), dilation_rate=(2, 2), name='conv6_1')(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D((2, 2))(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), dilation_rate=(2, 2), name='conv6_2')(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D((2, 2))(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), dilation_rate=(2, 2), name='conv6_3')(x)
    x = Activation('relu')(x)
    x = BatchNormalization(axis=3, name='conv6_3norm')(x)

    # ***** conv7 *****
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), dilation_rate=(1, 1), name='conv7_1')(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), dilation_rate=(1, 1), name='conv7_2')(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), dilation_rate=(1, 1), name='conv7_3')(x)
    x = Activation('relu')(x)
    x = BatchNormalization(axis=3, name='conv7_3norm')(x)

    # ***** conv8 *****
    x = Deconv2D(256, (4, 4), strides=(2, 2), dilation_rate=(1, 1), name='conv8_1', padding='same')(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), dilation_rate=(1, 1), name='conv8_2')(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), dilation_rate=(1, 1), name='conv8_3')(x)
    x = Activation('relu')(x)

    # ***** Unary prediction *****
    x = Conv2D(313, (1, 1), name='conv8_313')(x)
    x = Activation(activation=softmax)(x)
    model = Model(inputs=x_input, outputs=x, name='graph')
    return model


def compile_model(model, lr=0.005, optimizer_name='Adam', loss_name='cross_entropy_weighted', prior_probs=None):
    # Define Optimizer
    if optimizer_name == 'Adam':
        optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=10 ** (-8), clipnorm=1.)
    else:
        raise ValueError('Please, specify a valid optimizer')

    # Define Loss function
    if loss_name == 'cross_entropy':
        loss = categorical_crossentropy
    elif loss_name == 'cross_entropy_weighted':
        loss = categorical_crossentropy_weighted(prior_probs)
    else:
        raise ValueError('Please, specify a valid loss function')

    metrics = [mse, cc]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    print('Model compiled')
    return model


if __name__ == '__main__':
    model_graph = graph((256, 256, 1))
    print(model_graph.summary())
