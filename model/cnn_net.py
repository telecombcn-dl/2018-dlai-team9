# This file contains the CNN definition
from keras.optimizers import Adam
from model.loss import categorical_crossentropy, categorical_crossentropy_weighted
from keras.layers import Input, Activation, ZeroPadding2D, BatchNormalization, Conv2D, Deconv2D
from keras.models import Model


def graph(input_shape):
    """
    This function defines the graph
    :param input_shape: shape of the input
    :return: a model instance in Keras
    """

    # Define the input placeholder as a tensor with shape input_shape.
    X_input = Input(input_shape)

    # ***** conv1 *****
    X = ZeroPadding2D((1, 1))(X_input)
    X = Conv2D(64, (3, 3), strides=(1, 1), name='bw_conv1_1')(X)
    X = Activation('relu')(X)  # relu1_1
    X = ZeroPadding2D((1, 1))(X)
    X = Conv2D(64, (3, 3), strides=(2, 2), name='conv1_2')(X)
    X = Activation('relu')(X)  # relu1_2
    X = BatchNormalization(axis=3, name='conv1_2norm')(X)  # axis=3??

    # ***** conv2 *****
    X = ZeroPadding2D((1, 1))(X)
    X = Conv2D(128, (3, 3), strides=(1, 1), name='conv2_1')(X)
    X = Activation('relu')(X)
    X = ZeroPadding2D((1, 1))(X)
    X = Conv2D(128, (3, 3), strides=(2, 2), name='conv2_2')(X)
    X = Activation('relu')(X)
    X = BatchNormalization(axis=3, name='conv2_2norm')(X)  # axis=3??

    # ***** conv3 *****
    X = ZeroPadding2D((1, 1))(X)
    X = Conv2D(256, (3, 3), strides=(1, 1), name='conv3_1')(X)
    X = Activation('relu')(X)
    X = ZeroPadding2D((1, 1))(X)
    X = Conv2D(256, (3, 3), strides=(1, 1), name='conv3_2')(X)
    X = Activation('relu')(X)
    X = ZeroPadding2D((1, 1))(X)
    X = Conv2D(256, (3, 3), strides=(2, 2), name='conv3_3')(X)
    X = Activation('relu')(X)
    X = BatchNormalization(axis=3, name='conv3_3norm')(X)  # axis=3??

    # ***** conv4 *****
    X = ZeroPadding2D((1, 1))(X)
    X = Conv2D(512, (3, 3), strides=(1, 1), dilation_rate=(1, 1), name='conv4_1')(X)
    X = Activation('relu')(X)
    X = ZeroPadding2D((1, 1))(X)
    X = Conv2D(512, (3, 3), strides=(1, 1), dilation_rate=(1, 1), name='conv4_2')(X)
    X = Activation('relu')(X)
    X = ZeroPadding2D((1, 1))(X)
    X = Conv2D(512, (3, 3), strides=(1, 1), dilation_rate=(1, 1), name='conv4_3')(X)
    X = Activation('relu')(X)
    X = BatchNormalization(axis=3, name='conv4_3norm')(X)  # axis=3??

    # ***** conv5 *****
    X = ZeroPadding2D((2, 2))(X)
    X = Conv2D(512, (3, 3), strides=(1, 1), dilation_rate=(2, 2), name='conv5_1')(X)
    X = Activation('relu')(X)
    X = ZeroPadding2D((2, 2))(X)
    X = Conv2D(512, (3, 3), strides=(1, 1), dilation_rate=(2, 2), name='conv5_2')(X)
    X = Activation('relu')(X)
    X = ZeroPadding2D((2, 2))(X)
    X = Conv2D(512, (3, 3), strides=(1, 1), dilation_rate=(2, 2), name='conv5_3')(X)
    X = Activation('relu')(X)
    X = BatchNormalization(axis=3, name='conv5_3norm')(X)  # axis=3??

    # ***** conv6 *****
    X = ZeroPadding2D((2, 2))(X)
    X = Conv2D(512, (3, 3), strides=(1, 1), dilation_rate=(2, 2), name='conv6_1')(X)
    X = Activation('relu')(X)
    X = ZeroPadding2D((2, 2))(X)
    X = Conv2D(512, (3, 3), strides=(1, 1), dilation_rate=(2, 2), name='conv6_2')(X)
    X = Activation('relu')(X)
    X = ZeroPadding2D((2, 2))(X)
    X = Conv2D(512, (3, 3), strides=(1, 1), dilation_rate=(2, 2), name='conv6_3')(X)
    X = Activation('relu')(X)
    X = BatchNormalization(axis=3, name='conv6_3norm')(X)  # axis=3??

    # ***** conv7 *****
    X = ZeroPadding2D((1, 1))(X)
    X = Conv2D(512, (3, 3), strides=(1, 1), dilation_rate=(1, 1), name='conv7_1')(X)
    X = Activation('relu')(X)
    X = ZeroPadding2D((1, 1))(X)
    X = Conv2D(512, (3, 3), strides=(1, 1), dilation_rate=(1, 1), name='conv7_2')(X)
    X = Activation('relu')(X)
    X = ZeroPadding2D((1, 1))(X)
    X = Conv2D(512, (3, 3), strides=(1, 1), dilation_rate=(1, 1), name='conv7_3')(X)
    X = Activation('relu')(X)
    X = BatchNormalization(axis=3, name='conv7_3norm')(X)  # axis=3??

    # ***** conv8 *****
    X = Deconv2D(256, (4, 4), strides=(2, 2), dilation_rate=(1, 1), name='conv8_1', padding='same')(X)
    X = Activation('relu')(X)
    X = ZeroPadding2D((1, 1))(X)
    X = Conv2D(256, (3, 3), strides=(1, 1), dilation_rate=(1, 1), name='conv8_2')(X)
    X = Activation('relu')(X)
    X = ZeroPadding2D((1, 1))(X)
    X = Conv2D(256, (3, 3), strides=(1, 1), dilation_rate=(1, 1), name='conv8_3')(X)
    X = Activation('relu')(X)

    # ***** Unary prediction *****
    X = Conv2D(313, (1, 1), name='conv8_313')(X)

    model = Model(inputs=X_input, outputs=X, name='graph')

    return model


def compile(model, lr=0.005, optimizer_name='Adam', loss_name='cross_entropy_weighted', weights=None):
    # Define Optimizer
    if optimizer_name == 'Adam':
        beta_1 = 0.9
        beta_2 = 0.999
        epsilon = 10 ** (-8)
        optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, clipnorm=1.)
    else:
        raise ValueError('Please, specify a valid optimizer')

    # Define Loss function
    if loss_name == 'cross_entropy':
        print('Loss: cross_entropy')
        loss = categorical_crossentropy
    elif loss_name == 'cross_entropy_weighted':
        print('Loss: Cross Entropy Weighted')
        loss = categorical_crossentropy_weighted(weights)
    else:
        raise ValueError('Please, specify a valid loss function')

    # TODO: Define Metrics
    # metrics = [raw_accuracy]
    # model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


if __name__ == '__main__':
    model_graph = graph((256, 256, 1))
    print(model_graph.summary())
