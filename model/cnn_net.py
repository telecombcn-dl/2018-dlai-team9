# This file contains the CNN definition
from keras.optimizers import Adam
from model.loss import categorical_crossentropy, categorical_crossentropy_weighted
from keras.losses import mean_squared_error
from keras.layers import Input, Activation, ZeroPadding2D, BatchNormalization, Conv2D, Deconv2D, MaxPooling2D, Dropout,\
    Concatenate, Lambda, Add, UpSampling2D
from keras.regularizers import l1_l2
from keras.activations import softmax
from keras.metrics import categorical_accuracy


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


def graph_BN(input_shape):
    """
    This function defines the graph
    :param input_shape: shape of the input
    :return: a model instance in Keras
    """

    # Define the input placeholder as a tensor with shape input_shape.
    X_input = Input(input_shape)

    # ***** conv1 *****
    X = ZeroPadding2D((1, 1))(X_input)
    X = BatchNormalization(axis=3, name='batch_norm_1.1')(X)
    X = Conv2D(64, (3, 3), strides=(1, 1), name='bw_conv1_1')(X)
    X = Activation('relu')(X)  # relu1_1
    X = ZeroPadding2D((1, 1))(X)
    X = BatchNormalization(axis=3, name='batch_norm_1.2')(X)
    X = Conv2D(64, (3, 3), strides=(2, 2), name='conv1_2')(X)
    X = Activation('relu')(X)  # relu1_2
    X = BatchNormalization(axis=3, name='conv1_2norm')(X)  # axis=3??

    # ***** conv2 *****
    X = ZeroPadding2D((1, 1))(X)
    X = BatchNormalization(axis=3, name='batch_norm_2_1')(X)
    X = Conv2D(128, (3, 3), strides=(1, 1), name='conv2_1')(X)
    X = Activation('relu')(X)
    X = ZeroPadding2D((1, 1))(X)
    X = BatchNormalization(axis=3, name='batch_norm_2.2')(X)
    X = Conv2D(128, (3, 3), strides=(2, 2), name='conv2_2')(X)
    X = Activation('relu')(X)

    # ***** conv3 *****
    X = ZeroPadding2D((1, 1))(X)
    X = BatchNormalization(axis=3, name='batch_norm_3.1')(X)
    X = Conv2D(256, (3, 3), strides=(1, 1), name='conv3_1')(X)
    X = Activation('relu')(X)

    X = ZeroPadding2D((1, 1))(X)
    X = BatchNormalization(axis=3, name='batch_norm_3.2')(X)
    X = Conv2D(256, (3, 3), strides=(1, 1), name='conv3_2')(X)
    X = Activation('relu')(X)
    X = ZeroPadding2D((1, 1))(X)

    X = BatchNormalization(axis=3, name='batch_norm_3.3')(X)
    X = Conv2D(256, (3, 3), strides=(2, 2), name='conv3_3')(X)
    X = Activation('relu')(X)

    # ***** conv4 *****
    X = ZeroPadding2D((1, 1))(X)
    X = BatchNormalization(axis=3, name='batch_norm_4.1')(X)
    X = Conv2D(512, (3, 3), strides=(1, 1), dilation_rate=(1, 1), name='conv4_1')(X)
    X = Activation('relu')(X)
    X = ZeroPadding2D((1, 1))(X)
    X = BatchNormalization(axis=3, name='batch_norm_4.2')(X)
    X = Conv2D(512, (3, 3), strides=(1, 1), dilation_rate=(1, 1), name='conv4_2')(X)
    X = Activation('relu')(X)
    X = ZeroPadding2D((1, 1))(X)
    X = BatchNormalization(axis=3, name='batch_norm_4.3')(X)
    X = Conv2D(512, (3, 3), strides=(1, 1), dilation_rate=(1, 1), name='conv4_3')(X)
    X = Activation('relu')(X)

    # ***** conv5 *****
    X = ZeroPadding2D((2, 2))(X)
    X = BatchNormalization(axis=3, name='batch_norm_5.1')(X)
    X = Conv2D(512, (3, 3), strides=(1, 1), dilation_rate=(2, 2), name='conv5_1')(X)
    X = Activation('relu')(X)
    X = ZeroPadding2D((2, 2))(X)
    X = BatchNormalization(axis=3, name='batch_norm_5.2')(X)
    X = Conv2D(512, (3, 3), strides=(1, 1), dilation_rate=(2, 2), name='conv5_2')(X)
    X = Activation('relu')(X)
    X = ZeroPadding2D((2, 2))(X)
    X = BatchNormalization(axis=3, name='batch_norm_5.3')(X)
    X = Conv2D(512, (3, 3), strides=(1, 1), dilation_rate=(2, 2), name='conv5_3')(X)
    X = Activation('relu')(X)

    # ***** conv6 *****
    X = ZeroPadding2D((2, 2))(X)
    X = BatchNormalization(axis=3, name='batch_norm_6.1')(X)
    X = Conv2D(512, (3, 3), strides=(1, 1), dilation_rate=(2, 2), name='conv6_1')(X)
    X = Activation('relu')(X)
    X = ZeroPadding2D((2, 2))(X)
    X = BatchNormalization(axis=3, name='batch_norm_6.2')(X)
    X = Conv2D(512, (3, 3), strides=(1, 1), dilation_rate=(2, 2), name='conv6_2')(X)
    X = Activation('relu')(X)
    X = ZeroPadding2D((2, 2))(X)
    X = BatchNormalization(axis=3, name='batch_norm_6.3')(X)
    X = Conv2D(512, (3, 3), strides=(1, 1), dilation_rate=(2, 2), name='conv6_3')(X)
    X = Activation('relu')(X)

    # ***** conv7 *****
    X = ZeroPadding2D((1, 1))(X)
    X = BatchNormalization(axis=3, name='batch_norm_7.1')(X)
    X = Conv2D(512, (3, 3), strides=(1, 1), dilation_rate=(1, 1), name='conv7_1')(X)
    X = Activation('relu')(X)
    X = ZeroPadding2D((1, 1))(X)
    X = BatchNormalization(axis=3, name='batch_norm_7.2')(X)
    X = Conv2D(512, (3, 3), strides=(1, 1), dilation_rate=(1, 1), name='conv7_2')(X)
    X = Activation('relu')(X)
    X = ZeroPadding2D((1, 1))(X)
    X = BatchNormalization(axis=3, name='batch_norm_7.3')(X)
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
    X = Activation('softmax')(X)

    model = Model(inputs=X_input, outputs=X, name='graph')

    return model


def get_model(input_shape, **kwargs):
    return graph(input_shape)


def unet(input_shape=(256, 256, 1), l1=0.00001, l2=0.005):
    # Hyperaparametre values
    initializer = 'he_normal'
    pool_size = (2, 2)

    # Compute input shape, receptive field and output shape after softmax activation
    input_shape = input_shape
    output_shape = (256, 256, 313)

    # Activations, regularizers and optimizers
    regularizer = l1_l2(l1=l1, l2=l2)

    # Architecture definition
    # INPUT
    x = Input(shape=input_shape, name='V-net_input')

    # First block (down)
    first_conv = Conv2D(8, 3, kernel_initializer=initializer, kernel_regularizer=regularizer,
                        name='conv_initial', padding='same')(x)
    tmp = BatchNormalization(axis=3, name='batch_norm_1.1')(first_conv)
    tmp = Activation('relu')(tmp)
    z1 = Conv2D(8, 3, kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.1',
                padding='same')(tmp)

    c11 = Conv2D(8, 1, kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_1.1',
                 padding='same')(x)
    end_11 = Add()([z1, c11])

    # Second block (down)
    tmp = BatchNormalization(axis=3, name='batch_norm_2.1')(end_11)
    tmp = Activation('relu')(tmp)
    tmp = Conv2D(16, 2, strides=(2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                 name='downpool_1', padding='same')(tmp)
    tmp = BatchNormalization(axis=3, name='batch_norm_2.2')(tmp)
    tmp = Activation('relu')(tmp)
    tmp = Conv2D(16, 3, kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.1',
                 padding='same')(tmp)
    tmp = BatchNormalization(axis=3, name='batch_norm_2.3')(tmp)
    tmp = Activation('relu')(tmp)
    z2 = Conv2D(16, 3, kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.2',
                padding='same')(tmp)

    c21 = MaxPooling2D(pool_size=pool_size, name='pool_1')(end_11)
    c21 = Conv2D(16, 1, kernel_initializer=initializer, kernel_regularizer=regularizer,
                 name='conv_conn_2.1', padding='same')(c21)

    end_21 = Add()([z2, c21])

    # Third block (down)
    tmp = BatchNormalization(axis=3, name='batch_norm_3.1')(end_21)
    tmp = Activation('relu')(tmp)
    tmp = Conv2D(32, 2, strides=(2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                 name='downpool_2', padding='same')(tmp)
    tmp = BatchNormalization(axis=3, name='batch_norm_3.2')(tmp)
    tmp = Activation('relu')(tmp)
    tmp = Conv2D(32, (3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.1',
                 padding='same')(tmp)
    tmp = BatchNormalization(axis=3, name='batch_norm_3.3')(tmp)
    tmp = Activation('relu')(tmp)
    z3 = Conv2D(32, (3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.2',
                padding='same')(tmp)

    c31 = MaxPooling2D(pool_size=pool_size, name='pool_2')(end_21)
    c31 = Conv2D(32, (1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                 name='conv_conn_3.1', padding='same')(c31)

    end_31 = Add()([z3, c31])

    # Fourth block (down)
    tmp = BatchNormalization(axis=3, name='batch_norm_4.1')(end_31)
    tmp = Activation('relu')(tmp)
    tmp = Conv2D(64, (2, 2), strides=(2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                 name='downpool_3', padding='same')(tmp)
    tmp = BatchNormalization(axis=3, name='batch_norm_4.2')(tmp)
    tmp = Activation('relu')(tmp)
    tmp = Conv2D(64, (3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.1',
                 padding='same')(tmp)
    tmp = BatchNormalization(axis=3, name='batch_norm_4.3')(tmp)
    tmp = Activation('relu')(tmp)
    z4 = Conv2D(64, (3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.2',
                padding='same')(tmp)

    c41 = MaxPooling2D(pool_size=pool_size, name='pool_3')(end_31)
    c41 = Conv2D(64, (1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                 name='conv_conn_4.1', padding='same')(c41)

    end_41 = Add()([z4, c41])

    # Fifth block
    tmp = BatchNormalization(axis=3, name='batch_norm_5.1')(end_41)
    tmp = Activation('relu')(tmp)
    tmp = Conv2D(128, (2, 2), strides=(2, 2), kernel_initializer=initializer, kernel_regularizer=regularizer,
                 name='downpool_4', padding='same')(tmp)
    tmp = BatchNormalization(axis=4, name='batch_norm_5.2')(tmp)
    tmp = Activation('relu')(tmp)
    tmp = Conv2D(128, (3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_5.1',
                 padding='same')(tmp)
    tmp = BatchNormalization(axis=3, name='batch_norm_5.3')(tmp)
    tmp = Activation('relu')(tmp)
    tmp = Conv2D(128, (3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_5.2',
                 padding='same')(tmp)  # inflection point

    c5 = MaxPooling2D(pool_size=pool_size, name='pool_4')(end_41)
    c5 = Conv2D(128, (1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_5',
                padding='same')(c5)

    end_5 = Add()([tmp, c5])

    # Fourth block (up)
    tmp = BatchNormalization(axis=3, name='batch_norm_4.4')(end_5)
    tmp = Activation('relu')(tmp)
    tmp = UpSampling2D(size=pool_size, name='up_4')(tmp)
    tmp = Conv2D(64, (3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_4',
                 padding='same')(tmp)
    tmp = Concatenate(axis=3)([tmp, z4])
    tmp = BatchNormalization(axis=3, name='batch_norm_4.5')(tmp)
    tmp = Activation('relu')(tmp)
    tmp = Conv2D(64, (3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.3',
                 padding='same')(tmp)
    tmp = BatchNormalization(axis=3, name='batch_norm_4.6')(tmp)
    tmp = Activation('relu')(tmp)
    tmp = Conv2D(64, (3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_4.4',
                 padding='same')(tmp)

    c42 = UpSampling2D(size=pool_size, name='up_4conn')(end_5)
    c42 = Conv2D(64, (1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                 name='conv_conn_4.2', padding='same')(c42)

    end_42 = Add()([tmp, c42])

    # Third block (up)
    tmp = BatchNormalization(axis=3, name='batch_norm_3.4')(end_42)
    tmp = Activation('relu')(tmp)
    tmp = UpSampling2D(size=pool_size, name='up_3')(tmp)
    tmp = Conv2D(32, (3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_3',
                 padding='same')(tmp)
    tmp = Concatenate(axis=3)([tmp, z3])
    tmp = BatchNormalization(axis=3, name='batch_norm_3.5')(tmp)
    tmp = Activation('relu')(tmp)
    tmp = Conv2D(32, (3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.3',
                 padding='same')(tmp)
    tmp = BatchNormalization(axis=3, name='batch_norm_3.6')(tmp)
    tmp = Activation('relu')(tmp)
    tmp = Conv2D(32, (3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_3.4',
                 padding='same')(tmp)

    c32 = UpSampling2D(size=pool_size, name='up_3conn')(end_42)
    c32 = Conv2D(32, (1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                 name='conv_conn_3.2', padding='same')(c32)

    end_32 = Add()([tmp, c32])

    # Second block (up)
    tmp = BatchNormalization(axis=3, name='batch_norm_2.4')(end_32)
    tmp = Activation('relu')(tmp)
    tmp = UpSampling2D(size=pool_size, name='up_2')(tmp)
    tmp = Conv2D(16, (3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_2',
                 padding='same')(tmp)
    tmp = Concatenate(axis=3)([tmp, z2])
    tmp = BatchNormalization(axis=3, name='batch_norm_2.5')(tmp)
    tmp = Activation('relu')(tmp)
    tmp = Conv2D(16, (3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.3',
                 padding='same')(tmp)
    tmp = BatchNormalization(axis=3, name='batch_norm_2.6')(tmp)
    tmp = Activation('relu')(tmp)
    tmp = Conv2D(16, (3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_2.4',
                 padding='same')(tmp)

    c22 = UpSampling2D(size=pool_size, name='up_2conn')(end_32)
    c22 = Conv2D(16, (1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer,
                 name='conv_conn_2.2', padding='same')(c22)

    end_22 = Add()([tmp, c22])

    # First block (up)
    tmp = BatchNormalization(axis=3, name='batch_norm_1.4')(end_22)
    tmp = Activation('relu')(tmp)
    tmp = UpSampling2D(size=pool_size, name='up_1')(tmp)
    tmp = Conv2D(8, (3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_up_1',
                 padding='same')(tmp)
    tmp = Concatenate(axis=3)([tmp, z1])
    tmp = BatchNormalization(axis=3, name='batch_norm_1.5')(tmp)
    tmp = Activation('relu')(tmp)
    tmp = Conv2D(8, (3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.3',
                 padding='same')(tmp)
    tmp = BatchNormalization(axis=3, name='batch_norm_1.6')(tmp)
    tmp = Activation('relu')(tmp)
    tmp = Conv2D(8, (3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_1.4',
                 padding='same')(tmp)

    c12 = UpSampling2D(size=pool_size, name='up_1_conn')(end_22)
    c12 = Conv2D(8, (1, 1), kernel_initializer=initializer, kernel_regularizer=regularizer, name='conv_conn_1.2',
                 padding='same')(c12)

    end_12 = Add()([tmp, c12])

    # Final convolution
    tmp = BatchNormalization(axis=3, name='batch_norm_1.7')(end_12)
    tmp = Activation('relu')(tmp)
    tmp = Conv2D(8, (3, 3), kernel_initializer=initializer, kernel_regularizer=regularizer,
                 name='conv_pre_softmax', padding='same')(tmp)
    tmp = BatchNormalization(axis=3, name='batch_norm_pre_softmax')(tmp)

    in_softmax = Activation('relu')(tmp)

    classification = Conv2D(313, (1, 1, 1), kernel_initializer=initializer,
                            name='final_convolution_1x1x1')(in_softmax)

    y = softmax(classification)

    model = Model(inputs=x, outputs=y)

    return model


def get_unet(input_shape, **kwargs):
    return unet(input_shape)

def compile(model, lr=0.005, optimizer_name='Adam', loss_name='cross_entropy_weighted', prior_probs=None, input_shape= None):
    # Define Optimizer
    if optimizer_name == 'Adam':
        beta_1 = 0.9
        beta_2 = 0.999
        epsilon = 10 ** (-8)
        optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, decay=1e-3, epsilon=epsilon, clipnorm=1.)
    else:
        raise ValueError('Please, specify a valid optimizer')

    # Define Loss function
    if loss_name == 'cross_entropy':
        print('Loss: cross_entropy')
        loss = categorical_crossentropy
    elif loss_name == 'cross_entropy_weighted':
        print('Loss: Cross Entropy Weighted')
        loss = categorical_crossentropy_weighted(prior_probs, input_shape)
    elif loss_name == 'mse':
        loss = mean_squared_error
    else:
        raise ValueError('Please, specify a valid loss function!')

    # TODO: Define Metrics
    metrics = [categorical_accuracy]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


if __name__ == '__main__':
    model_graph = graph((256, 256, 1))
    print(model_graph.summary())
