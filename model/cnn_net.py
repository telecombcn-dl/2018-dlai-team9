# This file contains the CNN definition
from keras.optimizers import Adam
from model.loss import categorical_crossentropy, categorical_crossentropy_weighted



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

    # Define Metrics
    metrics = [raw_accuracy]


    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics)

    return model