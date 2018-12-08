# This file trains the network.

import argparse
import sys
import time
from os.path import join
import params as p
import params.Imagenet as pd
import numpy as np

from model.cnn_net import get_model, compile, get_unet
from data.data_generator import Data2
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

if __name__ == "__main__":

    """ PARAMETERS"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", help="Params", default=None, type=str)
    arg = parser.parse_args(sys.argv[1:])

    print('Getting parameters to train the model...')
    params_string = arg.p
    params = pd.PARAMS_DICT[params_string].get_params()
    output_path = params[p.OUTPUT_PATH]

    print('PARAMETERS')
    print(params)

    """ ARCHITECTURE DEFINITION """
    print(" Defining architecture ... ")
    # get model
    input_shape = [params[p.INPUT_SHAPE][0], params[p.INPUT_SHAPE][1], params[p.INPUT_SHAPE][2]]
    print('input shape', input_shape)
    model = get_unet(input_shape)
    print('compile model')
    # compile model
    prior_probs = np.load('/imatge/pvidal/2018-dlai-team9/data/prior_probs.npy')
    model = compile(model, prior_probs=prior_probs, input_shape=input_shape, loss_name= params[p.LOSS])
    model.summary()

    """ DATA LOADING """
    print("Loading data ...")
    dataset_file = './images_realpaths.txt'
    d = Data2()

    print('Creating generators ...')
    listdir_train, listdir_val = d.split_train_val(dataset_file, num_images_train=params[p.N_IMAGES_TRAIN_VAL],
                                                   train_size=params[p.TRAIN_SIZE])
    generator_train = d.data_generator(listdir=listdir_train, image_input_shape=params[p.INPUT_SHAPE],
                                       batch=params[p.BATCH_SIZE])
    generator_val = d.data_generator(listdir=listdir_val, image_input_shape=params[p.INPUT_SHAPE],
                                     batch=params[p.BATCH_SIZE])

    steps_per_epoch = len(listdir_train) / params[p.BATCH_SIZE]
    steps_per_val = len(listdir_val) / params[p.BATCH_SIZE]


    """ TENSORBOARD """
    # Define callbacks

    tensorboard = TensorBoard(log_dir="/imatge/mcaros/dlai/clara/logs/{}".format(time.time()), update_freq='batch')

    output_path = params[p.OUTPUT_PATH]
    #checkpoint = ModelCheckpoint(join(output_path,'checkpoints') , monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

    earlystopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

    """ MODEL TRAINING """

    print('Start training...')
    try:

        history = model.fit_generator(generator=generator_train,
                                      steps_per_epoch=steps_per_epoch,
                                      epochs=params[p.N_EPOCHS],
                                      validation_data=generator_val,
                                      validation_steps=steps_per_val,
                                      verbose=1,
                                      callbacks=[tensorboard, earlystopping])

        model.save('/imatge/mcaros/dlai/clara/model.h5')
        np.save('history', history.history)

    except KeyboardInterrupt:
        model.save('/imatge/mcaros/dlai/clara/model.h5')

    print("Training finished")
