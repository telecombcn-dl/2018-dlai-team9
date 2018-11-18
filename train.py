# This file trains the network.

import argparse
import sys

import params as p
import params.Imagenet as pd
import numpy as np

from model.cnn_net import get_model, compile
from data.data_generator import Data
from data.preprocess_data import mapped_batch

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
    input_shape =[params[p.BATCH_SIZE],params[p.INPUT_SHAPE][0],params[p.INPUT_SHAPE][1]]
    print('input shape', input_shape)
    model = get_model(input_shape)
    # compile model
    prior_probs = np.load('/imatge/pvidal/2018-dlai-team9/data/prior_probs.npy')
    model = compile(model, prior_probs = prior_probs, input_shape = input_shape)
    model.summary()

    """ DATA LOADING """
    print("Loading data ...")
    dataset_path = './data/dataset'
    d = Data(dataset_path)
    d.generate_data_from_url_file(dataset_file_path='./data/dataset.txt', h=256, w=256)

    listdir_train, listdir_val = d.split_train_val(num_images_train= params[p.N_IMAGES_TRAIN_VAL],train_size= params[p.TRAIN_SIZE])
    generator_train = d.data_generator(listdir=listdir_train,purpose='train', batch=params[p.BATCH_SIZE])
    generator_val = d.data_generator(listdir=listdir_val,purpose='val', batch=10)

    steps_per_epoch = len(listdir_train)
    steps_per_val = len(listdir_val)

    """ TENSORBOARD """
    # Define callbacks

    """ MODEL TRAINING """
    model.fit_generator(generator=generator_train,
                        steps_per_epoch=steps_per_epoch,
                        epochs=params[p.N_EPOCHS],
                        validation_data=generator_val,
                        validation_steps=steps_per_val)
    print("Training finished")
