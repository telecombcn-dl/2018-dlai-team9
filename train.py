# This file trains the network.
import time
import numpy as np
from model.cnn_net import graph, compile
from data.data_generator import Data
from keras.callbacks import TensorBoard


class Trainer(object):
    def __init__(self):
        # Parameters
        self.INPUT_SHAPE = [256, 256, 1]
        self.N_EPOCHS = 50
        self.BATCH_SIZE = 25
        self.LR = 0.0001
        self.N_IMAGES_TRAIN_VAL = 12677
        self.TRAIN_SIZE = 0.8
        self.MODEL_NAME = 'flowers_weightedxentropy_lr_0001.h5'

        # Dataset and prior_probs
        self.prior_probs = np.load('/imatge/pvidal/2018-dlai-team9/data/prior_probs.npy')
        # self.prior_probs = np.load('/home/adribarja/Documents/COLE/Q1/DLAI/2018-dlai-team9/data/prior_probs.npy')
        self.dataset_file = '/imatge/pvidal/dlai-flowers/train_flowers_realpaths.txt'

        # Class variables
        self.model = None
        self.tensorboard = None
        self.generator_val = None
        self.generator_train = None
        self.steps_per_val = 0
        self.steps_per_epoch = 0

    def train(self):
        self.define_logger()
        self.define_model()
        self.prepare_data()
        print('Start training...')
        self._train()
        print("Training finished")

    def define_logger(self):
        self.tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()), update_freq='batch')

    def prepare_data(self):
        print('Creating data generators...')
        data = Data()
        listdir_train, listdir_val = data.split_train_val(self.dataset_file,
                                                          num_images_train=self.N_IMAGES_TRAIN_VAL,
                                                          train_size=self.TRAIN_SIZE)
        self.generator_train = data.data_generator(listdir=listdir_train,
                                                   image_input_shape=self.INPUT_SHAPE,
                                                   batch=self.BATCH_SIZE)
        self.generator_val = data.data_generator(listdir=listdir_val,
                                                 image_input_shape=self.INPUT_SHAPE,
                                                 batch=self.BATCH_SIZE)
        self.steps_per_epoch = len(listdir_train) / self.BATCH_SIZE
        self.steps_per_val = len(listdir_val) / self.BATCH_SIZE

    def define_model(self):
        print("Defining architecture... ")
        print('Input shape: {}'.format(self.INPUT_SHAPE))
        self.model = graph(self.INPUT_SHAPE)
        self.model = compile(self.model, lr=self.LR, prior_probs=self.prior_probs)
        self.model.summary()

    def _train(self):
        try:
            self.model.fit_generator(generator=self.generator_train,
                                     steps_per_epoch=self.steps_per_epoch,
                                     epochs=self.N_EPOCHS,
                                     validation_data=self.generator_val,
                                     validation_steps=self.steps_per_val,
                                     callbacks=[self.tensorboard])

            self.model.save(self.MODEL_NAME)
        except KeyboardInterrupt:
            self.model.save(self.MODEL_NAME)


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
