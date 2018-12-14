# This file trains the network.
import time
import numpy as np
import data.data_generator as data
from keras.callbacks import TensorBoard
from model.cnn_net import graph, compile_model


class Trainer(object):
    def __init__(self):
        # Parameters
        self.LR = 0.0001
        self.N_EPOCHS = 50
        self.BATCH_SIZE = 25
        self.INPUT_SHAPE = [256, 256, 1]
        self.N_IMAGES_TRAIN_VAL = None
        self.TRAIN_SIZE = 0.8
        self.MODEL_NAME = 'flowers_weightedxentropy_lr_0001.h5'

        # Dataset file_paths and prior_probs
        self.prior_probs = np.load('/imatge/pvidal/2018-dlai-team9/data/prior_probs.npy')
        self.dataset_filepath = '/imatge/pvidal/dlai-flowers/train_flowers_realpaths.txt'

        # Class variables
        self.model = None
        self.tensorboard = None
        self.generator_val = None
        self.generator_train = None
        self.steps_per_val = 0
        self.steps_per_epoch = 0

    def train(self):
        self._print_params()
        self._define_logger()
        self._define_model()
        self._prepare_data()
        self._train()

    def _print_params(self):
        print('----------------------')
        print('----- PARAMETERS -----')
        print('----------------------')
        print('\tModel Name: {}'.format(self.MODEL_NAME))
        print('\tLearning Rate = {}'.format(self.LR))
        print('\tBatch Size = {}'.format(self.BATCH_SIZE))
        if self.N_IMAGES_TRAIN_VAL is not None:
            print('\tUsing only {} images.'.format(self.N_IMAGES_TRAIN_VAL))
        else:
            print('\tUsing all the images in the real_paths file.')
        print('\tFrom which {}% will be used as Training data and the remaining {}% as Validation data.'
              .format(self.TRAIN_SIZE*100., (1.-self.TRAIN_SIZE)*100.))
        print('\tNumber of Epochs = {}'.format(self.N_EPOCHS))
        print('\tInput shape: {}'.format(self.INPUT_SHAPE))

        print('----------------------')
        print('----------------------')
        print('----------------------')

    def _define_logger(self):
        self.tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()), update_freq='batch')

    def _prepare_data(self):
        print('Creating data generators...')
        listdir_train, listdir_val = data.split_train_val(dataset_filepath=self.dataset_filepath,
                                                          train_size=self.TRAIN_SIZE,
                                                          num_images=self.N_IMAGES_TRAIN_VAL)
        self.generator_train = data.data_generator(listdir=listdir_train, input_shape=self.INPUT_SHAPE,
                                                   batch_size=self.BATCH_SIZE)
        self.generator_val = data.data_generator(listdir=listdir_val, input_shape=self.INPUT_SHAPE,
                                                 batch_size=self.BATCH_SIZE)
        self.steps_per_epoch = len(listdir_train) / self.BATCH_SIZE
        self.steps_per_val = len(listdir_val) / self.BATCH_SIZE

    def _define_model(self):
        print("Defining architecture... ")
        self.model = graph(self.INPUT_SHAPE)
        self.model = compile_model(self.model, lr=self.LR, prior_probs=self.prior_probs)
        self.model.summary()

    def _train(self):
        print('Start training...')
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
        print("Training finished")


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
