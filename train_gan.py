# This file trains the network.
import time
import numpy as np
import data.data_generator as data
from keras.callbacks import TensorBoard
from model.cnn_net import graph, compile_model, discriminator, get_gan
from skimage.transform import resize


class Trainer(object):
    def __init__(self):
        # Parameters
        self.LR = 0.005
        self.N_ROUNDS = 20
        self.BATCH_SIZE = 10
        self.INPUT_SHAPE = [256, 256, 1]
        self.N_IMAGES_TRAIN_VAL = None
        self.TRAIN_SIZE = 0.8
        self.LOSS_NAME = 'cross_entropy_weighted'
        self.MODEL_NAME = 'flowers_weightedxentropy_lr_005_start_e8_e13_e5_e5_gan.h5'
        self.TRAIN_FROM_RESTORE_PATH = '/imatge/pvidal/2018-dlai-team9-joan/models/' \
                                       'flowers_weightedxentropy_lr_005_start_e8_e13_e5_weights.h5'

        # Dataset file_paths and prior_probs
        self.prior_probs = np.load('data/prior_probs.npy')
        self.dataset_filepath = '/imatge/pvidal/dlai-flowers/train_flowers_realpaths.txt'

        # Class variables
        self.model = None
        self.model_discriminator = None
        self.gan = None
        self.tensorboard = None
        self.generator_val = None
        self.generator_train = None
        self.steps_per_val = 0
        self.steps_per_epoch = 0

        # to show the progression of the losses
        self.val_gan_loss_array = np.zeros(self.N_ROUNDS)
        self.val_discr_loss_array = np.zeros(self.N_ROUNDS)
        self.gan_loss_array = np.zeros(self.N_ROUNDS)
        self.discr_loss_array = np.zeros(self.N_ROUNDS)

    def train_gan(self):
        self._define_logger()
        self._define_generator()
        self._define_discriminator()
        self._define_gan()
        self._make_trainable(self.model_discriminator, False)
        self._prepare_data()
        self._print_params()
        self._train()

    def _print_params(self):
        print('------------------------------------------------------------------')
        print('--------------------------- PARAMETERS ---------------------------')
        print('------------------------------------------------------------------')
        print('\tModel Name:        {}'.format(self.MODEL_NAME))
        print('\tLoss Name:         {}'.format(self.LOSS_NAME))
        print('\tLearning Rate =    {}'.format(self.LR))
        print('\tBatch Size =       {}'.format(self.BATCH_SIZE))
        print('\tNumber of Rounds = {}'.format(self.N_ROUNDS))
        if self.N_IMAGES_TRAIN_VAL is not None:
            print('\tUsing only {} images.'.format(self.N_IMAGES_TRAIN_VAL))
        else:
            print('\tUsing all the images in the real_paths file.')
        print('\tFrom which {:.2f}% is Training data\n'
              '\t       and {:.2f}% is Validation data.'
              .format(round(self.TRAIN_SIZE * 100., 2), round((1. - self.TRAIN_SIZE) * 100., 2)))
        print('\tInput shape: {}'.format(self.INPUT_SHAPE))
        print('------------------------------------------------------------------')
        print('------------------------------------------------------------------')
        print('------------------------------------------------------------------')

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
        # self.steps_per_epoch = int(len(listdir_train) / self.BATCH_SIZE)
        # self.steps_per_val = int(len(listdir_val) / self.BATCH_SIZE)
        self.steps_per_epoch = 3
        self.steps_per_val = 3

    def _define_model(self):
        print("Defining architecture... ")
        self.model = graph(self.INPUT_SHAPE)
        self.model = compile_model(self.model, lr=self.LR, loss_name=self.LOSS_NAME, prior_probs=self.prior_probs)
        self.model.summary()

    def _define_discriminator(self):
        print("Defining and compile discriminator architecture...")
        self.model_discriminator = discriminator()

        self.model_discriminator.summary()

    def _define_generator(self):
        print("Defining and compile generator architecture... ")
        self.model = graph(self.INPUT_SHAPE)
        self.model = compile_model(self.model, lr=self.LR, loss_name=self.LOSS_NAME, prior_probs=self.prior_probs)
        # Load the weights
        self.model.load_weights(self.TRAIN_FROM_RESTORE_PATH)
        self.model.summary()

    def _define_gan(self):
        self.gan = get_gan(self.model, self.model_discriminator, self.prior_probs)
        self.gan.summary()

    def _make_trainable(self, net, val):
        """
        If False, it fixes the network and it is not trainable (the weights are frozen)
        If True, the network is trainable (the weights can be updated)
        :param net: network
        :param val: boolean to make the network trainable or not
        """
        net.trainable = val
        for l in net.layers:
            l.trainable = val

    def _imgs2discr(self, real_images, real_labels, fake_labels):
        """
        It gets the input data to the discriminator
        :param real_images: input images
        :param real_labels: ground truth
        :param fake_labels: predicted labels
        :return: input images and labels to the discriminative network
        """
        upsampled_real_labels, upsampled_fake_labels = self._resize_labels(real_labels, fake_labels)

        real = np.concatenate((real_images, upsampled_real_labels), axis=3)
        fake = np.concatenate((real_images, upsampled_fake_labels), axis=3)

        img_batch = np.concatenate((real, fake), axis=0)
        lab_batch = np.ones((img_batch.shape[0], 1))
        lab_batch[real.shape[0]:, ...] = 0

        return img_batch, lab_batch
        # return [img_batch[:, :, :, 0], img_batch[:, :, :, 1:]], lab_batch

    def _resize_labels(self, real_labels, fake_labels):

        upsampled_real_labels = resize(real_labels, (self.BATCH_SIZE, 256, 256, 313), mode='constant',
                                       anti_aliasing=True)
        upsampled_fake_labels = resize(fake_labels, (self.BATCH_SIZE, 256, 256, 313), mode='constant',
                                       anti_aliasing=True)

        return upsampled_real_labels, upsampled_fake_labels

    def _imgs2gan(self, real_images, real_labels):
        """
        It gets the input data to the colorization network
        :param real_images: input images
        :param real_labels: ground truth
        :return: input images and labels to the colorization network
        """

        img_batch = [real_images, real_labels]
        lab_batch = np.ones((real_images.shape[0], 1))

        return img_batch, lab_batch

    def _train(self):
        print('Start adversarial training...')
        try:
            for n_round in range(self.N_ROUNDS):

                self._make_trainable(self.model_discriminator, True)
                for i in range(self.steps_per_epoch):
                    print("{}/{}".format(i,self.steps_per_epoch))
                    inputs_batch, labels_batch = next(self.generator_train)
                    pred_batch = self.model.predict(inputs_batch, batch_size=self.BATCH_SIZE)
                    inputs_discr_batch, labels_discr_batch = self._imgs2discr(inputs_batch, labels_batch, pred_batch)

                    loss, acc = self.model_discriminator.train_on_batch(inputs_discr_batch, labels_discr_batch)

                self.discr_loss_array[n_round] = loss
                print("DISCRIMINATOR Round: {0} -> Loss {1}".format((n_round + 1), loss))

                self._make_trainable(self.model_discriminator, False)
                for i in range(self.steps_per_epoch):
                    inputs_batch, labels_batch = next(self.generator_train)
                    images_gan_batch, labels_gan_batch = self._imgs2gan(inputs_batch, labels_batch)
                    loss, acc = self.gan.train_on_batch(images_gan_batch, labels_gan_batch)
                self.gan_loss_array[n_round] = loss
                print("GAN Round: {0} -> Loss {1}".format((n_round + 1), loss))

                print("Validation ...")
                for i in range(self.steps_per_epoch):
                    inputs_batch, labels_batch = next(self.generator_val)
                    pred_batch = self.model.predict(inputs_batch, batch_size=self.BATCH_SIZE)
                    inputs_discr_batch, labels_discr_batch = self._imgs2discr(inputs_batch, labels_batch, pred_batch)
                    loss_val, acc_val = self.model_discriminator.test_on_batch(inputs_discr_batch, labels_discr_batch)
                self.val_discr_loss_array[n_round] = loss_val
                print("DISCRIMINATOR VAL Round: {0} -> Loss {1}".format((n_round + 1), loss_val))

                for i in range(self.steps_per_epoch):
                    inputs_batch, labels_batch = next(self.generator_val)
                    images_gan_batch, labels_gan_batch = self._imgs2gan(inputs_batch, labels_batch)
                    loss_val, acc_val = self.gan.train_on_batch(images_gan_batch, labels_gan_batch)
                self.val_gan_loss_array[n_round] = loss_val
                print("GAN VAL Round: {0} -> Loss {1}".format((n_round + 1), loss_val))

                # save the weights of the colorization model
                self.model.save_weights(self.MODEL_NAME[:-3] + '_weights_{}.h5'.format(n_round))

            self.model.save(self.MODEL_NAME)
            self.model.save_weights(self.MODEL_NAME[:-3] + '_weights.h5')

            # print the evolution of the loss
            print("DISCR loss {}".format(self.discr_loss_array))
            print("GAN loss {}".format(self.gan_loss_array))
            print("DISCR validation loss {}".format(self.val_discr_loss_array))
            print("GAN validation loss {}".format(self.val_gan_loss_array))

        except KeyboardInterrupt:
            self.model.save(self.MODEL_NAME)
            self.model.save_weights(self.MODEL_NAME[:-3] + '_weights.h5')

        print("Training finished")


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train_gan()
