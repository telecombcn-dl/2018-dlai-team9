# This file contains the function that feeds the data to the NN.

from PIL import Image
import urllib.request as urllib2
import numpy as np
from skimage.transform import resize
from skimage.color import rgb2lab, lab2rgb
import os
# from joblib import Parallel, delayed
from data.preprocess_data import mapped_batch


# When using not in UPC cluster
class Data(object):
    def __init__(self, dataset_path):
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
            os.makedirs(dataset_path + '/train/')
            os.makedirs(dataset_path + '/test/')
        self.dataset_path = dataset_path

    def job(self, i, line, h, w, test_prop):
        try:
            url = line.split('\t')[1].strip('\n')
            name = line.split('\t')[0].strip('\n')
            print(name)
            img = Image.open(urllib2.urlopen(url))
            img_array = np.array(img)
            img_array_reshaped = resize(img_array, (h, w))
            img_array_reshaped_lab = rgb2lab(img_array_reshaped)

            if i % int(100 * test_prop):
                np.save(self.dataset_path + '/train/' + name + '.npy', img_array_reshaped_lab)
            else:
                np.save(self.dataset_path + '/test/' + name + '.npy', img_array_reshaped_lab)
        except:
            pass

    def generate_data_from_url_file(self, dataset_file_path, h, w, test_prop=0.1):
        # To generate a rondom sub set on size N from fall11_urls.txt execute on command line:
        # shuf -n N fall11_urls.txt > sub_dataset.txt
        # with open(dataset_file_path) as f:
        # Parallel(n_jobs=8)(delayed(self.job)(i, line, h, w, test_prop) for i, line in enumerate(f))
        pass

    def load_batch(self, purpose='train', batch=100):
        purpose_path = self.dataset_path + '/' + purpose + '/'
        listdir = os.listdir(purpose_path)
        return_list = []
        for i, imdir in enumerate(listdir):
            return_list.append(np.load(purpose_path + imdir))
            if not (i + 1) % batch:
                yield np.array(return_list)
                return_list = []
        yield np.array(return_list)

    def split_train_val(self, num_images_train, train_size, purpose="train"):
        purpose_path = self.dataset_path + '/' + purpose + '/'
        listdir = os.listdir(purpose_path)
        L = len(listdir)
        if num_images_train < L:
            L_train = int(np.round(train_size * num_images_train))
            L_val = num_images_train - L_train
        else:
            L_train = int(np.round(train_size * L))
            L_val = L - L_train

        np.random.seed(42)
        np.random.shuffle(listdir)
        return listdir[:L_train], listdir[L_train:L_train + L_val]

    def data_generator(self, listdir, purpose, batch):
        purpose_path = self.dataset_path + '/' + purpose + '/'
        while True:
            for i, imdir in enumerate(listdir):
                return_list.append(np.load(purpose_path + imdir))
                if not (i + 1) % batch:
                    inputs, labels = mapped_batch(return_list)
                    inputs = np.array(inputs)
                    labels = np.array(labels)
                    yield (inputs, labels)
                    return_list = []

            inputs, labels = mapped_batch(return_list)
            inputs = np.array(inputs)
            labels = np.array(labels)
            yield (inputs, labels)

    @staticmethod
    def show_image(img_array, encoding='RGB_norm'):
        if encoding == 'LAB':
            Image.fromarray(np.uint8(lab2rgb(img_array) * 255)).show()
        elif encoding == 'RGB_norm':
            Image.fromarray(np.uint8(img_array * 255)).show()
        elif encoding == 'RGB':
            Image.fromarray(img_array).show()


# When using in UPC cluster
class Data2(object):
    def __init__(self):
        pass

    def load_batch(self, dataset_file, h, w, batch=100):
        return_list = []
        with open(dataset_file) as f:
            for i, path in enumerate(f):
                path = path.strip('\n')
                print(path)
                return_list.append(self.load_image(h, w, path))
                if not (i + 1) % batch:
                    yield np.array(return_list)
                    return_list = []
            yield np.array(return_list)

    @staticmethod
    def split_train_val(dataset_file, num_images_train, train_size):
        listdir = []
        with open(dataset_file, 'r') as f:
            for i, path in enumerate(f):
                path = path.strip('\n')
                listdir.append(path)
        L = len(listdir)

        if num_images_train < L:
            L_train = int(np.round(train_size * num_images_train))
            L_val = num_images_train - L_train
        else:
            L_train = int(np.round(train_size * L))
            L_val = L - L_train

        np.random.seed(42)
        np.random.shuffle(listdir)
        return listdir[:L_train], listdir[L_train:L_train + L_val]

    def data_generator(self, listdir, image_input_shape, batch=10):
        w = image_input_shape[0]
        h = image_input_shape[1]
        return_list = []

        print('Dataset size = {0}'.format(len(listdir)))
        while True:
            valid_samples = 0
            invalid_samples1 = 0
            invalid_samples2 = 0
            for i, path in enumerate(listdir):
                try:
                    image = self.load_image(h, w, path)
                    if image.shape == (h, w, 3):
                        return_list.append(image)
                        valid_samples += 1
                    else:
                        invalid_samples1 += 1
                        continue
                except:
                    invalid_samples2 += 1
                    pass
                if not (len(return_list) + 1) % (batch + 1):
                    inputs, labels = mapped_batch(return_list)
                    inputs = np.array(inputs)
                    labels = np.array(labels)
                    if inputs.shape == (batch, 256, 256, 1) and labels.shape == (batch, 64, 64, 313):
                        yield (inputs, labels)
                    return_list = []
            # print('Valid samples: {}'.format(valid_samples))
            # print('Invalid shape samples: {}'.format(invalid_samples1))
            # print('Invalid format samples: {}'.format(invalid_samples2))
            inputs, labels = mapped_batch(return_list)
            inputs = np.array(inputs)
            labels = np.array(labels)
            if inputs.shape == (batch, 256, 256, 1) and labels.shape == (batch, 64, 64, 313):
                yield (inputs, labels)


    @staticmethod
    def load_image(h, w, path):
        img = Image.open(path)
        img_array = np.array(img)
        img_array_reshaped = resize(img_array, (h, w))
        img_array_reshaped_lab = rgb2lab(img_array_reshaped)
        return img_array_reshaped_lab

    @staticmethod
    def show_image(img_array, encoding='RGB_norm'):
        if encoding == 'LAB':
            Image.fromarray(np.uint8(lab2rgb(img_array) * 255)).show()
        elif encoding == 'RGB_norm':
            Image.fromarray(np.uint8(img_array * 255)).show()
        elif encoding == 'RGB':
            Image.fromarray(img_array).show()


### HOW TO USE DATA CLASS

# CREATE DATASET
def main1():
    dataset_path = './dataset'
    d = Data(dataset_path)
    d.generate_data_from_url_file(dataset_file_path='./dataset.txt', h=256, w=256)


# LOAD AND VISUALIZE DATASET
def main2():
    dataset_path = './dataset'
    d = Data(dataset_path)
    for batch in d.load_batch(purpose='train', batch=20):
        for img in batch:
            d.show_image(img, 'LAB')


# LOAD AND VISUALIZE DATASET WITH DATA2
def main3():
    d = Data2()
    train_dataset_file = '../train.txt'  # change to the image paths file
    h = 256
    w = 256
    batch = 100
    for batch in d.load_batch(train_dataset_file, h, w, batch):
        for img in batch:
            d.show_image(img, 'LAB')


if __name__ == "__main__":
    main2()
