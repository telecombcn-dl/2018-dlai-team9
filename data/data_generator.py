# This file contains the function that feeds the data to the NN.

from PIL import Image
import urllib.request as urllib2
import numpy as np
from skimage.transform import resize
from skimage.color import rgb2lab, lab2rgb
import os
from joblib import Parallel, delayed


# There are 14197122 images in imagenet

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
        with open(dataset_file_path) as f:
            Parallel(n_jobs=8)(delayed(self.job)(i, line, h, w, test_prop) for i, line in enumerate(f))

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


if __name__ == "__main__":
    main2()
