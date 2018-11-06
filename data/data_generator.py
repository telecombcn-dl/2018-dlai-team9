# This file contains the function that feeds the data to the NN.

from PIL import Image
import urllib.request as urllib2
import numpy as np
from skimage.transform import resize
from skimage.color import rgb2lab, lab2rgb
import os


# There are 14197122 images in imagenet

class Data(object):
    def __init__(self):
        pass

    @staticmethod
    def generate_data_from_url_file(dataset_file_path, h, w, test_prop=0.1):
        # To generate a rondom sub set on size N from fall11_urls.txt execute on command line:
        # shuf -n N fall11_urls.txt > sub_dataset.txt
        with open(dataset_file_path) as f:
            for i, line in enumerate(f):
                try:
                    url = line.split('\t')[1].strip('\n')
                    name = line.split('\t')[0].strip('\n')
                    img = Image.open(urllib2.urlopen(url))
                    img_array = np.array(img)
                    img_array_reshaped = resize(img_array, (h, w))
                    img_array_reshaped_lab = rgb2lab(img_array_reshaped)

                    if i % int(100 * test_prop):
                        np.save('./processed/train/' + name + '.npy', img_array_reshaped_lab)
                    else:
                        np.save('./processed/test/' + name + '.npy', img_array_reshaped_lab)

                except:
                    continue

    def load_data_from_dataset_path(self, dataset_path='./preprocessed'):
        os.listdir(dataset_path + '/train/')
        np.load('')
        pass

    @staticmethod
    def show_image(img_array, encoding='RGB_norm'):
        if encoding == 'LAB':
            Image.fromarray(np.uint8(lab2rgb(img_array) * 255)).show()
        elif encoding == 'RGB_norm':
            Image.fromarray(np.uint8(img_array * 255)).show()
        elif encoding == 'RGB':
            Image.fromarray(img_array).show()


if __name__ == "__main__":
    d = Data()
    d.generate_data_from_url_file(dataset_file_path='./dataset.txt', h=256, w=256)
