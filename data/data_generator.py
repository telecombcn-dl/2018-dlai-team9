# This file contains the function that feeds the data to the NN.
import numpy as np
from PIL import Image
from skimage.color import rgb2lab
from skimage.transform import resize
from utils.mappings import mapped_batch


def split_train_val(dataset_file, train_size, num_images_train=None):
    listdir = list()
    with open(dataset_file, 'r') as fr:
        for path in fr:
            path = path.strip('\n')
            listdir.append(path)
    listdir_len = len(listdir)
    if num_images_train is not None and num_images_train < listdir_len:
        l_train = train_size * num_images_train
        l_val = num_images_train - l_train
    else:
        l_train = train_size * listdir_len
        l_val = listdir_len - l_train

    np.random.seed(42)
    np.random.shuffle(listdir)
    return listdir[:l_train], listdir[l_train:l_train + l_val]


def data_generator(listdir, image_input_shape, batch=100):
    w = image_input_shape[0]
    h = image_input_shape[1]
    print('Dataset size = {0}'.format(len(listdir)))
    return_list = list()
    while True:
        valid_samples, invalid_samples1, invalid_samples2 = 0, 0, 0
        for path in listdir:
            try:
                image = load_image(h, w, path)
                if image.shape == (h, w, 3):
                    return_list.append(image)
                    valid_samples += 1
                else:
                    invalid_samples1 += 1
                    continue
            except:
                invalid_samples2 += 1
            if not (len(return_list) + 1) % (batch + 1):
                inputs, labels = mapped_batch(return_list)
                if inputs.shape == (batch, 256, 256, 1) and labels.shape == (batch, 64, 64, 313):
                    # while True:
                    yield (inputs, labels)
                return_list = list()
        print('Valid samples: {}'.format(valid_samples))
        print('Invalid shape samples: {}'.format(invalid_samples1))
        print('Invalid format samples: {}'.format(invalid_samples2))
        inputs, labels = mapped_batch(return_list)
        if inputs.shape == (batch, 256, 256, 1) and labels.shape == (batch, 64, 64, 313):
            yield (inputs, labels)


def load_image(h, w, path):
    img = Image.open(path)
    img_array = np.array(img)
    img_array_reshaped = resize(img_array, (h, w))
    img_array_reshaped_lab = rgb2lab(img_array_reshaped)
    return img_array_reshaped_lab

