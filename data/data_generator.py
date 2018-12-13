# This file contains the function that feeds the data to the NN.
import numpy as np
from PIL import Image
from skimage.transform import resize
from skimage.color import rgb2lab, lab2rgb
from data.preprocess_data import mapped_batch


class Data(object):
    def __init__(self):
        pass

    @staticmethod
    def split_train_val(dataset_file, num_images_train, train_size):
        listdir = list()
        with open(dataset_file, 'r') as f:
            for i, path in enumerate(f):
                path = path.strip('\n')
                listdir.append(path)
        listdir_len = len(listdir)

        if num_images_train < listdir_len:
            l_train = int(np.round(train_size * num_images_train))
            l_val = num_images_train - l_train
        else:
            l_train = int(np.round(train_size * listdir_len))
            l_val = listdir_len - l_train

        np.random.seed(42)
        np.random.shuffle(listdir)
        return listdir[:l_train], listdir[l_train:l_train + l_val]

    def data_generator(self, listdir, image_input_shape, batch=100):
        w = image_input_shape[0]
        h = image_input_shape[1]
        return_list = list()

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
                        # while True:
                        yield (inputs, labels)
                    return_list = []
            print('Valid samples: {}'.format(valid_samples))
            print('Invalid shape samples: {}'.format(invalid_samples1))
            print('Invalid format samples: {}'.format(invalid_samples2))
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


# LOAD AND VISUALIZE DATASET
def main():
    d = Data()
    train_dataset_file = '../train.txt'  # change to the image paths file
    h, w = 256, 256
    batch = 100
    for batch in d.load_batch(train_dataset_file, h, w, batch):
        for img in batch:
            d.show_image(img, 'LAB')


if __name__ == "__main__":
    main()
