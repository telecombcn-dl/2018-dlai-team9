# This file evaluates the model.
import os
import png
import argparse
import numpy as np
from PIL import Image
from skimage.transform import resize
from skimage.color import rgb2lab, lab2rgb
from keras.models import load_model
from utils import mappings


class Evaluation(object):
    def __init__(self, autotest=False, temperature=0.38):
        self.temperature = temperature
        self.edge_size = 256
        self.autotest = autotest
        self.ori_image = None
        self.ori_luminance = None
        args = self.parse_arguments()
        self.input_image_path = args.input_image
        self.model_path = args.model_path
        self.model = None
        self.output_path = 'outputs/'
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)

    @staticmethod
    def parse_arguments():
        parser = argparse.ArgumentParser(description='Evaluation inputs')
        parser.add_argument('-i', '--input_image', type=str, required=True,
                            help='Black and white image to be colorized')
        parser.add_argument('-m', '--model_path', type=str, help='Trained model')

        return parser.parse_args()

    def eval(self):
        lab_resized_image = self.load_image()
        luminance, chromaticity = self.split_image(lab_resized_image)

        if not self.autotest:
            self.load_model()
            predicted_chromaticity = self.predict_chromaticity(luminance)
        else:
            predicted_chromaticity = mappings.h(mappings.inverse_h(chromaticity), temp=self.temperature)

        pred_image = self.merge_and_resize_image(predicted_chromaticity)
        self.show_image(pred_image, title='Predicted image', encoding='LAB')
        self.show_image(self.ori_image, title='Original image', encoding='RGB')
        self.save_image(pred_image, name='predicted_image')
        self.save_image(self.ori_image, name='original_image', encoding='RGB')

    def load_image(self):
        img = Image.open(self.input_image_path)
        self.ori_image = np.array(img)
        self.ori_luminance = np.expand_dims(rgb2lab(self.ori_image)[:, :, 0], axis=2)
        img_array_reshaped = resize(self.ori_image, (self.edge_size, self.edge_size),
                                    mode='constant', anti_aliasing=True)
        return rgb2lab(img_array_reshaped)

    @staticmethod
    def split_image(image):
        luminance = np.expand_dims(image[:, :, 0], axis=2)
        chromaticity = resize((image[:, :, 1:]), (64, 64), mode='constant', anti_aliasing=True)
        return luminance, chromaticity

    def load_model(self):
        self.model = load_model(self.model_path)

    def predict_chromaticity(self, luminance):
        luminance_expanded = np.expand_dims(luminance, axis=0)
        q_chroma = self.model.predict(luminance_expanded, batch_size=1)[0]
        chromaticity = mappings.h(q_chroma, temp=self.temperature)
        return chromaticity

    def merge_and_resize_image(self, chromaticity):
        h, w, _ = self.ori_image.shape
        upsampled_chroma = resize(chromaticity, (h, w), mode='constant', anti_aliasing=True)
        image = np.concatenate((self.ori_luminance, upsampled_chroma), axis=2)
        return image

    @staticmethod
    def show_image(img_array, title=None, encoding='RGB_norm'):
        if encoding == 'LAB':
            Image.fromarray(np.uint8(lab2rgb(img_array) * 255)).show(title=title)
        elif encoding == 'RGB_norm':
            Image.fromarray(np.uint8(img_array * 255)).show(title=title)
        elif encoding == 'RGB':
            Image.fromarray(img_array).show(title=title)

    def save_image(self, image, name, encoding='LAB'):
        if encoding == 'LAB':
            png.from_array(np.uint8(lab2rgb(image) * 255), 'RGB') \
                .save(os.path.join(self.output_path, '{}.png'.format(name)))
        elif encoding == 'RGB_norm':
            png.from_array(np.uint8(image * 255), 'RGB').save(os.path.join(self.output_path, '{}.png'.format(name)))
        elif encoding == 'RGB':
            png.from_array(image, 'RGB').save(os.path.join(self.output_path, '{}.png'.format(name)))


if __name__ == '__main__':
    evaluation_class = Evaluation(autotest=False,
                                  temperature=0.38)
    evaluation_class.eval()
