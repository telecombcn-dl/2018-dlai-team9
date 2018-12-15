# This file evaluates the model.
import os
import png
import sys
import glob
import argparse
import numpy as np
from PIL import Image
from skimage.transform import resize
from skimage.color import rgb2lab, lab2rgb
from keras.models import load_model
from utils import mappings, helpers


class Evaluation(object):
    def __init__(self, autotest=False, show_images=False, temperature=0.38):
        self.temperature = temperature
        self.show_images = show_images
        self.edge_size = 256
        self.autotest = autotest
        self.ori_image = None
        self.ori_luminance = None
        args = self.parse_arguments()
        self.input_image_path = args.input_image
        self.input_path = args.input_path
        self.model_path = args.model_path
        self.model = None
        self.outputs_folder = 'outputs/'
        self.output_path = os.path.join(self.outputs_folder, self.model_path.split('/')[-1][:-3])
        if not os.path.exists(self.outputs_folder):
            os.mkdir(self.outputs_folder)
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        if not self.autotest:
            self.model = load_model(self.model_path)

    @staticmethod
    def parse_arguments():
        parser = argparse.ArgumentParser(description='Evaluation inputs')
        parser.add_argument('-i', '--input_image', type=str,
                            help='Black and white image to be colorized',
                            default='/home/rimmek/MATT/DLAI/2018-dlai-team9/data/test_images/dlai-flowers'
                                    '/oxford8189/image_04793.jpg')
        parser.add_argument('-ip', '--input_path', type=str,
                            help='Black and white image to be colorized',
                            default='/home/rimmek/MATT/DLAI/2018-dlai-team9/inputs/')
        parser.add_argument('-m', '--model_path', type=str, help='Trained model',
                            default='/home/rimmek/MATT/DLAI/2018-dlai-team9/models/flowers_mse_lr_005.h5')

        return parser.parse_args()

    def eval(self):
        input_paths = list()
        if self.input_path:
            input_paths = glob.glob(self.input_path + '*')
        elif self.input_image_path:
            input_paths.append(self.input_image_path)
        else:
            print('ERROR: You have to specify either an input image or an inputs path')
            sys.exit()
        for path in input_paths:
            print(path)
            lab_resized_image = self.load_image(path)
            luminance, chromaticity = self.split_image(lab_resized_image)

            if not self.autotest:
                predicted_chromaticity = self.predict_chromaticity(luminance)
            else:
                predicted_chromaticity = mappings.h(mappings.inverse_h(chromaticity), temp=self.temperature)

            pred_image = self.merge_and_resize_image(predicted_chromaticity)
            if self.show_images:
                helpers.show_image(pred_image, encoding='LAB')
                helpers.show_image(self.ori_image, encoding='RGB')
            self.save_image(pred_image, name=path.split('/')[-1])

    def load_image(self, path):
        img = Image.open(path)
        self.ori_image = np.array(img)
        if self.ori_image.shape[2] > 3:
            self.ori_image = self.ori_image[:, :, :3]
        self.ori_luminance = np.expand_dims(rgb2lab(self.ori_image)[:, :, 0], axis=2)
        img_array_reshaped = resize(self.ori_image, (self.edge_size, self.edge_size),
                                    mode='constant', anti_aliasing=True)
        return rgb2lab(img_array_reshaped)

    @staticmethod
    def split_image(image):
        luminance = np.expand_dims(image[:, :, 0], axis=2)
        chromaticity = resize((image[:, :, 1:]), (64, 64), mode='constant', anti_aliasing=True)
        return luminance, chromaticity

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
                                  show_images=True,
                                  temperature=0.38)
    evaluation_class.eval()
