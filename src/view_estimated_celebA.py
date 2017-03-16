"""View estimated images for celebA"""
# pylint: disable = C0301, R0903, R0902

import numpy as np
import celebA_input
from celebA_utils import view_image
import utils
import matplotlib.pyplot as plt


class Hparams(object):
    """Hyperparameters"""
    def __init__(self):
        self.input_type = 'full-input'
        self.input_path_pattern = './data/celebAtest/*.jpg'
        self.num_input_images = 64
        self.image_matrix = 0
        self.image_shape = (64, 64, 3)
        self.n_input = np.prod(self.image_shape)
        self.measurement_type = 'gaussian'
        self.model_types = ['Lasso (DCT)', 'Lasso (Wavelet)', 'DCGAN']


def view(xs_dict, patterns, images_nums, hparams, **kws):
    """View the images"""
    x_hats_dict = {}
    for model_type, pattern in zip(hparams.model_types, patterns):
        outfiles = [pattern.format(i) for i in images_nums]
        x_hats_dict[model_type] = {i: 2*plt.imread(outfile)-1 for i, outfile in enumerate(outfiles)}
    xs_dict_temp = {i : xs_dict[i] for i in images_nums}
    utils.image_matrix(xs_dict_temp, x_hats_dict, view_image, hparams, **kws)


def get_image_nums(start, stop, hparams):
    """Get range of images"""
    assert start >= 0
    assert stop <= hparams.num_input_images
    images_nums = list(range(start, stop))
    return images_nums


def main():
    """Make and save image matrices"""
    hparams = Hparams()
    xs_dict = celebA_input.model_input(hparams)
    start, stop = 20, 30
    images_nums = get_image_nums(start, stop, hparams)
    is_save = True
    for num_measurements in [50, 100, 200, 500, 1000, 2500, 5000, 7500, 10000]:
        pattern1 = './estimated/celebA/full-input/gaussian/0.01/' + str(num_measurements) + '/lasso-dct/0.1/{0}.png'
        pattern2 = './estimated/celebA/full-input/gaussian/0.01/' + str(num_measurements) + '/lasso-wavelet/1e-05/{0}.png'
        pattern3 = './estimated/celebA/full-input/gaussian/0.01/' + str(num_measurements) + '/dcgan/0.0_1.0_0.001_0.0_0.0_adam_0.1_0.9_False_500_10/{0}.png'
        patterns = [pattern1, pattern2, pattern3]
        view(xs_dict, patterns, images_nums, hparams)
        base_path = './results/celebA_reconstr_{}_orig_lasso-dct_lasso-wavelet_dcgan.pdf'
        save_path = base_path.format(num_measurements)
        utils.save_plot(is_save, save_path)

if __name__ == '__main__':
    main()
