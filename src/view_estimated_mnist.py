"""View estimated images for mnist"""
# pylint: disable = C0301, R0903

import numpy as np
import mnist_input
from mnist_utils import view_image
import utils
import matplotlib.pyplot as plt


class Hparams(object):
    """Hyperparameters"""
    def __init__(self):
        self.input_type = 'full-input'
        self.num_input_images = 30
        self.image_matrix = 0
        self.image_shape = (28, 28, 1)
        self.n_input = np.prod(self.image_shape)


def view(xs_dict, patterns, images_nums, hparams, **kws):
    """View the images"""
    x_hats_dict = {}
    for model_type, pattern in zip(hparams.model_types, patterns):
        outfiles = [pattern.format(i) for i in images_nums]
        x_hats_dict[model_type] = {i: plt.imread(outfile) for i, outfile in enumerate(outfiles)}
    xs_dict_temp = {i : xs_dict[i] for i in images_nums}
    utils.image_matrix(xs_dict_temp, x_hats_dict, view_image, hparams, **kws)


def lasso_vae(hparams, xs_dict, images_nums, is_save):
    """Images for Lasso and VAE"""
    hparams.measurement_type = 'gaussian'
    hparams.model_types = ['Lasso', 'VAE']

    for num_measurements in [10, 25, 50, 100, 200, 300, 400, 500, 750]:
        pattern1 = './estimated/mnist/full-input/gaussian/0.1/' + str(num_measurements) + '/lasso/0.1/{0}.png'
        pattern2 = './estimated/mnist/full-input/gaussian/0.1/' + str(num_measurements) + '/vae/0.0_1.0_0.1_adam_0.01_0.9_False_1000_10/{0}.png'
        patterns = [pattern1, pattern2]
        view(xs_dict, patterns, images_nums, hparams, alg_labels=True)

        base_path = './results/mnist_reconstr_{}_orig_lasso_vae.pdf'
        save_path = base_path.format(num_measurements)
        utils.save_plot(is_save, save_path)


def end_to_end(hparams, xs_dict, images_nums, is_save):
    """Image for End to end models"""
    hparams.measurement_type = 'fixed'
    is_save = True
    hparams.model_types = []
    patterns = []

    base_pattern = './estimated/mnist/full-input/{0}/0.1/{1}/learned/50-200/{2}.png'
    for measurement_type in ['fixed', 'learned']:
        for num_measurements in [10, 20, 30]:
            hparams.model_types.append('{}{}'.format(measurement_type.title(), num_measurements))
            patterns.append(base_pattern.format(measurement_type, num_measurements, '{0}'))

    view(xs_dict, patterns, images_nums, hparams, alg_labels=True)
    save_path = './results/mnist_e2e_orig_fixed_learned.pdf'
    utils.save_plot(is_save, save_path)


def get_image_nums(start, stop, hparams):
    """Get range of images"""
    assert start >= 0
    assert stop <= hparams.num_input_images
    images_nums = list(range(start, stop))
    return images_nums


def main():
    """Make and save image matrices"""

    hparams = Hparams()
    xs_dict = mnist_input.model_input(hparams)
    start, stop = 10, 20
    images_nums = get_image_nums(start, stop, hparams)
    is_save = True
    lasso_vae(hparams, xs_dict, images_nums, is_save)
    end_to_end(hparams, xs_dict, images_nums, is_save)


if __name__ == '__main__':
    main()
