"""Some utils for MNIST dataset"""
# pylint: disable=C0301,C0103

import png
import numpy as np
import utils


def display_transform(image):
    image = np.squeeze(image)
    return image


def view_image(image, hparams, mask=None):
    """Process and show the image"""
    image = display_transform(image)
    if len(image) == hparams.n_input:
        image = image.reshape([28, 28])
        if mask is not None:
            mask = mask.reshape([28, 28])
            image = np.maximum(np.minimum(1.0, image - 1.0*(1-mask)), 0.0)
    utils.plot_image(image, 'Greys')


def save_image(image, path):
    """Save an image as a png file"""
    png_writer = png.Writer(28, 28, greyscale=True)
    with open(path, 'wb') as outfile:
        png_writer.write(outfile, 255*image)
