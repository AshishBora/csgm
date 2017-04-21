"""Model definitions for MNIST"""
# pylint: disable = C0301, C0103, R0914, C0111

import os
import sys
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from mnist_vae.src import model_def as mnist_vae_model_def
from mnist_e2e.model_def import end_to_end


def construct_gen(hparams, model_def):

    model_hparams = model_def.Hparams()

    z = model_def.get_z_var(model_hparams, hparams.batch_size)
    _, x_hat = model_def.generator(model_hparams, z, 'gen', reuse=False)

    restore_vars = model_def.gen_restore_vars()
    restore_dict = {var.op.name: var for var in tf.global_variables() if var.op.name in restore_vars}
    restore_path = tf.train.latest_checkpoint(hparams.pretrained_model_dir)

    return z, x_hat, restore_path, restore_dict


def vae_gen(hparams):
    return construct_gen(hparams, mnist_vae_model_def)
