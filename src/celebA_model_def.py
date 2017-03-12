"""Model definitions for celebA

This file is partially based on
https://github.com/carpedm20/DCGAN-tensorflow/blob/master/main.py
https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py

They come with the following license: https://github.com/carpedm20/DCGAN-tensorflow/blob/master/LICENSE
"""

# pylint: disable = C0103

import numpy as np
import tensorflow as tf
import dcgan_model
import dcgan_ops

tf.app.flags.DEFINE_integer("m", 100, "Measurements [100]")
tf.app.flags.DEFINE_integer("nIter", 100, "Update steps[100]")
tf.app.flags.DEFINE_float("snr", 0.01, "Noise energy[0.01]")
tf.app.flags.DEFINE_float("lam", None, "Regularisation[None]")
tf.app.flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
tf.app.flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
tf.app.flags.DEFINE_integer("image_size", 108,
                            "The size of image to use (will be center cropped) [108]")
tf.app.flags.DEFINE_integer("output_size", 64, "The size of the output images to produce [64]")
tf.app.flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
tf.app.flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
tf.app.flags.DEFINE_string("checkpoint_dir", "../models/",
                           "Directory name to save the checkpoints [checkpoint]")
tf.app.flags.DEFINE_string("sample_dir", "samples",
                           "Directory name to save the image samples [samples]")
tf.app.flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
tf.app.flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
FLAGS = tf.app.flags.FLAGS


def dcgan_discrim(x_hat_batch, sess, hparams):
    dcgan = dcgan_model.DCGAN(sess,
                              image_size=FLAGS.image_size,
                              batch_size=FLAGS.batch_size,
                              output_size=FLAGS.output_size,
                              c_dim=FLAGS.c_dim,
                              dataset_name=FLAGS.dataset,
                              is_crop=FLAGS.is_crop,
                              checkpoint_dir=FLAGS.checkpoint_dir,
                              sample_dir=FLAGS.sample_dir)

    assert hparams.batch_size in [1, 64], 'batch size should be either 64 or 1'

    x_hat_image = tf.reshape(x_hat_batch, [-1, 64, 64, 3])
    all_zeros = tf.zeros([64, 64, 64, 3])
    discrim_input = all_zeros + x_hat_image
    prob, _ = dcgan.discriminator(discrim_input, is_train=False)

    restore_vars = ['d_bn1/beta',
                    'd_bn1/gamma',
                    'd_bn1/moving_mean',
                    'd_bn1/moving_variance',
                    'd_bn2/beta',
                    'd_bn2/gamma',
                    'd_bn2/moving_mean',
                    'd_bn2/moving_variance',
                    'd_bn3/beta',
                    'd_bn3/gamma',
                    'd_bn3/moving_mean',
                    'd_bn3/moving_variance',
                    'd_h0_conv/biases',
                    'd_h0_conv/w',
                    'd_h1_conv/biases',
                    'd_h1_conv/w',
                    'd_h2_conv/biases',
                    'd_h2_conv/w',
                    'd_h3_conv/biases',
                    'd_h3_conv/w',
                    'd_h3_lin/Matrix',
                    'd_h3_lin/bias']

    restore_dict = {var.op.name: var for var in tf.all_variables() if var.op.name in restore_vars}
    restore_path = tf.train.latest_checkpoint('../models/celebA_64_64/')

    prob = tf.reshape(prob, [-1])
    return prob[:hparams.batch_size], restore_dict, restore_path



def dcgan_gen(z, sess, hparams):

    assert hparams.batch_size in [1, 64], 'batch size should be either 64 or 1'
    z_full = tf.zeros([64, 100]) + z

    dcgan = dcgan_model.DCGAN(sess,
                              image_size=FLAGS.image_size,
                              batch_size=FLAGS.batch_size,
                              output_size=FLAGS.output_size,
                              c_dim=FLAGS.c_dim,
                              dataset_name=FLAGS.dataset,
                              is_crop=FLAGS.is_crop,
                              checkpoint_dir=FLAGS.checkpoint_dir,
                              sample_dir=FLAGS.sample_dir)

    tf.get_variable_scope().reuse_variables()

    s = dcgan.output_size
    s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

    # project `z` and reshape
    h0 = tf.reshape(dcgan_ops.linear(z_full, dcgan.gf_dim*8*s16*s16, 'g_h0_lin'),
                    [-1, s16, s16, dcgan.gf_dim * 8])
    h0 = tf.nn.relu(dcgan.g_bn0(h0, train=False))

    h1 = dcgan_ops.deconv2d(h0, [dcgan.batch_size, s8, s8, dcgan.gf_dim*4], name='g_h1')
    h1 = tf.nn.relu(dcgan.g_bn1(h1, train=False))

    h2 = dcgan_ops.deconv2d(h1, [dcgan.batch_size, s4, s4, dcgan.gf_dim*2], name='g_h2')
    h2 = tf.nn.relu(dcgan.g_bn2(h2, train=False))

    h3 = dcgan_ops.deconv2d(h2, [dcgan.batch_size, s2, s2, dcgan.gf_dim*1], name='g_h3')
    h3 = tf.nn.relu(dcgan.g_bn3(h3, train=False))

    h4 = dcgan_ops.deconv2d(h3, [dcgan.batch_size, s, s, dcgan.c_dim], name='g_h4')

    x_hat_full = tf.nn.tanh(h4)
    x_hat_batch = tf.reshape(x_hat_full[:hparams.batch_size], [hparams.batch_size, 64*64*3])

    restore_vars = ['g_bn0/beta',
                    'g_bn0/gamma',
                    'g_bn0/moving_mean',
                    'g_bn0/moving_variance',
                    'g_bn1/beta',
                    'g_bn1/gamma',
                    'g_bn1/moving_mean',
                    'g_bn1/moving_variance',
                    'g_bn2/beta',
                    'g_bn2/gamma',
                    'g_bn2/moving_mean',
                    'g_bn2/moving_variance',
                    'g_bn3/beta',
                    'g_bn3/gamma',
                    'g_bn3/moving_mean',
                    'g_bn3/moving_variance',
                    'g_h0_lin/Matrix',
                    'g_h0_lin/bias',
                    'g_h1/biases',
                    'g_h1/w',
                    'g_h2/biases',
                    'g_h2/w',
                    'g_h3/biases',
                    'g_h3/w',
                    'g_h4/biases',
                    'g_h4/w']

    restore_dict = {var.op.name: var for var in tf.all_variables() if var.op.name in restore_vars}
    restore_path = tf.train.latest_checkpoint('../models/celebA_64_64/')

    return x_hat_batch, restore_dict, restore_path
