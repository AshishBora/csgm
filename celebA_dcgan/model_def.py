# pylint: disable = C0103, C0111, C0301, R0914

"""Model definitions for celebA

This file is partially based on
https://github.com/carpedm20/DCGAN-tensorflow/blob/master/main.py
https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py

They come with the following license: https://github.com/carpedm20/DCGAN-tensorflow/blob/master/LICENSE
"""

import tensorflow as tf
import ops


class Hparams(object):
    def __init__(self):
        self.c_dim = 3
        self.z_dim = 100
        self.gf_dim = 64
        self.df_dim = 64
        self.gfc_dim = 1024
        self.dfc_dim = 1024
        self.batch_size = 64


def generator(hparams, z, train, reuse):

    if reuse:
        tf.get_variable_scope().reuse_variables()

    output_size = 64
    s = output_size
    s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

    g_bn0 = ops.batch_norm(name='g_bn0')
    g_bn1 = ops.batch_norm(name='g_bn1')
    g_bn2 = ops.batch_norm(name='g_bn2')
    g_bn3 = ops.batch_norm(name='g_bn3')

    # project `z` and reshape
    h0 = tf.reshape(ops.linear(z, hparams.gf_dim*8*s16*s16, 'g_h0_lin'), [-1, s16, s16, hparams.gf_dim * 8])
    h0 = tf.nn.relu(g_bn0(h0, train=train))

    h1 = ops.deconv2d(h0, [hparams.batch_size, s8, s8, hparams.gf_dim*4], name='g_h1')
    h1 = tf.nn.relu(g_bn1(h1, train=train))

    h2 = ops.deconv2d(h1, [hparams.batch_size, s4, s4, hparams.gf_dim*2], name='g_h2')
    h2 = tf.nn.relu(g_bn2(h2, train=train))

    h3 = ops.deconv2d(h2, [hparams.batch_size, s2, s2, hparams.gf_dim*1], name='g_h3')
    h3 = tf.nn.relu(g_bn3(h3, train=train))

    h4 = ops.deconv2d(h3, [hparams.batch_size, s, s, hparams.c_dim], name='g_h4')
    x_gen = tf.nn.tanh(h4)

    return x_gen


def discriminator(hparams, x, train, reuse):

    if reuse:
        tf.get_variable_scope().reuse_variables()

    d_bn1 = ops.batch_norm(name='d_bn1')
    d_bn2 = ops.batch_norm(name='d_bn2')
    d_bn3 = ops.batch_norm(name='d_bn3')

    h0 = ops.lrelu(ops.conv2d(x, hparams.df_dim, name='d_h0_conv'))

    h1 = ops.conv2d(h0, hparams.df_dim*2, name='d_h1_conv')
    h1 = ops.lrelu(d_bn1(h1, train=train))

    h2 = ops.conv2d(h1, hparams.df_dim*4, name='d_h2_conv')
    h2 = ops.lrelu(d_bn2(h2, train=train))

    h3 = ops.conv2d(h2, hparams.df_dim*8, name='d_h3_conv')
    h3 = ops.lrelu(d_bn3(h3, train=train))

    h4 = ops.linear(tf.reshape(h3, [hparams.batch_size, -1]), 1, 'd_h3_lin')

    d_logit = h4
    d = tf.nn.sigmoid(d_logit)

    return d, d_logit


def gen_restore_vars():
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
    return restore_vars



def discrim_restore_vars():
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
    return restore_vars
