"""Model definitions for MNIST"""
# pylint: disable = C0301, C0103, R0914, C0111

import os
import sys
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'mnist-vae', 'src'))
import main as mnist_vae


def vae_gen(num_images):
    """Definition of the generator"""

    mnist_vae_hparams = mnist_vae.Hparams()
    z = tf.Variable(tf.random_normal((num_images, mnist_vae_hparams.n_z)), name='z')
    _, x_hat = mnist_vae.generator(mnist_vae_hparams, z, 'gen', reuse=False)
    restore_path = tf.train.latest_checkpoint('./mnist-vae/models/mnist-vae/')
    restore_vars = ['gen/w1',
                    'gen/b1',
                    'gen/w2',
                    'gen/b2',
                    'gen/w3',
                    'gen/b3']
    restore_dict = {var.op.name: var for var in tf.global_variables() if var.op.name in restore_vars}
    return z, x_hat, restore_path, restore_dict


def end_to_end(hparams):
    layer_sizes = [50, 200]

    y_batch = tf.placeholder(tf.float32, shape=(hparams.batch_size, hparams.num_measurements), name='y_batch')

    hidden = y_batch
    prev_hidden_size = hparams.num_measurements
    for i, hidden_size in enumerate(layer_sizes):
        layer_name = 'hidden{0}'.format(i)
        with tf.variable_scope(layer_name):
            weights = tf.get_variable('weights', shape=[prev_hidden_size, hidden_size])
            biases = tf.get_variable('biases', initializer=tf.zeros([hidden_size]))
            hidden = tf.nn.relu(tf.matmul(hidden, weights) + biases, name=layer_name)
        prev_hidden_size = hidden_size

    with tf.variable_scope('sigmoid_logits'):
        weights = tf.get_variable('weights', shape=[prev_hidden_size, 784])
        biases = tf.get_variable('biases', initializer=tf.zeros([784]))
        logits = tf.add(tf.matmul(hidden, weights), biases, name='logits')

    x_hat_batch = tf.nn.sigmoid(logits)

    restore_vars = ['hidden0/weights',
                    'hidden0/biases',
                    'hidden1/weights',
                    'hidden1/biases',
                    'sigmoid_logits/weights',
                    'sigmoid_logits/biases']
    restore_dict = {var.op.name: var for var in tf.global_variables() if var.op.name in restore_vars}

    return y_batch, x_hat_batch, restore_dict
