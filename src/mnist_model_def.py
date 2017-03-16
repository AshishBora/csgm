"""Model definitions for MNIST"""
# pylint: disable = C0301, C0103

import tensorflow as tf

def vae_gen(num_images):
    """Definition of the generator"""

    n_z = 20
    n_hidden_gener_1 = 500
    n_hidden_gener_2 = 500
    n_input = 28 * 28
    z = tf.Variable(tf.random_normal((num_images, n_z)), name='z')

    with tf.variable_scope('generator'):
        weights1 = tf.get_variable('w1', shape=[n_z, n_hidden_gener_1])
        bias1 = tf.Variable(tf.zeros([n_hidden_gener_1], dtype=tf.float32), name='b1')
        hidden1 = tf.nn.softplus(tf.matmul(z, weights1) + bias1, name='h1')

        weights2 = tf.get_variable('w2', shape=[n_hidden_gener_1, n_hidden_gener_2])
        bias2 = tf.Variable(tf.zeros([n_hidden_gener_2], dtype=tf.float32), name='b2')
        hidden2 = tf.nn.softplus(tf.matmul(hidden1, weights2) + bias2, name='h2')

        w_out = tf.get_variable('w_out', shape=[n_hidden_gener_2, n_input])
        b_out = tf.Variable(tf.zeros([n_input], dtype=tf.float32), name='b_out')
        x_hat = tf.nn.sigmoid(tf.matmul(hidden2, w_out) + b_out, name='x_hat')

    restore_path = './models/mnist/model2.ckpt'
    restore_dict = {'Variable_7': weights1,
                    'Variable_8': weights2,
                    'Variable_9': w_out,
                    'Variable_11': bias1,
                    'Variable_12': bias2,
                    'Variable_13': b_out}

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
